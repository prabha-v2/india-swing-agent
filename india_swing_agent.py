import yfinance as yf
import pandas as pd
import ta
import time
import os
import requests
import json
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================
# SETTINGS
# =========================================

ACCOUNT_SIZE    = 500000       # Change to your actual capital
RISK_PER_TRADE  = 0.015        # 1.5% risk per trade
RR_RATIO        = 2.5
MAX_ATR_STOP    = 3.0
MAX_POSITION    = 2000
SCORE_THRESHOLD = 16           # Raised from 12 to 16 — stricter = better quality
TOP_PICKS       = 6

TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID         = os.environ.get("CHAT_ID", "")

CACHE_FILE = "/tmp/india_agent_cache.json"

def load_cache():
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            if cache.get("date") == str(date.today()):
                return cache
    except Exception:
        pass
    return {}

def save_cache(data):
    try:
        data["date"] = str(date.today())
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

# =========================================
# FUNDAMENTAL FILTERS
# These remove weak companies before
# technical scanning even begins
# =========================================

def passes_fundamental_filter(symbol, theme):
    if theme in SKIP_FUNDAMENTAL:
        return True
    cache      = load_cache()
    fund_cache = cache.get("fundamentals", {})
    if symbol in fund_cache:
        return fund_cache[symbol]
    result = True
    try:
        info = yf.Ticker(symbol).info
        if info:
            eps    = info.get("trailingEps", None)
            de     = info.get("debtToEquity", None)
            mktcap = info.get("marketCap", None)
            rev    = info.get("totalRevenue", None)
            roe    = info.get("returnOnEquity", None)
            sector = info.get("sector", "")
            if eps    is not None and eps    < -50:            result = False
            if de     is not None and de     > 300:            result = False
            if mktcap is not None and mktcap < 5_000_000_000: result = False
            if rev    is not None and rev    <= 0:             result = False
            if roe    is not None and roe    < 0.05:
                if "Industrials" not in sector and "Utilities" not in sector:
                    result = False
    except Exception:
        result = True
    cache = load_cache()
    fund_cache = cache.get("fundamentals", {})
    fund_cache[symbol] = result
    cache["fundamentals"] = fund_cache
    save_cache(cache)
    return result
# =========================================
# TELEGRAM
# =========================================

def send_telegram(msg):
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(
            url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10
        )
        if not resp.ok:
            print(f"⚠️ Telegram error: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")

# =========================================
# MARKET FILTER — NIFTY 50
# =========================================

def market_is_bullish():
    df = yf.download("^NSEI", period="1y", interval="1d", progress=False)
    df = df.dropna()
    df.columns = df.columns.get_level_values(0)
    if df.empty:
        return False
    df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    latest       = df.iloc[-1]
    close        = float(latest['Close'])
    ema50        = float(latest['EMA50'])
    ema200       = float(latest['EMA200'])
    print(f"NIFTY: {close:.0f} | EMA50: {ema50:.0f} | EMA200: {ema200:.0f}")
    # Bullish only if above BOTH EMAs
    return close > ema50 and close > ema200

# =========================================
# SECTOR STRENGTH FILTER
# =========================================

def sector_is_strong(etf_symbol):
    try:
        df = yf.download(etf_symbol, period="6mo", interval="1d", progress=False)
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 30:
            return True
        df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
        return float(df['Close'].iloc[-1]) > float(df['EMA50'].iloc[-1])
    except Exception:
        return True

# =========================================
# TECHNICAL SCANNER
# =========================================

def check_stock(symbol, nifty_df):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df is None or df.empty or len(df) < 200:
            return None

        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 200:
            return None

        # Price filter
        price = float(df['Close'].iloc[-1])
        if price < 20.0:
            return None

        # Liquidity — ₹2 Crore+ daily turnover (raised from 1 Cr)
        avg_turnover = float(
            df['Close'].iloc[-20:].mean() * df['Volume'].iloc[-20:].mean()
        )
        if avg_turnover < 20_000_000:
            return None

        # Indicators
        df['EMA10']  = ta.trend.ema_indicator(df['Close'], window=10)
        df['EMA20']  = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
        df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
        df['RSI']    = ta.momentum.rsi(df['Close'], window=14)
        df['AvgVol'] = df['Volume'].rolling(20).mean()
        df['HH20']   = df['High'].rolling(20).max()
        df['High52']  = df['High'].rolling(252).max()
        df['Low52']   = df['Low'].rolling(252).min()

        df['TR'] = (
            df['High'] - df['Low']
        ).combine(abs(df['High'] - df['Close'].shift(1)), max
        ).combine(abs(df['Low']  - df['Close'].shift(1)), max)
        df['ATR'] = df['TR'].rolling(14).mean()

        # Relative strength vs NIFTY
        s3  = df['Close'].pct_change(63).iloc[-1]
        s6  = df['Close'].pct_change(126).iloc[-1]
        s12 = df['Close'].pct_change(252).iloc[-1] if len(df) >= 252 else s6
        n3  = nifty_df['Close'].pct_change(63).iloc[-1]
        n6  = nifty_df['Close'].pct_change(126).iloc[-1]
        n12 = nifty_df['Close'].pct_change(252).iloc[-1]

        rs = sum([s3 > n3, s6 > n6, s12 > n12])

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        score = 0

        # ---- MOMENTUM SIGNALS ----
        if rs >= 2:                                               score += 2
        if rs == 3:                                               score += 1  # bonus for all 3
        if latest['EMA10']  > latest['EMA20']:                   score += 2
        if latest['EMA20']  > latest['EMA50']:                   score += 2
        if latest['EMA50']  > latest['EMA200']:                  score += 2
        if latest['Close']  > latest['EMA50']:                   score += 1
        if latest['Close']  > latest['EMA200']:                  score += 2

        # ---- RSI SIGNALS ----
        rsi = float(latest['RSI'])
        if prev['RSI'] < 50 and rsi > 50:                        score += 2  # RSI cross
        if 50 < rsi < 70:                                         score += 1  # healthy momentum zone
        if rsi > 70:                                              score -= 1  # overbought penalty

        # ---- BREAKOUT SIGNALS ----
        if latest['Close'] > prev['HH20']:                        score += 2  # 20-day breakout
        ath_dist = latest['Close'] / latest['High52']
        if ath_dist > 0.90:                                       score += 2  # within 10% of 52W high
        elif ath_dist > 0.80:                                     score += 1  # within 20%

        # ---- VOLUME SIGNALS ----
        if latest['AvgVol'] > 0:
            rvol = latest['Volume'] / latest['AvgVol']
            if rvol > 2.0:                                        score += 2  # strong volume surge
            elif rvol > 1.5:                                      score += 1  # moderate surge

        # ---- VOLATILITY / ATR ----
        if latest['ATR'] > df['ATR'].iloc[-5]:                    score += 1  # expanding ATR

        # ---- ENTRY QUALITY ----
        dist_ema20 = (latest['Close'] - latest['EMA20']) / latest['EMA20']
        if dist_ema20 < 0.05:                                     score += 2  # tight pullback — best entry
        elif dist_ema20 < 0.08:                                   score += 1  # acceptable pullback
        elif dist_ema20 > 0.15:                                   score -= 1  # too extended penalty

        # ---- 60-DAY OUTPERFORMANCE ----
        if len(df) >= 60 and len(nifty_df) >= 60:
            sr = float(df['Close'].squeeze().pct_change(60).iloc[-1])
            nr = float(nifty_df['Close'].squeeze().pct_change(60).iloc[-1])
            if sr > nr * 1.5:                                     score += 2  # significantly outperforming
            elif sr > nr:                                         score += 1  # outperforming

        # ---- CIRCUIT BREAKER SAFETY ----
        # Avoid stocks that are near circuit limits
        # (already down 15%+ from recent high = distribution)
        recent_high = df['High'].iloc[-10:].max()
        if latest['Close'] < recent_high * 0.85:
            score -= 2  # penalise stocks falling hard recently

        if score < SCORE_THRESHOLD:
            return None

        # ---- POSITION SIZING ----
        entry        = float(latest['Close'])
        atr          = float(latest['ATR'])
        five_bar_low = float(df['Low'].iloc[-5:].min())
        atr_stop     = entry - (MAX_ATR_STOP * atr)
        stop         = max(five_bar_low, atr_stop)
        risk         = entry - stop

        if risk <= 0 or risk > entry * 0.12:   # tightened to 12%
            return None

        risk_amt  = ACCOUNT_SIZE * RISK_PER_TRADE
        qty       = min(int(risk_amt / risk), MAX_POSITION)

        if qty <= 0:
            return None

        target   = entry + (risk * RR_RATIO)
        invested = round(entry * qty, 0)

        return {
            "Symbol":   symbol.replace(".NS", ""),
            "Theme":    sector_map.get(symbol, "OTHER"),
            "Score":    score,
            "Entry":    round(entry, 2),
            "Stop":     round(float(stop), 2),
            "Target":   round(float(target), 2),
            "Qty":      qty,
            "Invested": invested,
            "Risk₹":    round(risk * qty, 0),
            "Reward₹":  round((float(target) - entry) * qty, 0),
        }

    except Exception as e:
        print(f"  ⚠️ Error — {symbol}: {e}")
        return None

# =========================================
# STOCK UNIVERSE — 120 NSE stocks
# Curated across 15 high-conviction themes
# =========================================

sector_map = {

    # ---- DEFENCE PSU (strong govt order visibility) ----
    "HAL.NS":        "DEFENCE",
    "BEL.NS":        "DEFENCE",
    "MTAR.NS":       "DEFENCE",
    "GRSE.NS":       "DEFENCE",
    "COCHINSHIP.NS": "DEFENCE",
    "MAZDOCK.NS":    "DEFENCE",
    "PARAS.NS":      "DEFENCE",
    "DATAPATTNS.NS": "DEFENCE",
    "MIDHANI.NS":    "DEFENCE",
    "BEML.NS":       "DEFENCE",

    # ---- RAILWAYS & INFRA ----
    "IRFC.NS":       "RAILWAYS",
    "RVNL.NS":       "RAILWAYS",
    "IRCON.NS":      "RAILWAYS",
    "RAILTEL.NS":    "RAILWAYS",
    "IRCTC.NS":      "RAILWAYS",
    "TITAGARH.NS":   "RAILWAYS",
    "TEXRAIL.NS":    "RAILWAYS",
    "KNRCON.NS":     "INFRA",
    "LT.NS":         "INFRA",
    "LTTS.NS":       "INFRA",
    "SIEMENS.NS":    "INFRA",
    "ABB.NS":        "INFRA",

    # ---- PSU BANKS ----
    "SBIN.NS":       "PSU BANK",
    "PNB.NS":        "PSU BANK",
    "BANKBARODA.NS": "PSU BANK",
    "CANBK.NS":      "PSU BANK",
    "UNIONBANK.NS":  "PSU BANK",
    "INDIANB.NS":    "PSU BANK",

    # ---- PRIVATE BANKS & NBFC ----
    "HDFCBANK.NS":   "BANK",
    "ICICIBANK.NS":  "BANK",
    "AXISBANK.NS":   "BANK",
    "KOTAKBANK.NS":  "BANK",
    "INDUSINDBK.NS": "BANK",
    "FEDERALBNK.NS": "BANK",
    "BAJFINANCE.NS": "FINTECH",
    "BAJAJFINSV.NS": "FINTECH",
    "CHOLAFIN.NS":   "FINTECH",
    "MUTHOOTFIN.NS": "FINTECH",

    # ---- IT & AI ----
    "TCS.NS":        "IT",
    "INFY.NS":       "IT",
    "WIPRO.NS":      "IT",
    "HCLTECH.NS":    "IT",
    "TECHM.NS":      "IT",
    "PERSISTENT.NS": "IT",
    "COFORGE.NS":    "IT",
    "MPHASIS.NS":    "IT",
    "LTIM.NS":       "IT",
    "KPITTECH.NS":   "IT",
    "TATAELXSI.NS":  "IT",

    # ---- RENEWABLES / GREEN ENERGY ----
    "ADANIGREEN.NS": "RENEW",
    "TATAPOWER.NS":  "RENEW",
    "NTPC.NS":       "RENEW",
    "SJVN.NS":       "RENEW",
    "NHPC.NS":       "RENEW",
    "SUZLON.NS":     "RENEW",
    "INOXWIND.NS":   "RENEW",
    "WAAREEENER.NS": "RENEW",
    "JSWENERGY.NS":  "RENEW",
    "TORNTPOWER.NS": "RENEW",

    # ---- EV & AUTO ----
    "TATAMOTORS.NS": "EV",
    "M&M.NS":        "EV",
    "OLECTRA.NS":    "EV",
    "TVSMOTOR.NS":   "AUTO",
    "BAJAJ-AUTO.NS": "AUTO",
    "HEROMOTOCO.NS": "AUTO",
    "EICHERMOT.NS":  "AUTO",
    "MOTHERSON.NS":  "AUTO",

    # ---- CAPITAL MARKETS ----
    "BSE.NS":        "CAP MKT",
    "CDSL.NS":       "CAP MKT",
    "ANGELONE.NS":   "CAP MKT",
    "MCX.NS":        "CAP MKT",
    "MOFSL.NS":      "CAP MKT",
    "360ONE.NS":     "CAP MKT",
    "NUVAMA.NS":     "CAP MKT",

    # ---- CONSUMPTION & RETAIL ----
    "TITAN.NS":      "CONSUMP",
    "DMART.NS":      "CONSUMP",
    "TRENT.NS":      "CONSUMP",
    "NYKAA.NS":      "CONSUMP",
    "VEDL.NS":       "CONSUMP",
    "ZOMATO.NS":     "CONSUMP",
    "DEVYANI.NS":    "CONSUMP",
    "SAPPHIRE.NS":   "CONSUMP",

    # ---- PHARMA & HEALTHCARE ----
    "SUNPHARMA.NS":  "PHARMA",
    "DRREDDY.NS":    "PHARMA",
    "CIPLA.NS":      "PHARMA",
    "DIVISLAB.NS":   "PHARMA",
    "MANKIND.NS":    "PHARMA",
    "APOLLOHOSP.NS": "PHARMA",
    "FORTIS.NS":     "PHARMA",
    "MAXHEALTH.NS":  "PHARMA",

    # ---- CHEMICALS & SPECIALTY ----
    "PIDILITIND.NS": "CHEM",
    "ATUL.NS":       "CHEM",
    "NAVINFLUOR.NS": "CHEM",
    "CLEAN.NS":      "CHEM",
    "ROSSARI.NS":    "CHEM",
    "TATACHEM.NS":   "CHEM",

    # ---- METALS & MINING ----
    "TATASTEEL.NS":  "METAL",
    "JSWSTEEL.NS":   "METAL",
    "HINDALCO.NS":   "METAL",
    "COALINDIA.NS":  "METAL",
    "NMDC.NS":       "METAL",
    "SAIL.NS":       "METAL",

    # ---- ELECTRONICS MFG (PLI theme) ----
    "DIXON.NS":      "ELECTRONICS",
    "AMBER.NS":      "ELECTRONICS",
    "KAYNES.NS":     "ELECTRONICS",
    "SYRMA.NS":      "ELECTRONICS",
    "PGEL.NS":       "ELECTRONICS",
    "AVALON.NS":     "ELECTRONICS",

    # ---- REAL ESTATE ----
    "DLF.NS":        "REALTY",
    "GODREJPROP.NS": "REALTY",
    "OBEROIRLTY.NS": "REALTY",
    "PRESTIGE.NS":   "REALTY",
    "BRIGADE.NS":    "REALTY",
    "PHOENIXLTD.NS": "REALTY",

    # ---- TELECOM & DIGITAL ----
    "BHARTIARTL.NS": "TELECOM",
    "HFCL.NS":       "TELECOM",
    "INDUSTOWER.NS": "TELECOM",
}

stocks = list(sector_map.keys())

# Sector ETFs for strength check
sector_etf_map = {
    "PSU BANK":    "PSUBNKBEES.NS",
    "BANK":        "BANKBEES.NS",
    "IT":          "ITBEES.NS",
    "PHARMA":      "PHARMABEES.NS",
}

# Themes that don't need fundamental filter
# (PSUs have complex accounting — skip)
SKIP_FUNDAMENTAL = {
    "PSU BANK", "RAILWAYS", "DEFENCE", "RENEW", "METAL"
}

# =========================================
# MAIN SCANNER
# =========================================

def run_agent():

    print(f"\n{'='*55}")
    print(f"🇮🇳 India Scan — {datetime.now().strftime('%d %b %Y %H:%M:%S')} IST")
    print(f"{'='*55}")

    if not market_is_bullish():
        msg = (
            f"📉 NIFTY weak — staying in cash\n"
            f"Market below EMA50 or EMA200\n"
            f"{datetime.now().strftime('%d %b %Y %H:%M')} IST"
        )
        print(msg)
        send_telegram(msg)
        return

    # Download NIFTY benchmark once
    nifty_df = yf.download("^NSEI", period="1y", interval="1d", progress=False)
    if nifty_df is None or nifty_df.empty:
        print("⚠️ Failed to download NIFTY data")
        return
    nifty_df = nifty_df.dropna()
    nifty_df.columns = nifty_df.columns.get_level_values(0)

    picks   = []
    checked = 0
    skipped_fundamental = 0
    skipped_sector      = 0

    for symbol in stocks:
        time.sleep(0.8)
        checked += 1
        theme = sector_map.get(symbol, "OTHER")

        # Step 1 — Fundamental filter (skip for PSUs)
        if not passes_fundamental_filter(symbol, theme):
                skipped_fundamental += 1
                continue

        # Step 2 — Sector ETF strength filter
        etf = sector_etf_map.get(theme)
        if etf:
            if not sector_is_strong(etf):
                print(f"  {symbol}: sector ETF weak — skip")
                skipped_sector += 1
                continue

        # Step 3 — Technical scan
        result = check_stock(symbol, nifty_df)
        if result:
            print(
                f"  ✅ {result['Symbol']} [{result['Theme']}]"
                f" score:{result['Score']}  "
                f"entry:₹{result['Entry']}"
            )
            picks.append(result)

    print(f"\n{'='*55}")
    print(f"Scanned : {checked} stocks")
    print(f"Filtered: {skipped_fundamental} (fundamentals) "
          f"+ {skipped_sector} (sector)")
    print(f"Qualify : {len(picks)} stocks")
    print(f"{'='*55}")

    if not picks:
        send_telegram(
            f"🔍 India Scan — {datetime.now().strftime('%d %b %Y %H:%M')} IST\n"
            f"✅ NIFTY bullish but no high-quality setups found.\n"
            f"All signals checked — wait for next scan."
        )
        return

    # Sort: score first, then reward
    picks = sorted(picks, key=lambda x: (x['Score'], x['Reward₹']), reverse=True)

    # Summary
    send_telegram(
        f"📊 INDIA SCAN — {datetime.now().strftime('%d %b %Y %H:%M')} IST\n"
        f"✅ NIFTY bullish (above EMA50 + EMA200)\n"
        f"🔍 Scanned {checked} stocks\n"
        f"🎯 {len(picks)} high-quality setup(s) found\n"
        f"Top {min(TOP_PICKS, len(picks))} picks below ↓\n"
        f"Trade what suits you — alerts only."
    )

    # Individual alerts
    for pick in picks[:TOP_PICKS]:
        rr = round(pick['Reward₹'] / pick['Risk₹'], 1) if pick['Risk₹'] > 0 else 0
        msg = (
            f"{'='*32}\n"
            f"🚀 {pick['Symbol']}  [{pick['Theme']}]\n"
            f"Score    : {pick['Score']}/32\n"
            f"Entry    : ₹{pick['Entry']}\n"
            f"Stop     : ₹{pick['Stop']}\n"
            f"Target   : ₹{pick['Target']}\n"
            f"Qty      : {pick['Qty']} shares\n"
            f"Invested : ₹{int(pick['Invested']):,}\n"
            f"Risk     : ₹{int(pick['Risk₹']):,}\n"
            f"Reward   : ₹{int(pick['Reward₹']):,}\n"
            f"RR Ratio : 1:{rr}\n"
            f"{'='*32}"
        )
        send_telegram(msg)
        time.sleep(0.5)

# =========================================
# RUN
# =========================================

def is_market_hours():
    now_utc  = datetime.utcnow()
    if now_utc.weekday() > 4:
        return False
    # NSE: 9:15 AM to 3:30 PM IST = 3:45 to 10:00 UTC
    # We start 1hr early = 2:45 UTC, end 30min after close = 10:30 UTC
    time_val = now_utc.hour * 60 + now_utc.minute
    return (2 * 60 + 45) <= time_val <= (10 * 60 + 30)

if __name__ == "__main__":
    print("🚀 India Professional Swing Trading Agent")
    print(f"Started at {datetime.utcnow().strftime('%H:%M UTC')}")

    if is_market_hours():
        run_agent()
    else:
        print(f"Outside market hours — waiting...")

    while True:
        time.sleep(30 * 60)
        if is_market_hours():
            run_agent()
        else:
            print(f"Market closed — exiting.")
            break

    print("✅ Done — market closed.")
