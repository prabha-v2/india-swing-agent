import yfinance as yf
import pandas as pd
import ta
import time
import os
import requests
import json
from datetime import datetime, date, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================
# SETTINGS
# =========================================

ACCOUNT_SIZE    = 500000       # Change to your actual capital
RISK_PER_TRADE  = 0.01         # 1.0% risk per trade (reduced from 1.5% — tighter capital protection)
RR_RATIO        = 3.0          # 3:1 reward-to-risk (raised from 2.5 — only high-payoff setups)
MAX_ATR_STOP    = 2.0          # Max 2x ATR for stop loss (tighter than 3.0 — exit losers faster)
MAX_STOP_PCT    = 0.10         # Max 10% stop distance from entry (was 12%)
MAX_POSITION    = 2000
SCORE_THRESHOLD = 17           # Raised from 16 — stricter quality filter
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
# Tightened: EPS>-20, D/E<200, ROE>8%
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
            if eps    is not None and eps    < -20:             result = False   # stricter: was -50
            if de     is not None and de     > 200:             result = False   # stricter: was 300
            if mktcap is not None and mktcap < 5_000_000_000:  result = False
            if rev    is not None and rev    <= 0:              result = False
            if roe    is not None and roe    < 0.08:            # stricter: was 5%
                if "Industrials" not in sector and "Utilities" not in sector and "Financial" not in sector:
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
# Improved: ETF must be above both EMA50 and EMA200
# =========================================

def sector_is_strong(etf_symbol):
    try:
        df = yf.download(etf_symbol, period="1y", interval="1d", progress=False)
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 50:
            return True
        df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
        df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
        close  = float(df['Close'].iloc[-1])
        ema50  = float(df['EMA50'].iloc[-1])
        # Only require EMA200 if we have enough data
        if len(df) >= 200:
            ema200 = float(df['EMA200'].iloc[-1])
            return close > ema50 and close > ema200
        return close > ema50
    except Exception:
        return True

# =========================================
# TECHNICAL SCANNER
# Improvements:
#   - ADX filter (trend strength, avoids choppy markets)
#   - MACD confirmation (trend direction signal)
#   - ROC-21 momentum filter
#   - Combined breakout+volume bonus
#   - Tighter stop loss (2x ATR, 10% max)
#   - Improved RSI logic (penalise weak <40 and overbought >75)
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

        # Liquidity — ₹2 Crore+ daily turnover
        avg_turnover = float(
            df['Close'].iloc[-20:].mean() * df['Volume'].iloc[-20:].mean()
        )
        if avg_turnover < 20_000_000:
            return None

        # ---- INDICATORS ----
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

        # ADX — trend strength (avoids false signals in ranging/choppy markets)
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

        # MACD — trend confirmation
        df['MACD']   = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
        df['MACD_S'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)

        # ROC-21 — 21-day Rate of Change (momentum check)
        df['ROC21']  = df['Close'].pct_change(21)

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

        # ---- RSI SIGNALS (improved) ----
        rsi      = float(latest['RSI'])
        prev_rsi = float(prev['RSI'])
        if rsi < 40:                                              score -= 2  # weak momentum — avoid
        if prev_rsi < 50 and rsi > 50:                           score += 2  # fresh RSI cross above midline
        elif rsi > 55:                                            score += 1  # sustained above midline
        if 55 < rsi < 75:                                         score += 1  # healthy sweet spot
        if rsi > 80:                                              score -= 2  # very overbought
        elif rsi > 75:                                            score -= 1  # mildly overbought

        # ---- BREAKOUT SIGNALS ----
        broke_20d = latest['Close'] > prev['HH20']
        if broke_20d:                                             score += 2  # 20-day breakout
        ath_dist = latest['Close'] / latest['High52']
        if ath_dist > 0.90:                                       score += 2  # within 10% of 52W high
        elif ath_dist > 0.80:                                     score += 1  # within 20%

        # ---- VOLUME SIGNALS ----
        rvol = 0
        if latest['AvgVol'] > 0:
            rvol = latest['Volume'] / latest['AvgVol']
            if rvol > 2.0:                                        score += 2  # strong volume surge
            elif rvol > 1.5:                                      score += 1  # moderate surge

        # ---- BREAKOUT + VOLUME CONFIRMATION (bonus) ----
        if broke_20d and rvol > 1.5:                              score += 1  # volume-confirmed breakout

        # ---- ADX — TREND STRENGTH (new) ----
        adx = float(latest['ADX'])
        if adx > 30:                                              score += 2  # very strong trend
        elif adx > 20:                                            score += 1  # moderate trend
        if adx < 15:                                              score -= 2  # choppy/ranging — high risk

        # ---- MACD CONFIRMATION (new) ----
        macd    = float(latest['MACD'])
        macd_s  = float(latest['MACD_S'])
        p_macd  = float(prev['MACD'])
        p_macd_s = float(prev['MACD_S'])
        if p_macd < p_macd_s and macd > macd_s:                  score += 2  # fresh MACD bullish crossover
        elif macd > macd_s:                                       score += 1  # MACD above signal line
        if macd > 0:                                              score += 1  # MACD above zero line (uptrend)

        # ---- ROC-21 MOMENTUM (new) ----
        roc21 = float(latest['ROC21'])
        if roc21 > 0.15:                                          score += 1  # strong 21-day momentum
        if roc21 < 0:                                             score -= 1  # negative 21-day momentum

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

        if risk <= 0 or risk > entry * MAX_STOP_PCT:   # max 10% stop
            return None

        risk_amt  = ACCOUNT_SIZE * RISK_PER_TRADE
        qty       = min(int(risk_amt / risk), MAX_POSITION)

        if qty <= 0:
            return None

        target   = entry + (risk * RR_RATIO)
        invested = round(entry * qty, 0)

        return {
            "Symbol":    symbol.replace(".NS", ""),
            "Theme":     sector_map.get(symbol, "OTHER"),
            "Score":     score,
            "Entry":     round(entry, 2),
            "Stop":      round(float(stop), 2),
            "Target":    round(float(target), 2),
            "StopPct":   round((entry - float(stop)) / entry * 100, 1),
            "TargetPct": round((float(target) - entry) / entry * 100, 1),
            "Qty":       qty,
            "Invested":  invested,
            "Risk₹":     round(risk * qty, 0),
            "Reward₹":   round((float(target) - entry) * qty, 0),
            "ADX":       round(adx, 1),
        }

    except Exception as e:
        print(f"  ⚠️ Error — {symbol}: {e}")
        return None

# =========================================
# STOCK UNIVERSE — 150 NSE stocks
# Expanded: added FMCG, Cement, Insurance,
# Electrical/Consumer, Agrochem themes
# =========================================

sector_map = {

    # ---- DEFENCE PSU ----
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
    "CUMMINSIND.NS": "INFRA",
    "APLAPOLLO.NS":  "INFRA",

    # ---- PSU BANKS ----
    "SBIN.NS":       "PSU BANK",
    "PNB.NS":        "PSU BANK",
    "BANKBARODA.NS": "PSU BANK",
    "CANBK.NS":      "PSU BANK",
    "UNIONBANK.NS":  "PSU BANK",
    "INDIANB.NS":    "PSU BANK",

    # ---- PRIVATE BANKS & NBFC ----
    "HDFCBANK.NS":    "BANK",
    "ICICIBANK.NS":   "BANK",
    "AXISBANK.NS":    "BANK",
    "KOTAKBANK.NS":   "BANK",
    "INDUSINDBK.NS":  "BANK",
    "FEDERALBNK.NS":  "BANK",
    "IDFCFIRSTB.NS":  "BANK",
    "BAJFINANCE.NS":  "FINTECH",
    "BAJAJFINSV.NS":  "FINTECH",
    "CHOLAFIN.NS":    "FINTECH",
    "MUTHOOTFIN.NS":  "FINTECH",

    # ---- IT & AI ----
    "TCS.NS":         "IT",
    "INFY.NS":        "IT",
    "WIPRO.NS":       "IT",
    "HCLTECH.NS":     "IT",
    "TECHM.NS":       "IT",
    "PERSISTENT.NS":  "IT",
    "COFORGE.NS":     "IT",
    "MPHASIS.NS":     "IT",
    "LTIM.NS":        "IT",
    "KPITTECH.NS":    "IT",
    "TATAELXSI.NS":   "IT",
    "CYIENT.NS":      "IT",

    # ---- RENEWABLES / GREEN ENERGY ----
    "ADANIGREEN.NS":  "RENEW",
    "TATAPOWER.NS":   "RENEW",
    "NTPC.NS":        "RENEW",
    "SJVN.NS":        "RENEW",
    "NHPC.NS":        "RENEW",
    "SUZLON.NS":      "RENEW",
    "INOXWIND.NS":    "RENEW",
    "WAAREEENER.NS":  "RENEW",
    "JSWENERGY.NS":   "RENEW",
    "TORNTPOWER.NS":  "RENEW",

    # ---- EV & AUTO ----
    "TATAMOTORS.NS":  "EV",
    "M&M.NS":         "EV",
    "OLECTRA.NS":     "EV",
    "TVSMOTOR.NS":    "AUTO",
    "BAJAJ-AUTO.NS":  "AUTO",
    "HEROMOTOCO.NS":  "AUTO",
    "EICHERMOT.NS":   "AUTO",
    "MOTHERSON.NS":   "AUTO",
    "AMARAJABAT.NS":  "AUTO",
    "ESCORTS.NS":     "AUTO",

    # ---- CAPITAL MARKETS ----
    "BSE.NS":         "CAP MKT",
    "CDSL.NS":        "CAP MKT",
    "ANGELONE.NS":    "CAP MKT",
    "MCX.NS":         "CAP MKT",
    "MOFSL.NS":       "CAP MKT",
    "360ONE.NS":      "CAP MKT",
    "NUVAMA.NS":      "CAP MKT",

    # ---- CONSUMPTION & RETAIL ----
    "TITAN.NS":       "CONSUMP",
    "DMART.NS":       "CONSUMP",
    "TRENT.NS":       "CONSUMP",
    "NYKAA.NS":       "CONSUMP",
    "ZOMATO.NS":      "CONSUMP",
    "DEVYANI.NS":     "CONSUMP",
    "SAPPHIRE.NS":    "CONSUMP",

    # ---- FMCG (new theme) ----
    "HINDUNILVR.NS":  "FMCG",
    "BRITANNIA.NS":   "FMCG",
    "NESTLEIND.NS":   "FMCG",
    "MARICO.NS":      "FMCG",
    "DABUR.NS":       "FMCG",
    "GODREJCP.NS":    "FMCG",

    # ---- PHARMA & HEALTHCARE ----
    "SUNPHARMA.NS":   "PHARMA",
    "DRREDDY.NS":     "PHARMA",
    "CIPLA.NS":       "PHARMA",
    "DIVISLAB.NS":    "PHARMA",
    "MANKIND.NS":     "PHARMA",
    "APOLLOHOSP.NS":  "PHARMA",
    "FORTIS.NS":      "PHARMA",
    "MAXHEALTH.NS":   "PHARMA",
    "ZYDUSLIFE.NS":   "PHARMA",
    "TORNTPHARM.NS":  "PHARMA",
    "ALKEM.NS":       "PHARMA",

    # ---- CHEMICALS & SPECIALTY ----
    "PIDILITIND.NS":  "CHEM",
    "ATUL.NS":        "CHEM",
    "NAVINFLUOR.NS":  "CHEM",
    "CLEAN.NS":       "CHEM",
    "ROSSARI.NS":     "CHEM",
    "TATACHEM.NS":    "CHEM",
    "DEEPAKNTR.NS":   "CHEM",
    "FINEORG.NS":     "CHEM",

    # ---- AGROCHEM (new theme) ----
    "PIIND.NS":       "AGROCHEM",
    "COROMANDEL.NS":  "AGROCHEM",
    "RALLIS.NS":      "AGROCHEM",

    # ---- METALS & MINING ----
    "TATASTEEL.NS":   "METAL",
    "JSWSTEEL.NS":    "METAL",
    "HINDALCO.NS":    "METAL",
    "COALINDIA.NS":   "METAL",
    "NMDC.NS":        "METAL",
    "SAIL.NS":        "METAL",

    # ---- ELECTRONICS MFG (PLI theme) ----
    "DIXON.NS":       "ELECTRONICS",
    "AMBER.NS":       "ELECTRONICS",
    "KAYNES.NS":      "ELECTRONICS",
    "SYRMA.NS":       "ELECTRONICS",
    "PGEL.NS":        "ELECTRONICS",
    "AVALON.NS":      "ELECTRONICS",

    # ---- ELECTRICAL & CONSUMER DURABLE (new theme) ----
    "HAVELLS.NS":     "ELECTRICAL",
    "POLYCAB.NS":     "ELECTRICAL",
    "VOLTAS.NS":      "ELECTRICAL",
    "CROMPTON.NS":    "ELECTRICAL",
    "BLUESTARCO.NS":  "ELECTRICAL",

    # ---- CEMENT (new theme) ----
    "ULTRACEMCO.NS":  "CEMENT",
    "SHREECEM.NS":    "CEMENT",
    "AMBUJACEM.NS":   "CEMENT",
    "JKCEMENT.NS":    "CEMENT",

    # ---- INSURANCE (new theme) ----
    "HDFCLIFE.NS":    "INSURANCE",
    "ICICIPRULI.NS":  "INSURANCE",
    "SBILIFE.NS":     "INSURANCE",
    "STARHEALTH.NS":  "INSURANCE",

    # ---- REAL ESTATE ----
    "DLF.NS":         "REALTY",
    "GODREJPROP.NS":  "REALTY",
    "OBEROIRLTY.NS":  "REALTY",
    "PRESTIGE.NS":    "REALTY",
    "BRIGADE.NS":     "REALTY",
    "PHOENIXLTD.NS":  "REALTY",

    # ---- LOGISTICS (new theme) ----
    "CONCOR.NS":      "LOGISTICS",
    "BLUEDART.NS":    "LOGISTICS",
    "MAHLOG.NS":      "LOGISTICS",

    # ---- TELECOM & DIGITAL ----
    "BHARTIARTL.NS":  "TELECOM",
    "HFCL.NS":        "TELECOM",
    "INDUSTOWER.NS":  "TELECOM",
}

stocks = list(sector_map.keys())

# Sector ETFs for strength check — expanded
sector_etf_map = {
    "PSU BANK": "PSUBNKBEES.NS",
    "BANK":     "BANKBEES.NS",
    "IT":       "ITBEES.NS",
    "PHARMA":   "PHARMABEES.NS",
    "FMCG":     "NIFTYBEES.NS",      # proxy — no dedicated FMCG ETF
    "AUTO":     "MOM100.NS",
}

# Themes that skip fundamental filter (PSUs / complex accounting)
SKIP_FUNDAMENTAL = {
    "PSU BANK", "RAILWAYS", "DEFENCE", "RENEW", "METAL",
    "LOGISTICS", "CEMENT",
}

# =========================================
# MAIN SCANNER
# =========================================

def run_agent():

    print(f"\n{'='*55}")
    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(IST)
    print(f"🇮🇳 India Scan — {now_ist.strftime('%d %b %Y %H:%M:%S')} IST")
    print(f"{'='*55}")

    if not market_is_bullish():
        msg = (
            f"📉 NIFTY weak — staying in cash\n"
            f"Market below EMA50 or EMA200\n"
            f"{datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%d %b %Y %H:%M')} IST"
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
                f" score:{result['Score']}  ADX:{result['ADX']}"
                f"  entry:₹{result['Entry']}"
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
            f"🔍 India Scan — {datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%d %b %Y %H:%M')} IST\n"
            f"✅ NIFTY bullish but no high-quality setups found.\n"
            f"All signals checked — wait for next scan."
        )
        return

    # Sort: score first, then reward
    picks = sorted(picks, key=lambda x: (x['Score'], x['Reward₹']), reverse=True)

    # Summary
    send_telegram(
        f"📊 INDIA SCAN — {datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%d %b %Y %H:%M')} IST\n"
        f"✅ NIFTY bullish (above EMA50 + EMA200)\n"
        f"🔍 Scanned {checked} stocks\n"
        f"🎯 {len(picks)} high-quality setup(s) found\n"
        f"Top {min(TOP_PICKS, len(picks))} picks below ↓\n"
        f"Trade what suits you — alerts only."
    )

    # Individual alerts — now show % stop and % to target
    for pick in picks[:TOP_PICKS]:
        rr = round(pick['Reward₹'] / pick['Risk₹'], 1) if pick['Risk₹'] > 0 else 0
        msg = (
            f"{'='*32}\n"
            f"🚀 {pick['Symbol']}  [{pick['Theme']}]\n"
            f"Score    : {pick['Score']}/40  ADX:{pick['ADX']}\n"
            f"Entry    : ₹{pick['Entry']}\n"
            f"Stop     : ₹{pick['Stop']} (-{pick['StopPct']}%)\n"
            f"Target   : ₹{pick['Target']} (+{pick['TargetPct']}%)\n"
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
