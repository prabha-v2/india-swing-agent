import yfinance as yf
import pandas as pd
import ta
import time
import os
import requests
from datetime import datetime

# =========================================
# SETTINGS — Indian Market
# =========================================

ACCOUNT_SIZE = 500000          # ₹5 Lakhs default — change as needed
RISK_PER_TRADE = 0.015         # 1.5% risk per trade (slightly aggressive)
RR_RATIO = 2.5                 # Risk:Reward ratio
MAX_ATR_STOP_MULTIPLIER = 3.0  # Stop never more than 3x ATR away
MAX_POSITION_SIZE = 2000       # Max shares per trade
SCORE_THRESHOLD = 12           # Slightly lower = more picks (aggressive mode)
TOP_PICKS = 6                  # Top stocks to alert per scan

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

# =========================================
# TELEGRAM
# =========================================

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        if not resp.ok:
            print(f"⚠️ Telegram error: {resp.status_code} — {resp.text}")
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")

# =========================================
# MARKET TREND FILTER — NIFTY 50
# =========================================

def market_is_bullish():
    # Use NIFTY 50 as market benchmark
    df = yf.download("^NSEI", period="1y", interval="1d", progress=False)
    df = df.dropna()
    df.columns = df.columns.get_level_values(0)
    if df.empty:
        return False
    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
    latest_close = float(df['Close'].iloc[-1])
    latest_ema200 = float(df['EMA200'].iloc[-1])
    latest_ema50  = float(df['EMA50'].iloc[-1])
    print(f"NIFTY Close: {latest_close:.0f} | EMA50: {latest_ema50:.0f} | EMA200: {latest_ema200:.0f}")
    # Aggressive: use EMA50 filter instead of EMA200 (catches trends earlier)
    return latest_close > latest_ema50

# =========================================
# SECTOR STRENGTH FILTER
# =========================================

def sector_is_strong(etf_symbol):
    try:
        df = yf.download(etf_symbol, period="1y", interval="1d", progress=False)
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 50:
            return True  # if no data, don't block the stock
        df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
        return float(df['Close'].iloc[-1]) > float(df['EMA50'].iloc[-1])
    except Exception:
        return True  # fail open

# =========================================
# STOCK SCANNER
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

        # Minimum price filter — skip sub ₹20 stocks
        latest_close_price = float(df['Close'].iloc[-1])
        if latest_close_price < 20.0:
            return None

        # Minimum liquidity — ₹1 Cr+ avg daily turnover
        avg_turnover = float(df['Close'].iloc[-20:].mean() * df['Volume'].iloc[-20:].mean())
        if avg_turnover < 10_000_000:  # ₹1 Crore
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

        df['TR'] = (
            df['High'] - df['Low']
        ).combine(abs(df['High'] - df['Close'].shift(1)), max
        ).combine(abs(df['Low']  - df['Close'].shift(1)), max)
        df['ATR'] = df['TR'].rolling(14).mean()

        # Relative strength vs NIFTY
        stock_3m  = df['Close'].pct_change(63).iloc[-1]
        stock_6m  = df['Close'].pct_change(126).iloc[-1]
        stock_12m = df['Close'].pct_change(252).iloc[-1] if len(df) >= 252 else stock_6m

        nifty_3m  = nifty_df['Close'].pct_change(63).iloc[-1]
        nifty_6m  = nifty_df['Close'].pct_change(126).iloc[-1]
        nifty_12m = nifty_df['Close'].pct_change(252).iloc[-1]

        rs_score = 0
        if stock_3m  > nifty_3m:  rs_score += 1
        if stock_6m  > nifty_6m:  rs_score += 1
        if stock_12m > nifty_12m: rs_score += 1

        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        score = 0

        # Relative strength vs benchmark
        if rs_score >= 2:                                        score += 2

        # EMA trend stack
        if latest['EMA10']  > latest['EMA20']:                   score += 2
        if latest['EMA20']  > latest['EMA50']:                   score += 2
        if latest['EMA50']  > latest['EMA200']:                  score += 2

        # Price above key MAs
        if latest['Close']  > latest['EMA50']:                   score += 2
        if latest['Close']  > latest['EMA200']:                  score += 2

        # RSI momentum cross
        if prev['RSI'] < 50 and latest['RSI'] > 50:              score += 2

        # 20-day high breakout
        if latest['Close']  > prev['HH20']:                      score += 2

        # Near 52-week high (within 15%)
        if latest['Close'] / latest['High52'] > 0.85:            score += 2

        # Volume surge
        if latest['AvgVol'] > 0:
            if latest['Volume'] / latest['AvgVol'] > 1.5:        score += 2

        # ATR expanding (volatility increasing = momentum)
        if latest['ATR'] > df['ATR'].iloc[-5]:                   score += 2

        # Not extended — within 7% of EMA20 (pullback entry)
        dist_ema20 = (latest['Close'] - latest['EMA20']) / latest['EMA20']
        if dist_ema20 < 0.07:                                     score += 2

        # Outperforming NIFTY over 60 days
        if len(df) >= 60 and len(nifty_df) >= 60:
            sr = float(df['Close'].squeeze().pct_change(60).iloc[-1])
            nr = float(nifty_df['Close'].squeeze().pct_change(60).iloc[-1])
            if sr > nr:                                           score += 2

        if score < SCORE_THRESHOLD:
            return None

        entry = float(latest['Close'])
        atr   = float(latest['ATR'])

        five_bar_low = float(df['Low'].iloc[-5:].min())
        atr_stop     = entry - (MAX_ATR_STOP_MULTIPLIER * atr)
        stop         = max(five_bar_low, atr_stop)
        risk         = entry - stop

        if risk <= 0 or risk > entry * 0.15:
            return None

        risk_amount   = ACCOUNT_SIZE * RISK_PER_TRADE
        position_size = min(int(risk_amount / risk), MAX_POSITION_SIZE)

        if position_size <= 0:
            return None

        target = entry + (risk * RR_RATIO)

        return {
            "Symbol":   symbol.replace(".NS", ""),
            "Theme":    sector_map.get(symbol, "OTHER"),
            "Score":    score,
            "Entry":    round(entry, 2),
            "Stop":     round(float(stop), 2),
            "Target":   round(float(target), 2),
            "Size":     position_size,
            "Reward":   round(float(target) - entry, 2),
            "Risk₹":    round(risk * position_size, 0),
            "Reward₹":  round((float(target) - entry) * position_size, 0),
        }

    except Exception as e:
        print(f"  ⚠️ Error checking {symbol}: {e}")
        return None

# =========================================
# STOCK UNIVERSE — 80+ Indian stocks
# NSE tickers need .NS suffix for yfinance
# =========================================

sector_map = {
    # Defence PSU
    "HAL.NS":   "DEFENCE",  "BEL.NS":    "DEFENCE",  "BHEL.NS":  "DEFENCE",
    "MTAR.NS":  "DEFENCE",  "PARAS.NS":  "DEFENCE",  "GRSE.NS":  "DEFENCE",
    "COCHINSHIP.NS": "DEFENCE", "MAZDOCK.NS": "DEFENCE",

    # Railways & Infra
    "IRFC.NS":  "RAILWAYS", "RVNL.NS":   "RAILWAYS", "IRCON.NS": "RAILWAYS",
    "RAILTEL.NS":"RAILWAYS", "TITAGARH.NS":"RAILWAYS", "TEXRAIL.NS":"RAILWAYS",
    "LTIM.NS":  "INFRA",    "LT.NS":     "INFRA",    "IRCTC.NS": "RAILWAYS",

    # PSU Banks
    "SBIN.NS":  "PSU BANK", "PNB.NS":    "PSU BANK", "BANKBARODA.NS":"PSU BANK",
    "CANBK.NS": "PSU BANK", "UNIONBANK.NS":"PSU BANK",

    # Private Banks & Fintech
    "HDFCBANK.NS":"BANK",   "ICICIBANK.NS":"BANK",   "AXISBANK.NS":"BANK",
    "KOTAKBANK.NS":"BANK",  "BAJFINANCE.NS":"FINTECH","PAYTM.NS":  "FINTECH",
    "POLICYBZR.NS":"FINTECH",

    # IT & AI
    "TCS.NS":   "IT",       "INFY.NS":   "IT",       "WIPRO.NS":  "IT",
    "HCLTECH.NS":"IT",      "TECHM.NS":  "IT",       "PERSISTENT.NS":"IT",
    "COFORGE.NS":"IT",      "MPHASIS.NS":"IT",       "LTTS.NS":   "IT",

    # Renewables / Green Energy
    "ADANIGREEN.NS":"RENEW", "TATAPOWER.NS":"RENEW",  "GREENKO.NS":"RENEW",
    "NTPC.NS":  "RENEW",    "SJVN.NS":   "RENEW",    "NHPC.NS":   "RENEW",
    "SUZLON.NS":"RENEW",    "INOXWIND.NS":"RENEW",    "WAAREEENER.NS":"RENEW",

    # EV & Auto
    "TATAMOTORS.NS":"EV",   "M&M.NS":    "EV",       "BAJAJ-AUTO.NS":"AUTO",
    "HEROMOTOCO.NS":"AUTO",  "EICHERMOT.NS":"AUTO",   "OLECTRA.NS":"EV",
    "TVSMOTOR.NS":"AUTO",

    # Consumption & FMCG
    "TITAN.NS":  "CONSUMP",  "DMART.NS":  "CONSUMP",  "TRENT.NS":  "CONSUMP",
    "ABFRL.NS":  "CONSUMP",  "NYKAA.NS":  "CONSUMP",

    # Capital Markets
    "BSE.NS":    "CAP MKT",  "CDSL.NS":   "CAP MKT",  "ANGELONE.NS":"CAP MKT",
    "MCX.NS":    "CAP MKT",  "MOFSL.NS":  "CAP MKT",

    # Pharma / Healthcare
    "SUNPHARMA.NS":"PHARMA", "DRREDDY.NS":"PHARMA",   "CIPLA.NS":  "PHARMA",
    "DIVISLAB.NS":"PHARMA",  "MANKIND.NS":"PHARMA",   "YATHARTH.NS":"PHARMA",

    # Chemicals & Specialty
    "PIDILITIND.NS":"CHEM",  "ATUL.NS":   "CHEM",     "CLEAN.NS":  "CHEM",
    "NAVINFLUOR.NS":"CHEM",

    # Metals & Mining
    "TATASTEEL.NS":"METAL",  "JSWSTEEL.NS":"METAL",   "HINDALCO.NS":"METAL",
    "COALINDIA.NS":"METAL",  "NMDC.NS":   "METAL",

    # Real Estate
    "DLF.NS":    "REALTY",   "GODREJPROP.NS":"REALTY","OBEROIRLTY.NS":"REALTY",
    "PRESTIGE.NS":"REALTY",

    # Telecom & Media
    "BHARTIARTL.NS":"TELECOM","HFCL.NS":  "TELECOM",

    # Semiconductors / Electronics
    "DIXON.NS":  "ELECTRONICS","AMBER.NS": "ELECTRONICS","KAYNES.NS":"ELECTRONICS",
    "SYRMA.NS":  "ELECTRONICS",
}

stocks = list(sector_map.keys())

# Sector ETFs for strength filter (NSE ETFs on yfinance)
sector_etf_map = {
    "DEFENCE":      "^NSEI",       # No dedicated ETF — use NIFTY
    "RAILWAYS":     "^NSEI",
    "PSU BANK":     "PSUBNKBEES.NS",
    "BANK":         "BANKBEES.NS",
    "IT":           "ITBEES.NS",
    "RENEW":        "^NSEI",
    "EV":           "^NSEI",
    "AUTO":         "^NSEI",
    "PHARMA":       "PHARMABEES.NS",
    "METAL":        "^NSEI",
    "FINTECH":      "^NSEI",
    "CONSUMP":      "^NSEI",
    "CAP MKT":      "^NSEI",
    "CHEM":         "^NSEI",
    "REALTY":       "^NSEI",
    "TELECOM":      "^NSEI",
    "ELECTRONICS":  "^NSEI",
    "INFRA":        "^NSEI",
    "OTHER":        "^NSEI",
}

# =========================================
# MAIN — pure scanner
# =========================================

def run_agent():

    print(f"\n{'='*50}")
    print(f"🇮🇳 India Scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
    print(f"{'='*50}")

    if not market_is_bullish():
        msg = (
            f"📉 NIFTY below EMA50 — market weak\n"
            f"Staying in cash. No trades today.\n"
            f"Time: {datetime.now().strftime('%d %b %Y %H:%M')} IST"
        )
        print(msg)
        send_telegram(msg)
        return

    # Download NIFTY benchmark
    nifty_df = yf.download("^NSEI", period="1y", interval="1d", progress=False)
    if nifty_df is None or nifty_df.empty:
        print("⚠️ Failed to download NIFTY data")
        return
    nifty_df = nifty_df.dropna()
    nifty_df.columns = nifty_df.columns.get_level_values(0)

    picks = []
    checked = 0

    for symbol in stocks:
        time.sleep(0.8)
        checked += 1

        # Sector strength check
        theme = sector_map.get(symbol, "OTHER")
        etf   = sector_etf_map.get(theme, "^NSEI")
        if etf != "^NSEI":
            if not sector_is_strong(etf):
                print(f"  {symbol}: sector weak — skip")
                continue

        result = check_stock(symbol, nifty_df)
        if result:
            print(f"  ✅ {result['Symbol']} [{result['Theme']}] — score {result['Score']}/28")
            picks.append(result)

    print(f"\n📊 Scanned {checked} stocks. {len(picks)} qualify.")

    if not picks:
        send_telegram(
            f"🔍 India Scan — {datetime.now().strftime('%d %b %Y %H:%M')} IST\n"
            f"NIFTY is bullish but no stocks meet all criteria.\n"
            f"Nothing to act on — wait for next scan."
        )
        return

    # Sort by score then reward
    picks = sorted(picks, key=lambda x: (x['Score'], x['Reward₹']), reverse=True)

    # Summary header
    send_telegram(
        f"📊 INDIA SCAN — {datetime.now().strftime('%d %b %Y %H:%M')} IST\n"
        f"✅ NIFTY bullish (above EMA50)\n"
        f"🎯 {len(picks)} stock(s) qualify — top {min(TOP_PICKS, len(picks))} below\n"
        f"Trade what suits you — alerts only."
    )

    # One message per top pick
    for pick in picks[:TOP_PICKS]:
        risk_reward = round(pick['Reward₹'] / pick['Risk₹'], 1) if pick['Risk₹'] > 0 else 0
        msg = (
            f"{'='*30}\n"
            f"🚀 {pick['Symbol']}  [{pick['Theme']}]\n"
            f"Score  : {pick['Score']}/28\n"
            f"Entry  : ₹{pick['Entry']}\n"
            f"Stop   : ₹{pick['Stop']}\n"
            f"Target : ₹{pick['Target']}\n"
            f"Qty    : {pick['Size']} shares\n"
            f"Risk   : ₹{int(pick['Risk₹']):,}\n"
            f"Reward : ₹{int(pick['Reward₹']):,}\n"
            f"RR     : 1:{risk_reward}\n"
            f"{'='*30}"
        )
        send_telegram(msg)
        time.sleep(0.5)

# =========================================
# RUN
# =========================================

if __name__ == "__main__":
    print("🚀 India Swing Trading Agent Started")
    run_agent()
    print("✅ Done")
