import yfinance as yf
import pandas as pd
import ta
import time
import os
import requests
import json
import csv
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

# =========================================
# SETTINGS
# =========================================

ACCOUNT_SIZE     = 500000        # ₹5 lakh — change to your actual capital
RISK_PER_TRADE   = 0.01          # 1% risk per trade
RR_RATIO         = 3.0           # 3:1 reward-to-risk
MAX_ATR_STOP     = 2.0           # Max 2x ATR stop loss
MAX_STOP_PCT     = 0.10          # Max 10% stop from entry
MAX_POSITION     = 2000          # Max shares per trade
SCORE_THRESHOLD  = 20            # Min score to qualify (raised — more indicators now)
TOP_PICKS        = 6
MAX_PER_SECTOR   = 2             # Max picks per theme

# Feature flags
NEWS_SENTIMENT   = True
CHECK_15MIN      = True
PORTFOLIO_FILE   = "positions.csv"
TRADE_LOG_FILE   = "trade_log.csv"

# Portfolio risk limits
MAX_PORTFOLIO_HEAT = 0.60        # max 60% of capital deployed
MAX_SECTOR_HEAT    = 0.20        # max 20% in any one theme

# India VIX thresholds (different scale from US VIX)
VIX_EXTREME      = 25.0          # skip scan
VIX_ELEVATED     = 20.0          # half position size

IST = timezone(timedelta(hours=5, minutes=30))

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID          = os.environ.get("CHAT_ID", "")

CACHE_FILE       = "/tmp/india_agent_cache.json"

# =========================================
# CACHE (avoid re-fetching fundamentals intraday)
# =========================================

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
# TELEGRAM
# =========================================

def send_telegram(msg):
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        if not resp.ok:
            print(f"⚠️ Telegram error: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")

# =========================================
# INDIA VIX
# =========================================

def get_india_vix():
    try:
        df = yf.download("^INDIAVIX", period="5d", interval="1d", progress=False)
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        vix = float(df['Close'].iloc[-1])
        print(f"India VIX: {vix:.1f}")
        return vix
    except Exception:
        return 15.0   # fail-open with neutral value

# =========================================
# MARKET BREADTH (Nifty 500 sample)
# =========================================

def get_market_breadth():
    nifty500_sample = [
        # Large caps
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS",
        "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","BAJFINANCE.NS",
        "LT.NS","HCLTECH.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS",
        "TITAN.NS","SUNPHARMA.NS","ULTRACEMCO.NS","WIPRO.NS","NTPC.NS",
        # Mid caps
        "PERSISTENT.NS","COFORGE.NS","POLYCAB.NS","DIXON.NS","TRENT.NS",
        "ZOMATO.NS","DMART.NS","ADANIGREEN.NS","HAVELLS.NS","MUTHOOTFIN.NS",
        "CDSL.NS","BSE.NS","ANGELONE.NS","TATAMOTORS.NS","M&M.NS",
        "DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","PIDILITIND.NS","DEEPAKNTR.NS",
        "IRFC.NS","RVNL.NS","HAL.NS","BEL.NS","TATAPOWER.NS",
        "SUZLON.NS","WAAREEENER.NS","CHOLAFIN.NS","IDFCFIRSTB.NS","MANKIND.NS"
    ]
    above_50 = 0
    total    = 0
    for sym in nifty500_sample:
        try:
            df = yf.download(sym, period="6mo", interval="1d", progress=False)
            df = df.dropna()
            df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 50:
                continue
            df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
            if float(df['Close'].iloc[-1]) > float(df['EMA50'].iloc[-1]):
                above_50 += 1
            total += 1
            time.sleep(0.3)
        except Exception:
            continue
    if total == 0:
        return 50
    pct = round((above_50 / total) * 100, 1)
    print(f"Market Breadth: {above_50}/{total} above EMA50 = {pct}%")
    return pct

# =========================================
# SECTOR ROTATION (NSE sectoral ETFs)
# =========================================

# Sectoral ETFs available on NSE via yfinance
SECTOR_ETFS_INDIA = {
    "BANKBEES.NS":   "Banking",
    "PSUBNKBEES.NS": "PSU Banks",
    "ITBEES.NS":     "IT",
    "PHARMABEES.NS": "Pharma",
    "MOM100.NS":     "Momentum/Auto",
    "JUNIORBEES.NS": "Midcap",
    "NIFTYBEES.NS":  "Nifty 50",
    "CPSE.NS":       "PSU/Defence",
}

# Map each theme to its nearest sector ETF for hot-sector bonus
THEME_TO_ETF = {
    "BANK":       "BANKBEES.NS",
    "PSU BANK":   "PSUBNKBEES.NS",
    "FINTECH":    "BANKBEES.NS",
    "INSURANCE":  "BANKBEES.NS",
    "CAP MKT":    "BANKBEES.NS",
    "IT":         "ITBEES.NS",
    "PHARMA":     "PHARMABEES.NS",
    "AUTO":       "MOM100.NS",
    "EV":         "MOM100.NS",
    "DEFENCE":    "CPSE.NS",
    "RAILWAYS":   "CPSE.NS",
    "CONSUMP":    "JUNIORBEES.NS",
    "FMCG":       "NIFTYBEES.NS",
    "REALTY":     "JUNIORBEES.NS",
    "CEMENT":     "JUNIORBEES.NS",
    "METAL":      "JUNIORBEES.NS",
    "ELECTRONICS":"JUNIORBEES.NS",
    "ELECTRICAL": "JUNIORBEES.NS",
    "CHEM":       "JUNIORBEES.NS",
    "AGROCHEM":   "JUNIORBEES.NS",
    "RENEW":      "NIFTYBEES.NS",
    "INFRA":      "NIFTYBEES.NS",
    "LOGISTICS":  "JUNIORBEES.NS",
    "TELECOM":    "NIFTYBEES.NS",
}

def get_sector_rotation():
    hot_sectors = set()
    sector_perf = {}

    for etf, name in SECTOR_ETFS_INDIA.items():
        try:
            df = yf.download(etf, period="3mo", interval="1d", progress=False)
            df = df.dropna()
            df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 21:
                continue
            ret_1w = float(df['Close'].pct_change(5).iloc[-1])
            ret_1m = float(df['Close'].pct_change(21).iloc[-1])
            avg_r  = float(df['Volume'].iloc[-5:].mean())
            avg_o  = float(df['Volume'].iloc[-21:-5].mean())
            vol_tr = avg_r / avg_o if avg_o > 0 else 1
            sector_perf[etf] = {
                "name": name, "ret_1w": ret_1w,
                "ret_1m": ret_1m, "vol_trend": vol_tr
            }
            time.sleep(0.3)
        except Exception:
            continue

    if not sector_perf:
        return hot_sectors

    all_1w = [v['ret_1w'] for v in sector_perf.values()]
    all_1m = [v['ret_1m'] for v in sector_perf.values()]
    med_1w = sorted(all_1w)[len(all_1w) // 2]
    med_1m = sorted(all_1m)[len(all_1m) // 2]

    print("\nSector Rotation (India):")
    for etf, d in sorted(sector_perf.items(), key=lambda x: x[1]['ret_1w'], reverse=True):
        is_hot = d['ret_1w'] > med_1w and d['ret_1m'] > med_1m and d['vol_trend'] > 0.9
        if is_hot:
            hot_sectors.add(etf)
        flag = "🔥" if is_hot else "  "
        print(f"  {flag} {etf:18} {d['name']:20} 1W:{d['ret_1w']:+.1%} 1M:{d['ret_1m']:+.1%}")

    return hot_sectors

# =========================================
# CANDLE QUALITY
# =========================================

def candle_quality_score(df):
    score = 0
    try:
        c1 = df.iloc[-1]
        c2 = df.iloc[-2]
        c3 = df.iloc[-3]
        atr         = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else float(c1['High'] - c1['Low'])
        today_range = float(c1['High'] - c1['Low'])
        today_body  = abs(float(c1['Close'] - c1['Open']))
        today_upper = float(c1['High']) - max(float(c1['Close']), float(c1['Open']))
        close_pos   = ((float(c1['Close']) - float(c1['Low'])) / today_range) if today_range > 0 else 0.5

        if close_pos > 0.70:                                        score += 2
        elif close_pos > 0.50:                                      score += 1
        elif close_pos < 0.30:                                      score -= 2

        if today_range > 0 and today_upper / today_range > 0.40:    score -= 1
        if today_body > atr * 0.5:                                  score += 1
        if float(c1['Open']) > float(c2['Close']) * 1.005:          score += 1
        if float(c1['Close']) > float(c2['High']):                  score += 1

        bull = sum([
            float(c1['Close']) > float(c1['Open']),
            float(c2['Close']) > float(c2['Open']),
            float(c3['Close']) > float(c3['Open']),
        ])
        if bull == 3:    score += 1
        elif bull <= 1:  score -= 1

        if today_range > 0 and today_body / today_range < 0.10:     score -= 1  # doji
    except Exception:
        pass
    return score

# =========================================
# FUNDAMENTAL FILTER
# (Stricter: EPS > -20, D/E < 200, ROE > 8%)
# =========================================

SKIP_FUNDAMENTAL = {
    "PSU BANK", "RAILWAYS", "DEFENCE", "RENEW", "METAL",
    "LOGISTICS", "CEMENT",
}

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
            if eps    is not None and eps    < -20:             result = False
            if de     is not None and de     > 200:             result = False
            if mktcap is not None and mktcap < 5_000_000_000:  result = False
            if rev    is not None and rev    <= 0:              result = False
            if roe    is not None and roe    < 0.08:
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
# MARKET TREND — NIFTY 50
# =========================================

def market_is_bullish():
    df = yf.download("^NSEI", period="1y", interval="1d", progress=False)
    df = df.dropna()
    df.columns = df.columns.get_level_values(0)
    if df.empty:
        return False
    df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    latest = df.iloc[-1]
    close  = float(latest['Close'])
    ema50  = float(latest['EMA50'])
    ema200 = float(latest['EMA200'])
    print(f"NIFTY: {close:.0f} | EMA50: {ema50:.0f} | EMA200: {ema200:.0f}")
    return close > ema50 and close > ema200

# =========================================
# SECTOR STRENGTH
# =========================================

# Expanded ETF map — more themes covered
SECTOR_ETF_STRENGTH_MAP = {
    "BANK":       "BANKBEES.NS",
    "PSU BANK":   "PSUBNKBEES.NS",
    "FINTECH":    "BANKBEES.NS",
    "INSURANCE":  "BANKBEES.NS",
    "CAP MKT":    "BANKBEES.NS",
    "IT":         "ITBEES.NS",
    "PHARMA":     "PHARMABEES.NS",
    "AUTO":       "MOM100.NS",
    "EV":         "MOM100.NS",
    "DEFENCE":    "CPSE.NS",
    "RAILWAYS":   "CPSE.NS",
    "CONSUMP":    "JUNIORBEES.NS",
    "REALTY":     "JUNIORBEES.NS",
    "CEMENT":     "JUNIORBEES.NS",
    "METAL":      "JUNIORBEES.NS",
    "ELECTRONICS":"JUNIORBEES.NS",
    "ELECTRICAL": "JUNIORBEES.NS",
    "CHEM":       "JUNIORBEES.NS",
    "AGROCHEM":   "JUNIORBEES.NS",
    "LOGISTICS":  "JUNIORBEES.NS",
}

def sector_is_strong(theme):
    etf = SECTOR_ETF_STRENGTH_MAP.get(theme)
    if not etf:
        return True   # no ETF proxy — pass through
    try:
        df = yf.download(etf, period="1y", interval="1d", progress=False)
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 50:
            return True
        df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
        df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
        close = float(df['Close'].iloc[-1])
        ema50 = float(df['EMA50'].iloc[-1])
        if len(df) >= 200:
            ema200 = float(df['EMA200'].iloc[-1])
            return close > ema50 and close > ema200
        return close > ema50
    except Exception:
        return True

# =========================================
# EARNINGS FILTER
# =========================================

def is_near_earnings(symbol, theme, days=10):
    if theme in SKIP_FUNDAMENTAL:
        return False
    try:
        cal = yf.Ticker(symbol).calendar
        if not cal:
            return False
        if isinstance(cal, dict):
            earn_dates = cal.get('Earnings Date', [])
            if not earn_dates:
                return False
            earn_date = pd.Timestamp(earn_dates[0]).date()
        else:
            if 'Earnings Date' not in cal.index:
                return False
            earn_date = pd.Timestamp(cal.loc['Earnings Date'].iloc[0]).date()
        diff = abs((earn_date - datetime.now().date()).days)
        if diff <= days:
            print(f"⚠️ {symbol} earnings in {diff} days — skip")
            return True
    except Exception:
        pass
    return False

# =========================================
# NEWS SENTIMENT
# =========================================

BULLISH_WORDS = [
    "upgrade", "beat", "beats", "record", "breakout", "surge", "surges",
    "growth", "strong", "buy", "bullish", "outperform", "raises", "raised",
    "contract", "wins", "order", "profit", "revenue beat", "guidance raised",
    "buyback", "dividend", "partnership", "expansion", "acquisition"
]
BEARISH_WORDS = [
    "downgrade", "miss", "misses", "cut", "cuts", "warning", "weak",
    "loss", "losses", "sell", "probe", "fine", "recall", "investigation",
    "layoff", "guidance cut", "revenue miss", "bankruptcy", "lawsuit",
    "fraud", "halt", "suspended", "sebi", "default", "npa"
]

def get_news_sentiment(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news   = ticker.news
        if not news:
            return 0, "Neutral", []
        score     = 0
        headlines = []
        for article in news[:6]:
            title = article.get("title", "")
            if not title:
                continue
            low = title.lower()
            for w in BULLISH_WORDS:
                if w in low: score += 1
            for w in BEARISH_WORDS:
                if w in low: score -= 1
            headlines.append(title)
        if score >= 2:       label = "Positive"
        elif score <= -2:    label = "Negative"
        elif score == 1:     label = "Slightly Positive"
        elif score == -1:    label = "Slightly Negative"
        else:                label = "Neutral"
        return score, label, headlines[:3]
    except Exception:
        return 0, "Neutral", []

# =========================================
# 15-MIN CONFIRMATION (NSE intraday)
# =========================================

def passes_15min_check(symbol, daily_entry):
    try:
        df = yf.download(symbol, period="5d", interval="15m", progress=False)
        if df is None or df.empty or len(df) < 30:
            return True, "Data unavailable"
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if len(df) < 30:
            return True, "Insufficient bars"

        current_price = float(df['Close'].iloc[-1])
        drift = (current_price - daily_entry) / daily_entry

        if drift > 0.03:
            return False, f"Price ran +{drift:.1%} above entry — chasing risk"
        if drift < -0.04:
            return False, f"Price dropped {drift:.1%} — setup breaking"

        macd_line   = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
        macd_signal = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        macd_ok     = float(macd_line.iloc[-1]) > float(macd_signal.iloc[-1])

        rsi   = ta.momentum.rsi(df['Close'], window=14)
        rsi_v = float(rsi.iloc[-1])
        rsi_ok = 45 < rsi_v < 78

        avg_vol = float(df['Volume'].iloc[-20:].mean())
        cur_vol = float(df['Volume'].iloc[-3:].mean())
        vol_ok  = cur_vol >= avg_vol * 0.8

        fails = []
        if not macd_ok: fails.append("15m MACD bearish")
        if not rsi_ok:  fails.append(f"15m RSI={rsi_v:.0f}")
        if not vol_ok:  fails.append(f"15m vol thin ({cur_vol/avg_vol:.0%} avg)")

        if len(fails) >= 2:
            return False, " | ".join(fails)
        elif fails:
            return True, f"⚠️ Minor: {fails[0]}"
        else:
            return True, f"✅ RSI={rsi_v:.0f}, MACD bullish, vol OK"

    except Exception as e:
        return True, f"Check skipped ({e})"

# =========================================
# PORTFOLIO RISK
# =========================================

def get_portfolio_positions():
    pos_file = Path(PORTFOLIO_FILE)
    if not pos_file.exists():
        return {}
    positions = {}
    try:
        with open(pos_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get('symbol', '').strip().upper()
                if not sym:
                    continue
                try:
                    positions[sym] = {
                        'shares': int(float(row.get('shares', 0))),
                        'entry':  float(row.get('entry_price', 0)),
                        'theme':  row.get('sector', 'OTHER').strip(),
                    }
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"⚠️ Could not read {PORTFOLIO_FILE}: {e}")
    return positions

def get_portfolio_heat(positions):
    if not positions:
        return 0.0, {}, "No open positions"
    total_invested = 0.0
    theme_invested = {}
    for sym, pos in positions.items():
        value = pos['shares'] * pos['entry']
        total_invested += value
        th = pos['theme']
        theme_invested[th] = theme_invested.get(th, 0.0) + value
    total_pct  = total_invested / ACCOUNT_SIZE
    theme_pct  = {th: v / ACCOUNT_SIZE for th, v in theme_invested.items()}
    lines = [f"Portfolio heat: {total_pct:.0%} deployed (₹{total_invested:,.0f})"]
    for th, pct in sorted(theme_pct.items(), key=lambda x: -x[1]):
        bar = "🔴" if pct > MAX_SECTOR_HEAT else "🟡" if pct > MAX_SECTOR_HEAT * 0.7 else "🟢"
        lines.append(f"  {bar} {th}: {pct:.0%}")
    return total_pct, theme_pct, "\n".join(lines)

def pick_blocked_by_portfolio(pick, positions, theme_pct):
    sym    = pick['Symbol']
    theme  = pick['Theme']
    invest = pick['Invested']

    # Check using .NS suffix too
    if sym in positions or f"{sym}.NS" in positions:
        return True, f"{sym} already in portfolio"

    new_theme_pct = theme_pct.get(theme, 0.0) + (invest / ACCOUNT_SIZE)
    if new_theme_pct > MAX_SECTOR_HEAT:
        return True, f"{theme} theme would be {new_theme_pct:.0%} > {MAX_SECTOR_HEAT:.0%} limit"

    return False, ""

# =========================================
# TRADE LOGGING
# =========================================

TRADE_LOG_FIELDS = [
    'date', 'symbol', 'theme', 'setup', 'score',
    'entry', 'stop', 'target', 'qty', 'invested',
    'risk_inr', 'reward_inr', 'rr',
    'news_sentiment', 'confirmed_15m',
    'outcome', 'outcome_date', 'exit_price', 'pnl_inr', 'pnl_pct'
]

def log_picks(picks, confirmed_map, sentiment_map):
    log_file    = Path(TRADE_LOG_FILE)
    today_str   = datetime.now(IST).strftime('%Y-%m-%d')
    file_exists = log_file.exists()
    existing    = set()
    if file_exists:
        try:
            with open(log_file, newline='') as f:
                for row in csv.DictReader(f):
                    existing.add((row.get('date',''), row.get('symbol','')))
        except Exception:
            pass
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        for pick in picks:
            sym = pick['Symbol']
            if (today_str, sym) in existing:
                continue
            rr = round(pick['Reward₹'] / pick['Risk₹'], 2) if pick['Risk₹'] > 0 else 0
            writer.writerow({
                'date':           today_str,
                'symbol':         sym,
                'theme':          pick['Theme'],
                'setup':          pick.get('Setup', ''),
                'score':          pick['Score'],
                'entry':          pick['Entry'],
                'stop':           pick['Stop'],
                'target':         pick['Target'],
                'qty':            pick['Qty'],
                'invested':       int(pick['Invested']),
                'risk_inr':       int(pick['Risk₹']),
                'reward_inr':     int(pick['Reward₹']),
                'rr':             rr,
                'news_sentiment': sentiment_map.get(sym, ('', 'N/A', []))[1],
                'confirmed_15m':  'Yes' if confirmed_map.get(sym, (True,''))[0] else 'No',
                'outcome':        '',
                'outcome_date':   '',
                'exit_price':     '',
                'pnl_inr':        '',
                'pnl_pct':        '',
            })
    print(f"📋 Logged {len(picks)} picks to {TRADE_LOG_FILE}")

def update_trade_outcomes():
    log_file = Path(TRADE_LOG_FILE)
    if not log_file.exists():
        return
    rows    = []
    updated = 0
    today_str = datetime.now(IST).strftime('%Y-%m-%d')
    try:
        with open(log_file, newline='') as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"⚠️ Could not read trade log: {e}")
        return
    for row in rows:
        if row.get('outcome', '').strip():
            continue
        sym    = row.get('symbol', '')
        entry  = float(row.get('entry', 0) or 0)
        stop   = float(row.get('stop', 0) or 0)
        target = float(row.get('target', 0) or 0)
        qty    = int(float(row.get('qty', 0) or 0))
        if not sym or entry <= 0:
            continue
        # Add .NS suffix if not present
        fetch_sym = sym if sym.endswith('.NS') else f"{sym}.NS"
        try:
            df = yf.download(fetch_sym, period="5d", interval="1d", progress=False)
            if df is None or df.empty:
                continue
            df = df.dropna()
            df.columns = df.columns.get_level_values(0)
            last  = df.iloc[-1]
            hi    = float(last['High'])
            lo    = float(last['Low'])
            close = float(last['Close'])
            outcome    = ''
            exit_price = close
            if lo <= stop:
                outcome    = 'STOPPED'
                exit_price = stop
            elif hi >= target:
                outcome    = 'TARGET HIT'
                exit_price = target
            if outcome:
                pnl_inr = round((exit_price - entry) * qty, 2)
                pnl_pct = round((exit_price - entry) / entry * 100, 2)
                row['outcome']      = outcome
                row['outcome_date'] = today_str
                row['exit_price']   = exit_price
                row['pnl_inr']      = pnl_inr
                row['pnl_pct']      = pnl_pct
                updated += 1
                emoji = "✅" if outcome == 'TARGET HIT' else "❌"
                print(f"  {emoji} {sym}: {outcome} | P&L ₹{pnl_inr:+.0f} ({pnl_pct:+.1f}%)")
            else:
                unreal = round((close - entry) * qty, 2)
                unreal_pct = round((close - entry) / entry * 100, 2)
                print(f"  🔄 {sym}: open | ₹{close:.2f} | unrealized ₹{unreal:+.0f} ({unreal_pct:+.1f}%)")
        except Exception as e:
            print(f"  ⚠️ {sym} outcome check failed: {e}")
    if updated > 0:
        try:
            with open(log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_LOG_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"📋 Updated {updated} trade outcome(s) in {TRADE_LOG_FILE}")
        except Exception as e:
            print(f"⚠️ Could not write trade log: {e}")

def print_trade_stats():
    log_file = Path(TRADE_LOG_FILE)
    if not log_file.exists():
        return
    try:
        with open(log_file, newline='') as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return
    closed = [r for r in rows if r.get('outcome','').strip() in ('TARGET HIT','STOPPED')]
    if not closed:
        return
    wins  = [r for r in closed if r['outcome'] == 'TARGET HIT']
    total = len(closed)
    win_r = len(wins) / total * 100
    try:
        pnls = [float(r['pnl_inr']) for r in closed if r.get('pnl_inr')]
        net  = sum(pnls)
        avg  = net / len(pnls) if pnls else 0
        print(f"\n📈 Trade History: {total} closed | Win rate: {win_r:.0f}% | Net P&L: ₹{net:+,.0f} | Avg: ₹{avg:+.0f}")
    except Exception:
        print(f"\n📈 Trade History: {total} closed | Win rate: {win_r:.0f}%")

# =========================================
# MAIN TECHNICAL SCANNER
# =========================================

def check_stock(symbol, nifty_df, hot_sectors):
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df is None or df.empty or len(df) < 200:
            return None
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 200:
            return None

        price = float(df['Close'].iloc[-1])
        if price < 20.0:
            return None

        avg_turnover = float(df['Close'].iloc[-20:].mean() * df['Volume'].iloc[-20:].mean())
        if avg_turnover < 20_000_000:   # ₹2 crore daily
            return None

        # ---- INDICATORS ----
        df['EMA10']  = ta.trend.ema_indicator(df['Close'], window=10)
        df['EMA20']  = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
        df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
        df['RSI']    = ta.momentum.rsi(df['Close'], window=14)
        df['AvgVol'] = df['Volume'].rolling(20).mean()
        df['HH20']   = df['High'].rolling(20).max()
        df['High52'] = df['High'].rolling(252).max()
        df['Low52']  = df['Low'].rolling(252).min()

        df['TR'] = (
            df['High'] - df['Low']
        ).combine(abs(df['High'] - df['Close'].shift(1)), max
        ).combine(abs(df['Low']  - df['Close'].shift(1)), max)
        df['ATR'] = df['TR'].rolling(14).mean()

        # ADX — trend strength
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        adx_val   = float(df['ADX'].iloc[-1]) if not pd.isna(df['ADX'].iloc[-1]) else 20.0

        # Bollinger Band Squeeze
        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
        bb_squeeze = False
        if len(df) >= 60:
            pct20      = float(df['BB_width'].iloc[-60:].quantile(0.20))
            bb_squeeze = float(df['BB_width'].iloc[-1]) < pct20

        # OBV
        df['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['OBV_EMA20'] = ta.trend.ema_indicator(df['OBV'], window=20)
        obv_rising      = float(df['OBV'].iloc[-1]) > float(df['OBV_EMA20'].iloc[-1])
        obv_slope       = float(df['OBV'].iloc[-1]) - float(df['OBV'].iloc[-10])
        obv_trend_pos   = obv_slope > 0

        # MACD
        df['MACD']   = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
        df['MACD_S'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD_H'] = ta.trend.macd_diff(df['Close'], window_slow=26, window_fast=12, window_sign=9)

        macd_now     = float(df['MACD'].iloc[-1])
        macd_sig_now = float(df['MACD_S'].iloc[-1])
        macd_prev    = float(df['MACD'].iloc[-2])
        macd_sig_prv = float(df['MACD_S'].iloc[-2])
        macd_hist_n  = float(df['MACD_H'].iloc[-1])
        macd_hist_p  = float(df['MACD_H'].iloc[-2])

        macd_crossed_up  = macd_prev < macd_sig_prv and macd_now > macd_sig_now
        macd_above_sig   = macd_now > macd_sig_now
        macd_hist_rising = macd_hist_n > macd_hist_p and macd_hist_n > 0
        macd_above_zero  = macd_now > 0

        # ROC-21
        df['ROC21'] = df['Close'].pct_change(21)
        roc21 = float(df['ROC21'].iloc[-1])

        # Relative strength vs Nifty
        s3  = df['Close'].pct_change(63).iloc[-1]
        s6  = df['Close'].pct_change(126).iloc[-1]
        s12 = df['Close'].pct_change(252).iloc[-1] if len(df) >= 252 else s6
        n3  = nifty_df['Close'].pct_change(63).iloc[-1]
        n6  = nifty_df['Close'].pct_change(126).iloc[-1]
        n12 = nifty_df['Close'].pct_change(252).iloc[-1]
        rs  = sum([s3 > n3, s6 > n6, s12 > n12])

        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        score  = 0

        # ---- RELATIVE STRENGTH ----
        if rs >= 2: score += 2
        if rs == 3: score += 1

        # ---- EMA TREND STACK ----
        if latest['EMA10']  > latest['EMA20']:  score += 2
        if latest['EMA20']  > latest['EMA50']:  score += 2
        if latest['EMA50']  > latest['EMA200']: score += 2
        if latest['Close']  > latest['EMA50']:  score += 1
        if latest['Close']  > latest['EMA200']: score += 2

        # ---- RSI ----
        rsi      = float(latest['RSI'])
        rsi_prev = float(prev['RSI'])
        if rsi < 40:                                    score -= 2
        if rsi_prev < 50 and rsi > 50:                  score += 2   # fresh cross above midline
        elif 55 < rsi < 75:                             score += 2   # healthy momentum zone
        elif rsi > 50 and rsi > rsi_prev:               score += 1   # above 50 and rising
        if rsi > 80:                                    score -= 2
        elif rsi > 75:                                  score -= 1

        # ---- BREAKOUT ----
        broke_20d = latest['Close'] > prev['HH20']
        if broke_20d:                                   score += 2
        ath_dist = latest['Close'] / latest['High52']
        if ath_dist > 0.90:                             score += 2
        elif ath_dist > 0.80:                           score += 1

        # ---- VOLUME ----
        rvol = 0.0
        if latest['AvgVol'] > 0:
            rvol = latest['Volume'] / latest['AvgVol']
            if rvol > 2.0:   score += 2
            elif rvol > 1.5: score += 1

        # ---- BREAKOUT + VOLUME CONFIRMATION ----
        if broke_20d and rvol > 1.5: score += 1

        # ---- ATR EXPANDING ----
        if latest['ATR'] > df['ATR'].iloc[-5]: score += 1

        # ---- ENTRY QUALITY ----
        dist_ema20 = (latest['Close'] - latest['EMA20']) / latest['EMA20']
        if dist_ema20 < 0.05:   score += 2
        elif dist_ema20 < 0.08: score += 1
        elif dist_ema20 > 0.15: score -= 1

        # ---- 60-DAY OUTPERFORMANCE ----
        if len(df) >= 60 and len(nifty_df) >= 60:
            sr = float(df['Close'].squeeze().pct_change(60).iloc[-1])
            nr = float(nifty_df['Close'].squeeze().pct_change(60).iloc[-1])
            if sr > nr * 1.5:  score += 2
            elif sr > nr:      score += 1

        # ---- DISTRIBUTION PENALTY ----
        recent_high = df['High'].iloc[-10:].max()
        if latest['Close'] < recent_high * 0.85: score -= 2

        # ---- OBV ----
        if obv_rising and obv_trend_pos:               score += 2
        elif obv_rising:                                score += 1
        elif not obv_trend_pos and not obv_rising:      score -= 2

        # ---- MACD ----
        if macd_crossed_up:                             score += 2
        elif macd_above_sig and macd_hist_rising:       score += 2
        elif macd_above_sig:                            score += 1
        elif not macd_above_sig and macd_hist_n < 0:   score -= 1
        if macd_above_zero:                             score += 1   # India-specific: above zero line bonus

        # ---- ADX ----
        if adx_val > 30:   score += 2
        elif adx_val > 20: score += 1
        elif adx_val < 15: score -= 2

        # ---- ROC-21 (India-specific momentum) ----
        if roc21 > 0.15:   score += 1
        if roc21 < 0:      score -= 1

        # ---- BB SQUEEZE ----
        if bb_squeeze: score += 3

        # ---- CANDLE QUALITY ----
        candle_score = candle_quality_score(df)
        score += candle_score

        # ---- HOT SECTOR BONUS ----
        theme     = sector_map.get(symbol, "OTHER")
        theme_etf = THEME_TO_ETF.get(theme)
        if theme_etf and theme_etf in hot_sectors:
            score += 2

        if score < SCORE_THRESHOLD:
            return None

        # ---- POSITION SIZING ----
        entry        = float(latest['Close'])
        atr          = float(latest['ATR'])
        five_bar_low = float(df['Low'].iloc[-5:].min())
        atr_stop     = entry - (MAX_ATR_STOP * atr)
        stop         = max(five_bar_low, atr_stop)
        risk         = entry - stop

        if risk <= 0 or risk > entry * MAX_STOP_PCT:
            return None

        risk_amt = ACCOUNT_SIZE * RISK_PER_TRADE
        qty      = min(int(risk_amt / risk), MAX_POSITION)
        if qty <= 0:
            return None

        target   = entry + (risk * RR_RATIO)
        invested = round(entry * qty, 0)
        acct_pct = round((invested / ACCOUNT_SIZE) * 100, 1)

        # ---- SETUP TYPE ----
        if bb_squeeze and broke_20d:
            setup = "Squeeze Breakout"
        elif broke_20d and rvol > 2.0:
            setup = "Volume Breakout"
        elif dist_ema20 < 0.05 and rsi > 50:
            setup = "EMA20 Pullback"
        elif ath_dist > 0.95:
            setup = "ATH Breakout"
        elif bb_squeeze:
            setup = "Squeeze Setup"
        else:
            setup = "Trend Continuation"

        # ---- LABELS ----
        if candle_score >= 3:    candle_label = "Strong"
        elif candle_score >= 1:  candle_label = "Good"
        elif candle_score == 0:  candle_label = "Neutral"
        else:                    candle_label = "Weak"

        if macd_crossed_up:   macd_label = "Fresh cross"
        elif macd_above_sig:  macd_label = "Bullish"
        else:                 macd_label = "Bearish"

        obv_label = "Confirming" if obv_rising and obv_trend_pos else \
                    "Rising"     if obv_rising else "Diverging"

        adx_label = f"{adx_val:.0f} ({'Strong' if adx_val > 30 else 'Moderate' if adx_val > 20 else 'Weak'})"

        return {
            "Symbol":    symbol.replace(".NS", ""),
            "Theme":     theme,
            "Score":     score,
            "Setup":     setup,
            "Candle":    candle_label,
            "MACD":      macd_label,
            "OBV":       obv_label,
            "ADX":       adx_label,
            "Squeeze":   "Yes" if bb_squeeze else "No",
            "Entry":     round(entry, 2),
            "Stop":      round(float(stop), 2),
            "Target":    round(float(target), 2),
            "StopPct":   round((entry - float(stop)) / entry * 100, 1),
            "TargetPct": round((float(target) - entry) / entry * 100, 1),
            "Qty":       qty,
            "Invested":  invested,
            "AcctPct":   acct_pct,
            "Risk₹":     round(risk * qty, 0),
            "Reward₹":   round((float(target) - entry) * qty, 0),
        }

    except Exception as e:
        print(f"  ⚠️ {symbol}: {e}")
        return None

# =========================================
# STOCK UNIVERSE — NSE stocks
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

    # ---- FMCG ----
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

    # ---- AGROCHEM ----
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

    # ---- ELECTRONICS MFG (PLI) ----
    "DIXON.NS":       "ELECTRONICS",
    "AMBER.NS":       "ELECTRONICS",
    "KAYNES.NS":      "ELECTRONICS",
    "SYRMA.NS":       "ELECTRONICS",
    "PGEL.NS":        "ELECTRONICS",
    "AVALON.NS":      "ELECTRONICS",

    # ---- ELECTRICAL & CONSUMER DURABLE ----
    "HAVELLS.NS":     "ELECTRICAL",
    "POLYCAB.NS":     "ELECTRICAL",
    "VOLTAS.NS":      "ELECTRICAL",
    "CROMPTON.NS":    "ELECTRICAL",
    "BLUESTARCO.NS":  "ELECTRICAL",

    # ---- CEMENT ----
    "ULTRACEMCO.NS":  "CEMENT",
    "SHREECEM.NS":    "CEMENT",
    "AMBUJACEM.NS":   "CEMENT",
    "JKCEMENT.NS":    "CEMENT",

    # ---- INSURANCE ----
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

    # ---- LOGISTICS ----
    "CONCOR.NS":      "LOGISTICS",
    "BLUEDART.NS":    "LOGISTICS",
    "MAHLOG.NS":      "LOGISTICS",

    # ---- TELECOM & DIGITAL ----
    "BHARTIARTL.NS":  "TELECOM",
    "HFCL.NS":        "TELECOM",
    "INDUSTOWER.NS":  "TELECOM",
}

stocks = list(sector_map.keys())

# =========================================
# MAIN
# =========================================

def run_agent():
    print(f"\n{'='*55}")
    now_ist = datetime.now(IST)
    print(f"🇮🇳 India Pro Scan — {now_ist.strftime('%d %b %Y %H:%M:%S')} IST")
    print(f"{'='*55}")

    # ---- Step 0: Check open trade outcomes ----
    print("\nChecking open trade outcomes...")
    update_trade_outcomes()
    print_trade_stats()

    # ---- Step 1: Market regime filters ----
    if not market_is_bullish():
        msg = (
            f"📉 NIFTY weak — staying in cash\n"
            f"Market below EMA50 or EMA200\n"
            f"{now_ist.strftime('%d %b %Y %H:%M')} IST"
        )
        print(msg)
        send_telegram(msg)
        return

    vix = get_india_vix()
    if vix > VIX_EXTREME:
        send_telegram(
            f"⚠️ India VIX={vix:.1f} — extreme fear\n"
            f"Swing setups fail at high rates when VIX >{VIX_EXTREME}.\n"
            f"Skipping scan — stay in cash.\n"
            f"{now_ist.strftime('%d %b %Y %H:%M')} IST"
        )
        return

    effective_risk = RISK_PER_TRADE * (0.5 if vix > VIX_ELEVATED else 1.0)
    vix_label      = f"Elevated ({vix:.1f}) — half size" if vix > VIX_ELEVATED else f"Normal ({vix:.1f})"

    print("\nChecking market breadth...")
    breadth = get_market_breadth()
    if breadth < 40:
        send_telegram(
            f"⚠️ Market breadth WEAK ({breadth}%)\n"
            f"Only {breadth}% of Nifty 500 stocks above EMA50.\n"
            f"Skipping — too risky.\n"
            f"{now_ist.strftime('%d %b %Y %H:%M')} IST"
        )
        return

    breadth_label = "Strong" if breadth >= 60 else "Mixed" if breadth >= 40 else "Weak"

    print("\nChecking sector rotation...")
    hot_sectors = get_sector_rotation()

    # ---- Step 2: Portfolio snapshot ----
    positions             = get_portfolio_positions()
    total_heat, theme_pct, heat_summary = get_portfolio_heat(positions)
    print(f"\n{heat_summary}")

    if total_heat >= MAX_PORTFOLIO_HEAT:
        send_telegram(
            f"🔴 Portfolio fully deployed ({total_heat:.0%}) — no new entries\n"
            f"{now_ist.strftime('%d %b %Y %H:%M')} IST"
        )
        return

    # ---- Step 3: Nifty reference ----
    nifty_df = yf.download("^NSEI", period="1y", interval="1d", progress=False)
    if nifty_df is None or nifty_df.empty:
        print("⚠️ Failed to download NIFTY data")
        return
    nifty_df = nifty_df.dropna()
    nifty_df.columns = nifty_df.columns.get_level_values(0)

    # ---- Step 4: Scan ----
    picks          = []
    sector_counts  = {}
    skipped_fund   = 0
    skipped_sec    = 0
    skipped_earn   = 0
    skipped_port   = 0
    skipped_corr   = 0

    print(f"\nScanning {len(stocks)} stocks...")

    for symbol in stocks:
        time.sleep(0.8)
        theme = sector_map.get(symbol, "OTHER")

        if not passes_fundamental_filter(symbol, theme):
            skipped_fund += 1
            continue

        if not sector_is_strong(theme):
            print(f"  {symbol}: sector {theme} weak — skip")
            skipped_sec += 1
            continue

        if is_near_earnings(symbol, theme):
            skipped_earn += 1
            continue

        result = check_stock(symbol, nifty_df, hot_sectors)

        if result:
            blocked, block_reason = pick_blocked_by_portfolio(result, positions, theme_pct)
            if blocked:
                print(f"  {symbol}: portfolio block — {block_reason}")
                skipped_port += 1
                continue

            th = result['Theme']
            if sector_counts.get(th, 0) >= MAX_PER_SECTOR:
                print(f"  {symbol}: theme {th} full ({MAX_PER_SECTOR}) — skip")
                skipped_corr += 1
                continue
            sector_counts[th] = sector_counts.get(th, 0) + 1

            print(
                f"  ✅ {result['Symbol']:15} [{result['Theme']:10}]"
                f" score:{result['Score']:3}"
                f" setup:{result['Setup']:20}"
                f" ADX:{result['ADX']}"
            )
            picks.append(result)

    print(f"\n{'='*55}")
    print(f"Scanned:{len(stocks)} Fund❌:{skipped_fund} Sector❌:{skipped_sec} "
          f"Earn❌:{skipped_earn} Port❌:{skipped_port} Corr❌:{skipped_corr} ✅:{len(picks)}")
    print(f"{'='*55}")

    if not picks:
        send_telegram(
            f"🔍 India Scan — {now_ist.strftime('%d %b %Y %H:%M')} IST\n"
            f"✅ NIFTY bullish | Breadth: {breadth_label} ({breadth}%)\n"
            f"No high-quality setups found — wait for next scan."
        )
        return

    picks = sorted(picks, key=lambda x: (x['Score'], x['Reward₹']), reverse=True)
    top_picks = picks[:TOP_PICKS]

    # ---- Step 5: News + 15-min checks ----
    sentiment_map = {}
    confirmed_map = {}

    print(f"\nRunning post-scan checks on top {len(top_picks)} picks...")
    for pick in top_picks:
        sym = pick['Symbol']
        sym_ns = f"{sym}.NS"

        if NEWS_SENTIMENT:
            ns, nl, nh = get_news_sentiment(sym_ns)
            sentiment_map[sym] = (ns, nl, nh)
            print(f"  📰 {sym} news: {nl} (score {ns:+d})")
            time.sleep(0.3)

        if CHECK_15MIN:
            ok, reason = passes_15min_check(sym_ns, pick['Entry'])
            confirmed_map[sym] = (ok, reason)
            status = "✅" if ok else "❌"
            print(f"  {status} {sym} 15m: {reason}")
            time.sleep(0.5)

    # ---- Step 6: Log picks ----
    log_picks(top_picks, confirmed_map, sentiment_map)

    # ---- Step 7: Telegram alerts ----
    hot_str = ", ".join(sorted(hot_sectors)) if hot_sectors else "None"
    pos_str = f"{len(positions)} open" if positions else "None"

    send_telegram(
        f"📊 INDIA PRO SCAN — {now_ist.strftime('%d %b %Y %H:%M')} IST\n"
        f"{'='*34}\n"
        f"NIFTY  : Bullish (EMA50 + EMA200)\n"
        f"VIX    : {vix_label}\n"
        f"Breadth: {breadth_label} ({breadth}%)\n"
        f"Hot    : {hot_str}\n"
        f"Portfolio: {pos_str} | Heat: {total_heat:.0%}\n"
        f"Setups : {len(picks)} found\n"
        f"Top {len(top_picks)} picks below ↓"
    )

    for pick in top_picks:
        sym  = pick['Symbol']
        rr   = round(pick['Reward₹'] / pick['Risk₹'], 1) if pick['Risk₹'] > 0 else 0

        ns, nl, headlines = sentiment_map.get(sym, (0, "N/A", []))
        news_emoji = "📰✅" if ns >= 2 else "📰⚠️" if ns <= -2 else "📰"
        news_block = f"{news_emoji} News   : {nl}"
        if headlines:
            news_block += f"\n  → {headlines[0][:60]}"

        ok15, reason15 = confirmed_map.get(sym, (True, "Not checked"))
        conf_emoji = "✅" if ok15 else "⚠️"
        conf_block = f"{conf_emoji} 15m    : {reason15}"

        news_warn = "\n⚠️ NEGATIVE NEWS — review before entering" if ns <= -2 else ""

        msg = (
            f"{'='*34}\n"
            f"🚀 {sym}  [{pick['Theme']}]\n"
            f"Setup   : {pick['Setup']}\n"
            f"Score   : {pick['Score']}\n"
            f"Candle  : {pick['Candle']}\n"
            f"MACD    : {pick['MACD']}\n"
            f"OBV     : {pick['OBV']}\n"
            f"ADX     : {pick['ADX']}\n"
            f"Squeeze : {pick['Squeeze']}\n"
            f"{news_block}\n"
            f"{conf_block}\n"
            f"Entry   : ₹{pick['Entry']}\n"
            f"Stop    : ₹{pick['Stop']} (-{pick['StopPct']}%)\n"
            f"Target  : ₹{pick['Target']} (+{pick['TargetPct']}%)\n"
            f"Qty     : {pick['Qty']} shares\n"
            f"Invested: ₹{int(pick['Invested']):,} ({pick['AcctPct']}%)\n"
            f"Risk    : ₹{int(pick['Risk₹']):,}\n"
            f"Reward  : ₹{int(pick['Reward₹']):,}\n"
            f"RR      : 1:{rr}"
            f"{news_warn}\n"
            f"{'='*34}"
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
    # Start 1hr early, end 30min after close
    time_val = now_utc.hour * 60 + now_utc.minute
    return (2 * 60 + 45) <= time_val <= (10 * 60 + 30)

if __name__ == "__main__":
    print("🚀 India Professional Swing Trading Agent")
    print(f"Started at {datetime.utcnow().strftime('%H:%M UTC')}")

    if is_market_hours():
        run_agent()
    else:
        print("Outside market hours — waiting...")

    while True:
        time.sleep(30 * 60)
        if is_market_hours():
            run_agent()
        else:
            print("Market closed — exiting.")
            break

    print("✅ Done — market closed.")
