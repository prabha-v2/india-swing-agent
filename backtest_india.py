"""
backtest_india.py — Walk-forward backtest for the India swing trading agent.

Usage:
    python backtest_india.py                            # full default list, 2 years
    python backtest_india.py --symbols TCS INFY HDFCBANK  # specific symbols (no .NS needed)
    python backtest_india.py --days 365                 # 1-year lookback
    python backtest_india.py --threshold 18             # lower score threshold
"""

import argparse
import yfinance as yf
import pandas as pd
import ta
import time
from datetime import datetime, timedelta

# =========================================
# CONFIG
# =========================================

SCORE_THRESHOLD  = 20
RR_RATIO         = 3.0
MAX_ATR_STOP     = 2.0
MAX_HOLD_DAYS    = 20
MIN_BARS         = 250

# =========================================
# INDICATOR COMPUTATION
# =========================================

def compute_indicators(df):
    df = df.copy()
    df['EMA10']  = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA20']  = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA50']  = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['RSI']    = ta.momentum.rsi(df['Close'], window=14)
    df['AvgVol'] = df['Volume'].rolling(20).mean()
    df['HH20']   = df['High'].rolling(20).max()
    df['High52'] = df['High'].rolling(252).max()

    df['TR'] = (
        df['High'] - df['Low']
    ).combine(abs(df['High'] - df['Close'].shift(1)), max
    ).combine(abs(df['Low']  - df['Close'].shift(1)), max)
    df['ATR'] = df['TR'].rolling(14).mean()

    df['ADX']    = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20, window_dev=2)
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20, window_dev=2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']

    df['OBV']       = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['OBV_EMA20'] = ta.trend.ema_indicator(df['OBV'], window=20)

    df['MACD']   = ta.trend.macd(df['Close'], window_slow=26, window_fast=12)
    df['MACD_S'] = ta.trend.macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_H'] = ta.trend.macd_diff(df['Close'], window_slow=26, window_fast=12, window_sign=9)

    df['ROC21']  = df['Close'].pct_change(21)

    return df

# =========================================
# SCORING (mirrors main agent)
# =========================================

def score_bar(df, idx, nifty_slice):
    if idx < MIN_BARS:
        return None, None

    latest = df.iloc[idx]
    prev   = df.iloc[idx - 1]

    for col in ['EMA10','EMA20','EMA50','EMA200','RSI','ATR','ADX','MACD','MACD_S']:
        if pd.isna(latest.get(col, float('nan'))):
            return None, None

    price = float(latest['Close'])
    if price < 20.0:
        return None, None

    avg_dv = float(df['Close'].iloc[idx-20:idx].mean() * df['Volume'].iloc[idx-20:idx].mean())
    if avg_dv < 20_000_000:
        return None, None

    score = 0

    # RS vs Nifty
    if len(nifty_slice) >= 63:
        s3  = float(df['Close'].iloc[idx] / df['Close'].iloc[max(idx-63, 0)] - 1)
        s6  = float(df['Close'].iloc[idx] / df['Close'].iloc[max(idx-126,0)] - 1)
        s12 = float(df['Close'].iloc[idx] / df['Close'].iloc[max(idx-252,0)] - 1)
        n3  = float(nifty_slice['Close'].iloc[-1] / nifty_slice['Close'].iloc[max(-63,-len(nifty_slice))] - 1)
        n6  = float(nifty_slice['Close'].iloc[-1] / nifty_slice['Close'].iloc[max(-126,-len(nifty_slice))] - 1)
        n12 = float(nifty_slice['Close'].iloc[-1] / nifty_slice['Close'].iloc[max(-252,-len(nifty_slice))] - 1)
        rs  = sum([s3 > n3, s6 > n6, s12 > n12])
        if rs >= 2: score += 2
        if rs == 3: score += 1

    # EMA stack
    if latest['EMA10']  > latest['EMA20']:  score += 2
    if latest['EMA20']  > latest['EMA50']:  score += 2
    if latest['EMA50']  > latest['EMA200']: score += 2
    if latest['Close']  > latest['EMA50']:  score += 1
    if latest['Close']  > latest['EMA200']: score += 2

    # RSI
    rsi      = float(latest['RSI'])
    rsi_prev = float(prev['RSI']) if not pd.isna(prev['RSI']) else rsi
    if rsi < 40:                                 score -= 2
    if rsi_prev < 50 and rsi > 50:               score += 2
    elif 55 < rsi < 75:                          score += 2
    elif rsi > 50 and rsi > rsi_prev:            score += 1
    if rsi > 80:                                 score -= 2
    elif rsi > 75:                               score -= 1

    # Breakout
    hh20     = float(df['HH20'].iloc[idx-1]) if not pd.isna(df['HH20'].iloc[idx-1]) else 0
    broke_20d = latest['Close'] > hh20
    if broke_20d: score += 2
    h52 = float(df['High52'].iloc[idx]) if not pd.isna(df['High52'].iloc[idx]) else latest['Close']
    ath_dist = latest['Close'] / h52 if h52 > 0 else 1.0
    if ath_dist > 0.90:  score += 2
    elif ath_dist > 0.80: score += 1

    # Volume
    avg_vol = float(df['AvgVol'].iloc[idx]) if not pd.isna(df['AvgVol'].iloc[idx]) else 1
    rvol = float(latest['Volume']) / avg_vol if avg_vol > 0 else 1.0
    if rvol > 2.0:   score += 2
    elif rvol > 1.5: score += 1
    if broke_20d and rvol > 1.5: score += 1

    # ATR
    atr_5ago = float(df['ATR'].iloc[idx-5]) if idx >= 5 and not pd.isna(df['ATR'].iloc[idx-5]) else float(latest['ATR'])
    if float(latest['ATR']) > atr_5ago: score += 1

    # Entry quality
    dist = (latest['Close'] - latest['EMA20']) / latest['EMA20'] if latest['EMA20'] > 0 else 0
    if dist < 0.05:   score += 2
    elif dist < 0.08: score += 1
    elif dist > 0.15: score -= 1

    # 60-day outperformance
    if idx >= 60 and len(nifty_slice) >= 60:
        sr = float(df['Close'].iloc[idx] / df['Close'].iloc[idx-60] - 1)
        nr = float(nifty_slice['Close'].iloc[-1] / nifty_slice['Close'].iloc[-60] - 1)
        if sr > nr * 1.5:  score += 2
        elif sr > nr:      score += 1

    # Distribution penalty
    rh10 = float(df['High'].iloc[idx-10:idx].max())
    if latest['Close'] < rh10 * 0.85: score -= 2

    # OBV
    obv_rising  = float(df['OBV'].iloc[idx]) > float(df['OBV_EMA20'].iloc[idx])
    obv_slope   = float(df['OBV'].iloc[idx]) - float(df['OBV'].iloc[max(idx-10,0)])
    obv_trend_p = obv_slope > 0
    if obv_rising and obv_trend_p:          score += 2
    elif obv_rising:                         score += 1
    elif not obv_trend_p and not obv_rising: score -= 2

    # MACD
    macd_now  = float(latest['MACD'])
    macd_sig  = float(latest['MACD_S'])
    macd_prev_v = float(prev['MACD']) if not pd.isna(prev['MACD']) else macd_now
    macd_sig_p  = float(prev['MACD_S']) if not pd.isna(prev['MACD_S']) else macd_sig
    macd_hist_n = float(latest['MACD_H'])
    macd_hist_p = float(prev['MACD_H']) if not pd.isna(prev['MACD_H']) else macd_hist_n

    crossed_up  = macd_prev_v < macd_sig_p and macd_now > macd_sig
    above_sig   = macd_now > macd_sig
    hist_rising = macd_hist_n > macd_hist_p and macd_hist_n > 0

    if crossed_up:                          score += 2
    elif above_sig and hist_rising:         score += 2
    elif above_sig:                         score += 1
    elif not above_sig and macd_hist_n < 0: score -= 1
    if macd_now > 0:                        score += 1

    # ADX
    adx = float(latest['ADX']) if not pd.isna(latest['ADX']) else 20.0
    if adx > 30:   score += 2
    elif adx > 20: score += 1
    elif adx < 15: score -= 2

    # ROC-21
    roc21 = float(latest['ROC21']) if not pd.isna(latest.get('ROC21', float('nan'))) else 0
    if roc21 > 0.15: score += 1
    if roc21 < 0:    score -= 1

    # BB Squeeze
    bb_squeeze = False
    if idx >= 60:
        pct20 = float(df['BB_width'].iloc[idx-60:idx].quantile(0.20))
        bb_squeeze = float(df['BB_width'].iloc[idx]) < pct20
    if bb_squeeze: score += 3

    # Setup type
    if bb_squeeze and broke_20d:
        setup = "Squeeze Breakout"
    elif broke_20d and rvol > 2.0:
        setup = "Volume Breakout"
    elif dist < 0.05 and rsi > 50:
        setup = "EMA20 Pullback"
    elif ath_dist > 0.95:
        setup = "ATH Breakout"
    elif bb_squeeze:
        setup = "Squeeze Setup"
    else:
        setup = "Trend Continuation"

    return score, setup

# =========================================
# TRADE SIMULATION
# =========================================

def simulate_trade(df, signal_idx):
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None

    entry = float(df['Open'].iloc[entry_idx])
    atr   = float(df['ATR'].iloc[signal_idx])
    if pd.isna(atr) or atr <= 0:
        return None

    five_bar_low = float(df['Low'].iloc[max(signal_idx-5, 0):signal_idx+1].min())
    atr_stop     = entry - (MAX_ATR_STOP * atr)
    stop         = max(five_bar_low, atr_stop)
    risk         = entry - stop

    if risk <= 0 or risk > entry * 0.10:
        return None

    target = entry + (risk * RR_RATIO)

    outcome    = "TIMEOUT"
    exit_price = float(df['Close'].iloc[min(entry_idx + MAX_HOLD_DAYS - 1, len(df)-1)])
    exit_bar   = MAX_HOLD_DAYS

    for i in range(1, MAX_HOLD_DAYS + 1):
        bar_idx = entry_idx + i
        if bar_idx >= len(df):
            break
        hi = float(df['High'].iloc[bar_idx])
        lo = float(df['Low'].iloc[bar_idx])
        if lo <= stop:
            outcome    = "STOPPED"
            exit_price = stop
            exit_bar   = i
            break
        if hi >= target:
            outcome    = "TARGET"
            exit_price = target
            exit_bar   = i
            break

    r_multiple = (exit_price - entry) / risk if risk > 0 else 0

    return {
        "entry":      round(entry, 2),
        "stop":       round(stop, 2),
        "target":     round(target, 2),
        "exit_price": round(exit_price, 2),
        "outcome":    outcome,
        "r_multiple": round(r_multiple, 2),
        "hold_bars":  exit_bar,
        "risk_pct":   round(risk / entry * 100, 2),
    }

# =========================================
# BACKTEST ONE SYMBOL
# =========================================

def backtest_symbol(symbol, nifty_df, lookback_days=730, threshold=SCORE_THRESHOLD):
    sym_ns = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
    start  = (datetime.now() - timedelta(days=lookback_days + 100)).strftime('%Y-%m-%d')
    try:
        df = yf.download(sym_ns, start=start, interval='1d', progress=False)
        if df is None or df.empty:
            return []
        df = df.dropna()
        df.columns = df.columns.get_level_values(0)
        if len(df) < MIN_BARS:
            return []
        df = compute_indicators(df)
    except Exception as e:
        print(f"  ⚠️ {symbol}: {e}")
        return []

    cutoff    = datetime.now() - timedelta(days=lookback_days)
    start_idx = 0
    for i, idx_val in enumerate(df.index):
        dt = pd.Timestamp(idx_val).to_pydatetime().replace(tzinfo=None)
        if dt >= cutoff:
            start_idx = i
            break

    trades         = []
    i              = start_idx
    last_entry_idx = -MAX_HOLD_DAYS

    while i < len(df) - MAX_HOLD_DAYS:
        nifty_slice = nifty_df.iloc[:min(i + 1, len(nifty_df))]
        score, setup = score_bar(df, i, nifty_slice)

        if score is not None and score >= threshold and (i - last_entry_idx) >= MAX_HOLD_DAYS:
            result = simulate_trade(df, i)
            if result:
                date_str = pd.Timestamp(df.index[i]).strftime('%Y-%m-%d')
                result.update({
                    "symbol": symbol.replace('.NS', ''),
                    "date":   date_str,
                    "score":  score,
                    "setup":  setup,
                })
                trades.append(result)
                last_entry_idx = i + 1
                i += MAX_HOLD_DAYS
        i += 1

    return trades

# =========================================
# AGGREGATE STATS
# =========================================

def print_stats(all_trades):
    if not all_trades:
        print("No trades found.")
        return

    df = pd.DataFrame(all_trades)

    total    = len(df)
    wins     = df[df['outcome'] == 'TARGET']
    losses   = df[df['outcome'] == 'STOPPED']
    timeouts = df[df['outcome'] == 'TIMEOUT']
    win_rate = len(wins) / total * 100

    avg_r_win  = wins['r_multiple'].mean()  if len(wins)   else 0
    avg_r_loss = losses['r_multiple'].mean() if len(losses) else 0
    avg_r_all  = df['r_multiple'].mean()

    gross_win  = wins['r_multiple'].sum()        if len(wins)   else 0
    gross_loss = abs(losses['r_multiple'].sum()) if len(losses) else 0
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')

    avg_hold = df['hold_bars'].mean()

    print(f"\n{'='*55}")
    print(f"INDIA BACKTEST RESULTS — {total} trades")
    print(f"{'='*55}")
    print(f"Win Rate      : {win_rate:.1f}%  ({len(wins)} wins / {len(losses)} stops / {len(timeouts)} timeouts)")
    print(f"Avg R (all)   : {avg_r_all:+.2f}R")
    print(f"Avg R (wins)  : {avg_r_win:+.2f}R")
    print(f"Avg R (losses): {avg_r_loss:+.2f}R")
    print(f"Profit Factor : {pf:.2f}")
    print(f"Avg Hold      : {avg_hold:.1f} bars")

    print(f"\n--- By Setup Type ---")
    for setup, grp in df.groupby('setup'):
        wr = len(grp[grp['outcome']=='TARGET']) / len(grp) * 100
        ar = grp['r_multiple'].mean()
        print(f"  {setup:22} {len(grp):3} trades | WR:{wr:.0f}% | AvgR:{ar:+.2f}")

    print(f"\n--- By Score Bucket ---")
    df['score_bucket'] = pd.cut(df['score'], bins=[0,22,25,28,99], labels=['20-22','23-25','26-28','29+'])
    for bucket, grp in df.groupby('score_bucket', observed=True):
        wr = len(grp[grp['outcome']=='TARGET']) / len(grp) * 100 if len(grp) else 0
        ar = grp['r_multiple'].mean() if len(grp) else 0
        print(f"  Score {bucket}: {len(grp):3} trades | WR:{wr:.0f}% | AvgR:{ar:+.2f}")

    print(f"\n--- Top 10 Best Trades ---")
    top = df.nlargest(10, 'r_multiple')[['symbol','date','setup','score','r_multiple','outcome','hold_bars']]
    print(top.to_string(index=False))

    print(f"\n--- Top 10 Worst Trades ---")
    worst = df.nsmallest(10, 'r_multiple')[['symbol','date','setup','score','r_multiple','outcome','hold_bars']]
    print(worst.to_string(index=False))

    out_file = f"backtest_india_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(out_file, index=False)
    print(f"\n📊 Full results saved to {out_file}")

# =========================================
# MAIN
# =========================================

DEFAULT_SYMBOLS = [
    "HDFCBANK","ICICIBANK","AXISBANK","SBIN","KOTAKBANK",
    "TCS","INFY","HCLTECH","WIPRO","PERSISTENT","COFORGE",
    "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","MANKIND",
    "TATAMOTORS","M&M","EICHERMOT","TVSMOTOR","BAJAJ-AUTO",
    "ADANIGREEN","TATAPOWER","NTPC","SUZLON","WAAREEENER",
    "TITAN","DMART","TRENT","ZOMATO",
    "BAJFINANCE","CHOLAFIN","MUTHOOTFIN",
    "DIXON","POLYCAB","HAVELLS","KAYNES",
    "HAL","BEL","IRFC","RVNL",
    "PIDILITIND","DEEPAKNTR","TATACHEM",
    "TATASTEEL","HINDALCO","JSWSTEEL",
    "BSE","CDSL","ANGELONE",
    "DLF","GODREJPROP","TRENT",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="India Swing Trading Backtest")
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to backtest — no .NS needed (default: built-in list of ~50)')
    parser.add_argument('--days', type=int, default=730,
                        help='Lookback in calendar days (default: 730)')
    parser.add_argument('--threshold', type=int, default=SCORE_THRESHOLD,
                        help=f'Min score threshold (default: {SCORE_THRESHOLD})')
    args = parser.parse_args()

    symbols   = list(dict.fromkeys(args.symbols or DEFAULT_SYMBOLS))
    lookback  = args.days
    threshold = args.threshold

    print(f"🔬 India Backtest: {len(symbols)} symbols | {lookback}d lookback | score>={threshold}")
    print(f"Started at {datetime.now().strftime('%H:%M:%S')}\n")

    # Download Nifty once
    print("Downloading NIFTY reference data...")
    nifty_start = (datetime.now() - timedelta(days=lookback + 300)).strftime('%Y-%m-%d')
    nifty_df    = yf.download("^NSEI", start=nifty_start, interval='1d', progress=False)
    nifty_df    = nifty_df.dropna()
    nifty_df.columns = nifty_df.columns.get_level_values(0)

    all_trades = []
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}", end='', flush=True)
        trades = backtest_symbol(sym, nifty_df, lookback_days=lookback, threshold=threshold)
        print(f" → {len(trades)} trades")
        all_trades.extend(trades)
        time.sleep(0.5)

    print_stats(all_trades)
