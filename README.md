# 🇮🇳 India Swing Trading Agent

A momentum-based swing trading scanner for NSE-listed Indian stocks.
Sends Telegram alerts during market hours — you decide what to trade.

## Themes Covered (~80 stocks)

| Theme | Key Stocks |
|-------|-----------|
| 🛡️ Defence PSU | HAL, BEL, BHEL, GRSE, Cochin Shipyard, Mazagon Dock |
| 🚂 Railways | IRFC, RVNL, IRCON, IRCTC, RailTel, Titagarh |
| 🏦 PSU Banks | SBI, PNB, Bank of Baroda, Canara, Union Bank |
| 💳 Private Banks | HDFC, ICICI, Axis, Kotak, Bajaj Finance |
| 💻 IT & AI | TCS, Infosys, Wipro, HCL, Persistent, Coforge |
| ☀️ Renewables | Adani Green, Tata Power, NTPC, SJVN, Suzlon, Waaree |
| 🚗 EV & Auto | Tata Motors, M&M, Olectra, TVS, Bajaj Auto |
| 🛒 Consumption | Titan, DMart, Trent, Nykaa |
| 📈 Capital Markets | BSE, CDSL, Angel One, MCX |
| 💊 Pharma | Sun Pharma, Dr Reddy, Cipla, Mankind |
| ⚗️ Chemicals | Pidilite, Atul, Navin Fluorine |
| 🏗️ Metals & Infra | Tata Steel, JSW, Hindalco, L&T |
| 📡 Electronics | Dixon, Amber, Kaynes, Syrma |
| 🏘️ Real Estate | DLF, Godrej Properties, Prestige |

## How It Works

1. Checks if NIFTY is above EMA50 (bullish market filter)
2. Scans all ~80 stocks for momentum signals
3. Scores each stock out of 28 (needs 12+ to qualify)
4. Sends top 6 picks to your Telegram

## Signals Used

- Relative strength vs NIFTY (3M, 6M, 12M)
- EMA stack: EMA10 > EMA20 > EMA50 > EMA200
- Price above EMA50 and EMA200
- RSI crossing above 50
- 20-day high breakout
- Within 15% of 52-week high
- Volume surge (1.5x average)
- ATR expanding
- Pullback entry (within 7% of EMA20)
- 60-day outperformance vs NIFTY

## Setup

### 1. Add Telegram Secrets
Go to repo → Settings → Secrets → Actions → New secret:
- `TELEGRAM_TOKEN` — your bot token
- `CHAT_ID` — your Telegram chat ID

### 2. Enable Actions
Go to Actions tab → Enable workflows

### 3. Test Manually
Actions tab → India Swing Trading Agent → Run workflow

## Schedule
Runs every 30 minutes from **8:15 AM to 3:30 PM IST**, Monday to Friday.
16 scans per trading day.

## Telegram Alert Format
```
==============================
🚀 HAL  [DEFENCE]
Score  : 22/28
Entry  : ₹4,250.00
Stop   : ₹3,980.00
Target : ₹4,925.00
Qty    : 27 shares
Risk   : ₹7,290
Reward : ₹18,225
RR     : 1:2.5
==============================
```

## Important
- These are **alerts only** — you place orders manually on your broker
- Works best with Zerodha, Groww, Angel One, or Upstox
- Adjust `ACCOUNT_SIZE` in the script to match your capital
- Past performance does not guarantee future results
