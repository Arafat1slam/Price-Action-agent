# Live Price Action Scanner — Project Specification

## 1. Overview
Build a standalone, lightweight Python CLI application that connects to the Binance API and continuously monitors live price action for a user-selected trading pair (e.g., BTCUSDT, ETHUSDT). Unlike a one-shot analysis tool, this must run **persistently** — watching every candle from open to close in real time — and output live price-action-based predictions with a confidence score and bias (bullish/bearish/neutral).

This is a standalone test project, separate from any existing trading system. No local heavy ML model training is required — this uses rule-based / statistical price action pattern detection, since the target machine has no GPU (16 GB RAM, Ryzen 5 5600, no dedicated graphics card).

## 2. Objective
- Launch via a single `start.bat` file.
- On launch, show a simple, user-friendly command-line "chat-like" interface.
- User types a market/pair name (e.g., `bitcoin`, `ethereum`, or `BTCUSDT`).
- App connects to Binance, resolves the correct trading pair symbol, and starts a **continuous live scan** — not a single pass.
- App tracks every candle from open to close, tick by tick, and continuously re-evaluates price action patterns as the candle forms.
- On each significant update (or at a configurable interval), it prints an updated read: bias (bullish/bearish/neutral), confidence %, key pattern detected, and a rough price target.

## 3. Tech Stack (recommended)
- **Language:** Python 3.10+
- **Binance connectivity:** `python-binance` library or raw `websocket-client` + `requests`
- **WebSocket stream:** Binance Kline/Candlestick WebSocket stream (`<symbol>@kline_<interval>`) — this stream pushes updates continuously while a candle is open (`"x": false`) and marks it closed when finished (`"x": true`). This is what enables "always on" live tracking rather than one-shot scans.
- **Data handling:** `pandas` for rolling candle history and indicator calculations
- **CLI UI:** plain `input()`/`print()` loop, or `rich` library for a nicer colored terminal UI (optional, keep it lightweight)
- **Config:** `.env` file for Binance API key/secret (python-dotenv)
- No GPU/deep learning frameworks required for this version.

## 4. Core Features
1. CLI startup screen with instructions.
2. Market/pair selection (accepts common names like "bitcoin" → maps to BTCUSDT).
3. Timeframe selection (default 1h, but allow user to choose: 1m, 5m, 15m, 1h, 4h, 1d).
4. Persistent Binance WebSocket connection — stays alive until user exits.
5. Live, continuous price action analysis engine (see Section 6).
6. Real-time console output showing updated bias/confidence as the candle evolves.
7. Graceful error handling and auto-reconnect on dropped connections.

## 5. File Structure
```
price-action-scanner/
│
├── start.bat                  # Windows launcher
├── main.py                    # Entry point — CLI loop
├── config.py                  # Loads API keys from .env
├── .env                       # BINANCE_API_KEY, BINANCE_API_SECRET (not committed)
├── binance_client.py          # Handles REST + WebSocket connections
├── price_action_engine.py     # Core pattern detection & prediction logic
├── symbol_mapper.py           # Maps friendly names ("bitcoin") to symbols ("BTCUSDT")
├── requirements.txt
└── README.md
```

## 6. Price Action Analysis Engine — Detailed Logic

The engine should maintain a rolling window of the last N candles (e.g., 100–200) for the selected symbol/timeframe, updating live as new ticks arrive, and re-run its analysis on every meaningful update.

### 6.1 Candlestick Pattern Detection (per candle, including the still-forming live candle)
Detect and flag these patterns using standard OHLC ratio rules:
- Doji (open ≈ close, small body vs range)
- Hammer / Inverted Hammer
- Bullish/Bearish Engulfing (compare current candle body to previous candle body)
- Pin Bar / Long Wick rejection candles
- Inside Bar / Outside Bar

### 6.2 Trend Structure Analysis
- Identify recent swing highs/lows over the rolling window.
- Classify trend as:
  - Uptrend: higher highs + higher lows
  - Downtrend: lower highs + lower lows
  - Range/Sideways: no clear higher/lower structure
- Track how many recent swings confirm the current structure (used for confidence weighting).

### 6.3 Support / Resistance Detection
- Detect price zones with repeated highs/lows (cluster nearby swing points within a tolerance %).
- Flag when live price approaches or breaks a known support/resistance zone.

### 6.4 Volume Confirmation
- Compare current candle volume to the rolling average volume.
- Flag volume spikes that coincide with breakout/reversal candles — these increase confidence weighting.

### 6.5 Momentum / Continuation Check
- Count consecutive same-direction candles (bullish/bearish streak).
- Detect momentum slowdown (shrinking candle bodies near a trend extreme) as an early reversal signal.

### 6.6 Live (Forming) Candle Handling
- While the current candle is still open (`"x": false` in the kline stream), continuously re-evaluate its shape (wick length, body size, direction) against the pattern rules above as price updates arrive.
- Clearly label output as "Live candle forming" vs "Confirmed closed candle" so the user knows the difference in reliability.

### 6.7 Output Synthesis
Combine the above signals into:
- **Bias:** Bullish / Bearish / Neutral
- **Confidence %:** weighted score based on how many independent signals agree (pattern + trend + S/R + volume + momentum)
- **Key pattern(s) detected**
- **Rough price target:** next relevant support/resistance zone in the direction of bias

## 7. Sample CLI Interaction
```
==============================
 LIVE PRICE ACTION SCANNER
==============================
Enter market (e.g. bitcoin, ethereum): bitcoin
Enter timeframe [1m/5m/15m/1h/4h/1d] (default 1h): 1h

Connecting to Binance... connected.
Streaming live BTCUSDT 1h candles...

[14:32:10] Live candle forming | Bias: BULLISH (68%) | Pattern: Bullish engulfing forming
           Trend: Uptrend (HH/HL) | Near resistance: 63,200
           Rough target: 63,800

[14:33:05] Live candle forming | Bias: BULLISH (71%) | Volume rising | Pattern: Bullish engulfing confirmed
...
```

## 8. Configuration
`.env` file:
```
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```
(Public market data does not strictly require API keys, but include support for authenticated requests for future extension.)

## 9. Error Handling & Reliability
- Auto-reconnect WebSocket on disconnect (exponential backoff).
- Handle invalid symbol/pair names gracefully — suggest closest match.
- Handle Binance API rate limits/errors without crashing the app.
- Log errors to a local `logs/` folder for debugging.

## 10. start.bat
```bat
@echo off
cd /d %~dp0
python -m venv venv 2>nul
call venv\Scripts\activate
pip install -r requirements.txt --quiet
python main.py
pause
```

## 11. Non-Goals (for this version)
- No local ML/deep learning model training.
- No order execution/trading — this is a read-only signal/analysis tool.
- No GUI — CLI only for this test version.

## 12. Build Instructions for the Coding Agent
Implement each file listed in Section 5 following the logic in Sections 6–10. Prioritize:
1. Working Binance WebSocket live connection first.
2. Then rolling candle storage.
3. Then price action pattern detection functions (unit-testable independently).
4. Then CLI loop tying it all together.
Keep the code modular so individual pattern-detection functions can be tuned or replaced later.