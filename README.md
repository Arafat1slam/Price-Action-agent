# Live Price Action Scanner (Persistent Agent)

A standalone, lightweight Python CLI application that connects directly to the Binance API and continuously monitors live price action for any cryptocurrency trading pair (e.g. BTCUSDT, ETHUSDT, SOLUSDT).

Unlike a one-shot analysis script, this tool runs **persistently** — monitoring every candle from open to close in real time via Binance WebSocket streams — and outputs live price-action-based predictions with confidence scores, bias (Bullish / Bearish / Neutral), detected candlestick patterns, and price targets.

Designed to be lightweight and CPU-friendly (100% rule-based and statistical, no GPU or heavy ML dependencies required).

---

## Features

- **Persistent Live Streaming:** Watches live Binance Kline/Candlestick WebSocket streams (`<symbol>@kline_<interval>`) tick-by-tick.
- **Forming vs Confirmed Candle Analysis:** Real-time updates as the live candle develops, with clear indicators distinguishing forming candles from confirmed closed candles.
- **Candlestick Pattern Recognition:**
  - Doji (Standard, Dragonfly, Gravestone)
  - Hammer & Inverted Hammer
  - Bullish & Bearish Engulfing
  - Pin Bar / Long Wick Rejections
  - Inside Bar & Outside Bar
- **Market Structure & Trend Detection:**
  - Higher Highs & Higher Lows (Uptrend)
  - Lower Highs & Lower Lows (Downtrend)
  - Sideways / Consolidation
- **Dynamic Support & Resistance:**
  - Automatic swing high/low clustering within dynamic tolerance zones.
  - Identification of nearest support, resistance, and breakout/bounce levels.
- **Volume & Momentum Verification:**
  - Relative Volume (RVOL) vs 20-period moving average to detect institutional volume spikes.
  - Consecutive directional candle streak and exhaustion/slowdown detection.
- **Natural Language Market Mapper:**
  - Enter friendly names like `bitcoin`, `eth`, `solana`, `doge`, or standard pairs like `BTCUSDT`.
  - Automatic symbol suggestion on typos.
- **Zero-Config Windows Launcher:** Single-click execution via `start.bat`.

---

## Project Structure

```
Price-Action-agent/
│
├── start.bat                  # One-click Windows launcher (creates venv, installs deps, runs app)
├── main.py                    # Main interactive CLI application
├── config.py                  # Environment config and logging setup
├── binance_client.py          # REST historical kline fetcher & resilient WebSocket streaming client
├── price_action_engine.py     # Core pattern detection, trend, S/R, volume & momentum engine
├── symbol_mapper.py           # Friendly name resolver & symbol fuzzy matcher
├── requirements.txt           # Minimal Python dependencies
├── .env.example               # Configuration template
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
└── tests/                     # Unit test suite
    ├── test_symbol_mapper.py
    └── test_engine.py
```

---

## Quick Start (Windows)

Simply double-click:
```bat
start.bat
```
The script will automatically:
1. Verify Python installation.
2. Create an isolated virtual environment (`venv`).
3. Install required dependencies.
4. Launch the Live Price Action Scanner CLI.

---

## Manual Installation

### 1. Clone the repository
```bash
git clone https://github.com/Arafat1slam/Price-Action-agent.git
cd Price-Action-agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Public market data streaming does not require API keys, but keys can be added for private API features).*

### 5. Run Scanner
```bash
python main.py
```

---

## Sample CLI Interaction

```text
======================================================
           LIVE PRICE ACTION SCANNER
  Persistent Real-Time Candlestick & Pattern Analysis
======================================================
Enter market (e.g. bitcoin, eth, sol, BTCUSDT): bitcoin
 Resolved symbol: BTCUSDT
Enter timeframe [1m / 3m / 5m / 15m / 30m / 1h] (default: 1h): 1h
 Active timeframe: 1h

Connecting to Binance REST API for historical BTCUSDT (1h) candles...
Connected! Seeded with 150 historical candles.
Starting real-time WebSocket stream for BTCUSDT (1h)...

[14:32:10] Live candle forming | Price: $64,250.00 | Bias: BULLISH (72%) | Pattern: Bullish Engulfing (forming)
           Trend: Uptrend (HH/HL) | Volume Rising (1.4x avg) | Momentum: 2 consecutive bullish candles
           Support: 63,500.00 | Resistance: 65,100.00 | Rough Target: 65,100.00
```

---

## Running Unit Tests

Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
