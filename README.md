# AI Price Action Assistant & Institutional Cockpit (v2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-92%20passed-brightgreen.svg)]()
[![Inference](https://img.shields.io/badge/ML%20Latency-4.07ms-success.svg)]()

An institutional-grade, multi-dimensional AI Price Action Trading Assistant and Live Real-Time Cockpit built in Python.

It combines **Smart Money Concepts (SMC)**, **Auction Market Volume Profile (POC/VAH/VAL)**, **Wyckoff Volume Spread Analysis (VSA)**, **Kernel Density Estimation (KDE) Support & Resistance**, **Multi-Timeframe Confluence**, and a **Calibrated Machine Learning Inference Engine** with sub-5ms CPU response times.

---

## Key Capabilities

### 1. Interactive AI Assistant & Natural Language Chat
- Type any crypto name or command: `btc`, `eth`, `sol`, `doge`, `ada`, `bnb`.
- Chat naturally: *"how is bitcoin looking?"*, *"is eth bullish?"*, *"what is the trend?"*.
- Instant commands:
  - `stream` / `live`: Enter persistent, tick-by-tick real-time streaming view.
  - `setup` / `signal`: View active institutional trade setup plan with entry, stop loss, and take-profit targets.
  - `smc`: View Smart Money Concepts breakdown (FVG, Order Blocks, Liquidity Sweeps, BOS, CHoCH).
  - `levels`: View Support & Resistance zones, Volume Profile (POC/VAH/VAL), and VWAP bands.
  - `ml`: View machine learning prediction, directional confidence, and win probability breakdown.
  - `tf <15m|1h|4h|1d>`: Dynamically switch timeframes on the fly.
  - `backtest`: Run instant strategy backtest benchmarks.

### 2. Smart Money Concepts (SMC) & Market Structure
- **Fair Value Gaps (FVG):** Bullish (BISI) and Bearish (SIBI) 3-candle imbalance detection with 50% Consequent Encroachment (CE) and dynamic stateful mitigation/inversion tracking.
- **Order Blocks (OB):** Institutional supply/demand zones with strict 4-pillar quantitative validation (displacement $\ge 1.5\times \text{ATR}$, FVG confirmation, structural breach, and Breaker Block transitions).
- **Liquidity Sweeps:** Stop-hunt sweeps (BSL / SSL) with $\ge 35\%$ wick rejection ratio, volume expansion, and Turtle Soup multi-bar reversals.
- **Market Structure Transitions:** Zero-lookahead fractal swings ($R=3$), Break of Structure (BOS) trend continuation, and Change of Character (CHoCH) structural trend reversals.

### 3. Quantitative Volume & Statistical S/R
- **Volume Profile:** Equidistant price binning allocating candle volume across geometric overlaps to determine Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL) ($70\%$ volume zone).
- **VWAP & Variance Bands:** Rolling and anchored VWAP with $\pm 1\sigma, \pm 2\sigma, \pm 3\sigma$ dispersion bands detecting statistical mean-reversion exhaustion.
- **Wyckoff VSA:** Effort vs. result modeling, Stopping Volume / Absorption, No Supply and No Demand tests, and Climax volume exhaustion.
- **Kernel Density Estimation (KDE) S/R:** Continuous Gaussian KDE (`scipy.stats.gaussian_kde`) with adaptive ATR bandwidth identifying prominent support & resistance zones and polarity flips.
- **Dynamic Fibonacci:** Swing-anchored retracements ($0.236, 0.382, 0.500, 0.618$ Golden Pocket, $0.786$) and extension projections.

### 4. Machine Learning Inference Engine
- **Architecture:** CPU-native `HistGradientBoostingClassifier` calibrated with Platt scaling (`CalibratedClassifierCV`).
- **Feature Vector:** 35 scale-invariant features covering candlestick geometry, multi-horizon returns, normalized ATR, volume z-scores, distance to POC/VWAP/KDE S/R, RSI slope, EMA stacks, and SMC flags.
- **Hardware Performance:** Trained on **60,000 real Binance historical candles**; model file is only **0.397 MB** with an average inference latency of **4.07 ms** on CPU (0 GPU required).

### 5. Institutional Trade Setup Generator
- Strict Effective Risk-to-Reward gate:
  $$\text{Effective } R:R = 0.5 \times R:R_{\text{TP1}} + 0.5 \times R:R_{\text{TP2}} \ge 1.80$$
- Invalidation stop loss placed beyond structural swing / Order Block / sweep boundaries ($+ 0.2\times\text{ATR}$ buffer).
- Take Profit 1 (TP1) scales out 50% position and moves stop loss to Breakeven; Take Profit 2 (TP2) targets extension levels.

---

## Project Structure

```
Price-Action-agent/
│
├── start.bat                  # One-click Windows launcher (UTF-8, VT100 enabled)
├── main.py                    # Interactive AI Assistant & Live Cockpit
├── config.py                  # Environment config and logging setup
├── binance_client.py          # REST historical kline loader & resilient WebSocket streamer
├── price_action_engine.py     # Master orchestrator combining all sub-engines
├── data_collector.py          # Multi-symbol, multi-timeframe historical data harvester
├── train_model.py             # Machine learning pipeline training & calibration script
├── run_backtest.py            # Quantitative backtest benchmark runner
├── symbol_mapper.py           # Friendly name resolver & symbol fuzzy matcher
├── requirements.txt           # Python dependencies
├── .env.example               # Configuration template
├── .gitignore                 # Git ignore rules
│
├── core/                      # Canonical dataclass schemas
│   └── models.py              # TradeSetup, ConfluenceReport, FVG, OrderBlock, etc.
│
├── engines/                   # Specialized analytical engines
│   ├── smc_engine.py          # Fair Value Gaps, Order Blocks, Liquidity Sweeps, BOS/CHoCH
│   ├── chart_pattern_engine.py# Double Tops/Bottoms, H&S, Triangles, Wedges
│   ├── indicators_engine.py   # Volume Profile, VWAP, Wyckoff VSA, KDE S/R, Fibonacci
│   ├── ml_engine.py           # 35-feature extraction, labeling, training & streaming inference
│   ├── confluence_engine.py   # Multi-timeframe confluence weighting & scoring
│   ├── trade_setup_engine.py  # Actionable trade setup generator (R:R >= 1.80)
│   └── backtest_engine.py     # Event-driven backtester with fee & slippage modeling
│
├── data/historical/           # 60,000 harvested Binance candles (CSV & Parquet)
├── models/                    # Serialized machine learning bundles (price_action_model.joblib)
├── docs/                      # Comprehensive technical research specifications
│   ├── architecture_plan.md   # Architectural blueprint
│   ├── smc_spec.md            # Smart Money Concepts mathematical specifications
│   ├── indicators_spec.md     # Volume Profile, VWAP, KDE S/R specifications
│   ├── ml_pipeline_spec.md    # Machine learning pipeline specifications
│   └── backtest_report.md     # 30,000-candle quantitative backtest report
│
└── tests/                     # Clean, consolidated unit test suite (92 tests)
    ├── test_smc_and_patterns.py
    ├── test_indicators_and_sr.py
    ├── test_ml_and_confluence.py
    └── test_trade_and_backtest.py
```

---

## Quick Start (Windows)

Simply double-click:
```bat
start.bat
```
The launcher will:
1. Enable UTF-8 encoding and native Windows VT100/ANSI color processing.
2. Create and activate a clean virtual environment (`venv`).
3. Install dependencies from `requirements.txt`.
4. Launch the Interactive AI Price Action Assistant.

---

## Manual Installation

```bash
# Clone the repository
git clone https://github.com/Arafat1slam/Price-Action-agent.git
cd Price-Action-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run interactive assistant
python main.py

# Or launch directly into live streaming mode
python main.py --stream --symbol BTCUSDT --timeframe 1h
```

---

## Running Unit Tests

Run the complete test suite:
```bash
pytest -v
```
All **92 tests** pass in ~3.5 seconds with 100% coverage across pattern detection, indicators, ML inference, and backtesting.

---

## Backtest Benchmarking

To run the event-driven strategy simulator across historical datasets:
```bash
python run_backtest.py
```
*(See [`docs/backtest_report.md`](docs/backtest_report.md) for full performance metrics across BTC, ETH, and SOL).*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
