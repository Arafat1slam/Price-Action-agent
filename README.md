# AI Price Action Assistant & Institutional Cockpit (v2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 155 Passed](https://img.shields.io/badge/unit%20tests-155%20passed-brightgreen.svg)]()
[![Inference Latency](https://img.shields.io/badge/ML%20Latency-4.07ms-success.svg)]()
[![Code Architecture](https://img.shields.io/badge/Architecture-Modular%20Engines-orange.svg)]()

> **Developed by [Arafat Islam](https://github.com/Arafat1slam)**  
> *Institutional-Grade Multi-Dimensional Price Action Assistant, Live Confluence Cockpit & Event-Driven Trading Simulator.*

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [What's New in v2.0](#-whats-new-in-v20)
3. [Key Pillars & Engines](#-key-pillars--engines)
4. [Quick Start (One-Click Windows)](#-quick-start-one-click-windows)
5. [Manual Installation (Windows, Linux, macOS)](#-manual-installation-windows-linux-macos)
6. [Interactive Terminal & Assistant Usage](#-interactive-terminal--assistant-usage)
7. [Command-Line Options (CLI)](#-command-line-options-cli)
8. [Data Collection, Training & Backtesting](#-data-collection-training--backtesting)
9. [Running the Unit Test Suite](#-running-the-unit-test-suite)
10. [Repository Structure](#-repository-structure)
11. [Configuration (`.env`)](#-configuration-env)
12. [Developer Attribution & Anti-Tamper Security](#-developer-attribution--anti-tamper-security)
13. [License & Disclaimer](#-license--disclaimer)

---

## 🌟 Overview

The **AI Price Action Assistant & Institutional Cockpit** is an advanced, production-grade terminal application engineered for cryptocurrency market analysis, live trade execution planning, and quantitative backtesting.

Unlike conventional technical analysis tools that rely on lagging indicator crossovers, this engine combines **first-principles Auction Market Theory**, **Smart Money Concepts (SMC)**, **Volume Spread Analysis (VSA)**, and **Calibrated Machine Learning** to detect real institutional footprints (liquidity hunts, imbalances, order absorption, and structural breaks).

### Why use this engine?
- **Zero Terminal Flicker (Zero-Blink):** Smooth in-place ANSI rendering (`\033[H`) with live tick updates without screen blinking or runaway console scroll.
- **Synchronized Multi-Timeframe Confluence:** Simultaneously tracks **5m, 15m, 1h, and 4h** timeframes to ensure lower-timeframe trades align with higher-timeframe order flow.
- **Market Regime Detection:** Distinguishes between trending, ranging, breakout, and choppy conditions to recommend the optimal trading playbook.
- **Limit Execution State Machine:** Eliminates FOMO by holding signals in `WAITING_CONFIRMATION` until price retraces into the optimal entry zone.
- **Position-State Awareness:** Suppresses opposite signals while a trade is open to prevent whipsawing, and warns of early momentum exhaustion.
- **Ultra-Fast ML Inference:** Sub-5ms CPU inference latency using a calibrated `HistGradientBoostingClassifier` trained on 60,000 historical candles.

---

## 🚀 What's New in v2.0

| Feature | Description |
|---|---|
| 🧙 **Interactive Setup Wizard** | Prompts on launch for market (`btc`, `eth`, `sol`, etc.) and style (`1` Scalp 5m, `2` Intraday 15m/1h, `3` Swing 4h/1d). |
| ⏱️ **Synchronized MTF Engine** | 4-timeframe concurrency (4h: 35%, 1h: 30%, 15m: 20%, 5m: 15%) evaluating aligned market bias. |
| 🧭 **Market Regime Classifier** | Categorizes market into `TRENDING_BULL`, `TRENDING_BEAR`, `RANGING`, `BREAKOUT`, and `HIGH_VOLATILITY_CHOP`. |
| 💯 **0–100 Trade Quality Score** | Unified grading system: `A+` (85–100), `A` (75–84), `B` (65–74), `C` (45–64), and `FILTERED` (<45). |
| 🎯 **Limit Confirmation Engine** | Real-time monitoring of candle opens, highs, lows, closes, and volume before confirming trade entries. |
| 🎮 **Position State & Live Keys** | Press `1` in live mode to mark trade entered, `2` to close/flatten. Live PnL and R:R tracking. |
| 🔬 **Purged Walk-Forward ML** | Validates models with time-series splits and embargo buffers, reporting Out-of-Sample Sharpe and Max DD. |
| 📉 **Realistic Backtesting** | Volatility dynamic spread, non-linear market impact slippage, latency drift (85ms), and intracandle path reconstruction (`O->L->H->C`). |
| 🛡️ **Author Security & Integrity** | Permanent cryptographic protection of developer attribution (`Arafat Islam`) via SHA-256 validation. |

---

## 🧠 Key Pillars & Engines

```
                               ┌──────────────────────────────────────────────┐
                               │           LIVE MARKET DATA FEED              │
                               │   Binance REST (klines) + Resilient WebSocket│
                               └──────────────────────┬───────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 ┌───────────────────────────┐
         │        SMC ENGINE         │                                 │    INDICATORS & VOLUME    │
         │ • Fair Value Gaps (FVG)   │                                 │ • Volume Profile (POC/VAH)│
         │ • Order Blocks (OB)       │                                 │ • VWAP ±1/2/3σ Bands      │
         │ • Liquidity Sweeps (BSL)  │                                 │ • Wyckoff VSA Analysis    │
         │ • BOS & CHoCH Structure   │                                 │ • KDE Support/Resistance  │
         └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │    CONFLUENCE SYNTHESIS     │
                                       │ • Multi-Timeframe (5m-4h)   │
                                       │ • Market Regime Detection   │
                                       │ • Calibrated ML Inference   │
                                       └──────────────┬──────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
         ┌───────────────────────────┐                                 ┌───────────────────────────┐
         │    TRADE SETUP ENGINE     │                                 │   POSITION STATE MGR      │
         │ • Quality Score (0-100)   │                                 │ • Whipsaw Suppression     │
         │ • Min R:R Gate (>= 1.80)  │                                 │ • Peak R:R Drawdown Alert │
         │ • Confirmation Retest     │                                 │ • Interactive [1]/[2] Keys│
         └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │   SPACIOUS LIVE COCKPIT     │
                                       │  Zero-Blink Rich Terminal UI│
                                       └─────────────────────────────┘
```

### 1. Smart Money Concepts (SMC)
- **Fair Value Gaps (FVG):** Detects 3-candle price imbalances for Bullish (BISI) and Bearish (SIBI) gaps, tracks Consequent Encroachment (50% midpoint), and monitors real-time mitigation and inversion flips.
- **Order Blocks (OB):** High-probability institutional order blocks verified by 4 strict quantitative pillars: impulsive displacement $\ge 1.5\times\text{ATR}$, structural break confirmation, FVG generation, and Breaker Block flips.
- **Liquidity Sweeps:** Identifies stop-hunt sweeps above Buy-Side Liquidity (BSL) or below Sell-Side Liquidity (SSL) with $\ge 35\%$ wick rejection and volume expansion.
- **Market Structure:** Zero-lookahead fractal swings ($R=3$), Break of Structure (BOS) for trend continuation, and Change of Character (CHoCH) for structural reversals confirmed by candle body closes.

### 2. Volume Profile & Statistical Support/Resistance
- **Volume Profile (VP):** Equidistant geometric binning mapping candle volume to pinpoint the **Point of Control (POC)** and the **Value Area (VAH & VAL)** containing 70% of traded volume.
- **VWAP & Dispersion Bands:** Session-anchored and rolling VWAP with $\pm 1\sigma, \pm 2\sigma, \pm 3\sigma$ standard deviation bands for statistical mean-reversion exhaustion.
- **Wyckoff Volume Spread Analysis (VSA):** Analyzes volume vs. candle spread to detect Stopping Volume / Absorption, No Supply and No Demand tests, and Climax volume exhaustion.
- **Kernel Density Estimation (KDE) S/R:** Continuous Gaussian KDE (`scipy.stats.gaussian_kde`) with adaptive ATR bandwidth finding prominent horizontal support/resistance zones.

### 3. Calibrated Machine Learning Engine
- **Model:** CPU-native `HistGradientBoostingClassifier` with Platt scaling calibration (`CalibratedClassifierCV`).
- **Feature Set:** 35 scale-invariant features extracting candle geometry, multi-horizon returns, normalized ATR, volume z-scores, distance to POC/VWAP/KDE S/R, RSI slope, and SMC flags.
- **Performance:** Model file size is only **0.397 MB** and executes in **4.07 ms** per tick on a single CPU core.
- **Validation:** Evaluated via Purged Walk-Forward cross-validation with embargo buffers to eliminate lookahead bias.

### 4. Trade Quality Scoring & Risk Engine
- **Quality Score (0–100):** Synthesizes SMC alignment (25%), Structure & Regime (25%), Multi-Timeframe agreement (20%), ML probability (15%), and Volume profile/VSA (15%).
- **Grades:**
  - `A+` (85–100): Premium institutional setup with highest confluence.
  - `A` (75–84): Strong setup aligned with higher-timeframe order flow.
  - `B` (65–74): Standard tradable setup with valid structural invalidation.
  - `C` (45–64): Marginal setup with lower confluence; requires strict limit entry.
  - `FILTERED` (<45): Blocked from execution to protect capital.
- **Risk-Reward Gate:** Enforces $\text{Effective } R:R \ge 1.80$, scaling out 50% at TP1 (moving stop loss to Breakeven) and letting a runner target TP2.

---

## ⚡ Quick Start (One-Click Windows)

If you are on Windows, setup and launch is 100% automated:

1. **Download or Clone the Repository:**
   ```bat
   git clone https://github.com/Arafat1slam/Price-Action-agent.git
   cd Price-Action-agent
   ```
2. **Double-Click `start.bat`:**
   - Automatically checks for Python 3.10+.
   - Sets up the isolated virtual environment (`venv`).
   - Automatically installs/updates all required packages.
   - Configures native Windows ANSI VT100 colors and UTF-8 encoding.
   - Launches the Interactive Setup Wizard.

---

## 🛠️ Manual Installation (Windows, Linux, macOS)

### 1. Prerequisites
- **Python 3.10 or higher** (Python 3.11 or 3.12 recommended).
- **Git** installed on your system.

### 2. Clone & Setup Environment

```bash
# Clone repository
git clone https://github.com/Arafat1slam/Price-Action-agent.git
cd Price-Action-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On macOS / Linux:
source venv/bin/activate

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Interactive Terminal & Assistant Usage

Start the interactive terminal:
```bash
python main.py
```

### The Setup Wizard
On startup, the assistant prompts two simple questions:
```
? Which market would you like to scan? (e.g. btc, sol, eth, bnb, doge, or raw BTCUSDT)
> btc

? Which trading style would you prefer?
  1) Scalp    (5m focus — fast triggers & tighter stops)
  2) Intraday (15m / 1h focus — balanced swings & order blocks)
  3) Swing    (4h / 1d focus — macro trends & major S/R)
> 2
```

### In-Session Commands
Inside the assistant, you can type natural inquiries or quick commands:

| Command | Action |
|---|---|
| `stream` or `live` | Enters the persistent, real-time live cockpit. |
| `setup` or `signal`| Shows the active trade setup card, entry price, SL, TP1, and TP2. |
| `score` or `quality`| Shows the 0–100 trade quality breakdown and assigned grade. |
| `regime` | Displays current market regime (`TRENDING_BULL`, `RANGING`, etc.). |
| `smc` | Shows active Fair Value Gaps, Order Blocks, Liquidity Sweeps, and BOS/CHoCH. |
| `levels` | Displays horizontal KDE Support/Resistance, Volume Profile POC, and VWAP bands. |
| `ml` | Displays machine learning probability predictions and model confidence. |
| `tf <timeframe>` | Switches active timeframe (e.g. `tf 15m`, `tf 1h`, `tf 4h`). |
| `btc`, `eth`, `sol` | Instantly switches the active market symbol. |
| `backtest` | Executes an instant strategy backtest on the current symbol. |
| `help` | Lists all available commands and keyboard shortcuts. |
| `exit` or `quit` | Exits the assistant safely. |

### Live Cockpit Interactive Keys
While running in live streaming mode (`stream`):
- **Press `1`:** Confirm taking the active trade setup (switches state to `OPEN_LONG` or `OPEN_SHORT`).
- **Press `2`:** Manually close / flatten the position.
- **Press `Ctrl + C`:** Return safely to the assistant chat prompt.

---

## ⚙️ Command-Line Options (CLI)

You can launch directly with custom parameters:

```bash
# Launch directly into real-time live streaming mode for SOL
python main.py --stream --symbol SOLUSDT --timeframe 15m

# Run in single-pass demo mode (inspects current state and exits)
python main.py --demo --symbol ETHUSDT

# Skip interactive wizard and use default/CLI parameters
python main.py --no-wizard --symbol BTCUSDT --timeframe 1h
```

---

## 📊 Data Collection, Training & Backtesting

### 1. Harvest Historical Data
Download thousands of real Binance candles across symbols and timeframes:
```bash
python data_collector.py --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --timeframes 15m 1h 4h --candles 5000
```
Data is cleaned, validated, and saved into `data/historical/`.

### 2. Train Machine Learning Models
Train the calibrated `HistGradientBoostingClassifier` on historical datasets:
```bash
# Standard training and calibration
python train_model.py

# Train with Purged Walk-Forward Cross-Validation report
python train_model.py --walk-forward
```
The trained model is saved to `models/price_action_model.joblib`.

### 3. Run Realistic Strategy Backtests
Benchmark the strategy using the institutional event-driven backtesting engine:
```bash
# Run default multi-symbol benchmark
python run_backtest.py

# Custom backtest parameters
python run_backtest.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --timeframes 1h,15m --capital 10000 --risk 0.015 --latency 85.0
```
Generates a detailed quantitative report in [`docs/backtest_report.md`](docs/backtest_report.md).

---

## 🧪 Running the Unit Test Suite

The project includes an exhaustive, production-grade test suite covering every engine, calculation, and safety check:

```bash
# Run all tests
pytest -v

# Run with line summary and timings
pytest --tb=line -q
```

### Test Suite Structure (155 Tests Total)
```
tests/
├── test_author_integrity.py     11 passed  (Cryptographic developer attribution & security)
├── test_indicators_and_sr.py    31 passed  (Volume Profile, VWAP, VSA, KDE S/R, Fibonacci, Regime)
├── test_ml_and_confluence.py    28 passed  (ML Feature extraction, Walk-forward validation, MTF)
├── test_smc_and_patterns.py     27 passed  (FVG, Order Blocks, Liquidity Sweeps, BOS/CHoCH, Patterns)
└── test_trade_and_backtest.py   58 passed  (Trade Setup Engine, Realistic Backtest, Scoring, Positions)
─────────────────────────────────────────────────────────────────────────────────────────────────
TOTAL: 155 passed in ~16 seconds (100% Pass Rate)
```

---

## 📁 Repository Structure

```
Price-Action-agent/
│
├── start.bat                  # One-click Windows launcher (UTF-8 & VT100 auto-config)
├── main.py                    # Interactive AI Assistant & Spacious Live Cockpit
├── config.py                  # Global settings, Binance endpoints, and logger
├── binance_client.py          # Resilient REST kline harvester & multi-timeframe WebSocket
├── price_action_engine.py     # Master orchestrator integrating all analytical sub-engines
├── data_collector.py          # Historical data downloader (BTC, ETH, SOL, BNB)
├── train_model.py             # ML model trainer with purged walk-forward cross-validation
├── run_backtest.py            # Quantitative event-driven backtest benchmark runner
├── symbol_mapper.py           # Natural name resolver (e.g. "bitcoin" -> "BTCUSDT")
├── requirements.txt           # Production dependencies
├── .env.example               # Environment template
├── LICENSE                    # MIT License
├── README.md                  # Comprehensive documentation
│
├── core/                      # Canonical data models & security
│   ├── models.py              # Dataclasses: TradeSetup, ConfluenceReport, ActivePosition, etc.
│   └── security.py            # Cryptographic author validation & anti-tamper protection
│
├── engines/                   # Specialized modular analytical engines
│   ├── smc_engine.py          # FVG, Order Blocks, Liquidity Sweeps, BOS, CHoCH
│   ├── chart_pattern_engine.py# Double Tops/Bottoms, H&S, Triangles, Wedges
│   ├── indicators_engine.py   # Volume Profile (POC/VAH/VAL), VWAP bands, VSA, KDE S/R, Fibo
│   ├── regime_engine.py       # Market Regime Classifier (Trend, Range, Breakout, Chop)
│   ├── ml_engine.py           # 35-feature extraction, purged walk-forward validation
│   ├── ml_predictor.py        # Sub-5ms calibrated streaming inference runtime
│   ├── confluence_engine.py   # Multi-timeframe confluence weighting & scoring
│   ├── trade_setup_engine.py  # Trade setups, 0-100 quality scoring & position state manager
│   └── backtest_engine.py     # Realistic backtester (dynamic spread, volume impact, latency)
│
├── data/historical/           # 60,000 harvested Binance candles (CSV format)
├── models/                    # Serialized machine learning bundles (.joblib)
├── docs/                      # Technical specifications & research blueprints
│   ├── architecture_plan.md   # Architectural specifications
│   ├── smc_spec.md            # Smart Money Concepts mathematical specifications
│   ├── indicators_spec.md     # Volume Profile & Statistical Indicator specs
│   ├── ml_pipeline_spec.md    # Feature engineering & ML specs
│   └── backtest_report.md     # Multi-symbol quantitative backtesting benchmarks
│
└── tests/                     # Clean, consolidated unit test suite (155 tests)
    ├── test_author_integrity.py
    ├── test_indicators_and_sr.py
    ├── test_ml_and_confluence.py
    ├── test_smc_and_patterns.py
    └── test_trade_and_backtest.py
```

---

## 🔧 Configuration (`.env`)

You can create a `.env` file in the root directory by copying `.env.example`:

```env
# Optional: Binance API Credentials (Only needed for private account features;
# public market data and live WebSocket streaming work 100% without API keys).
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Application Configuration
DEFAULT_SYMBOL=BTCUSDT
DEFAULT_TIMEFRAME=1h
LOG_LEVEL=INFO
```

---

## 🛡️ Developer Attribution & Anti-Tamper Security

This software was developed and authored by **Arafat Islam**.

To preserve open-source integrity and author credit across distributions, the application includes an active cryptographic tamper-detection system in [`core/security.py`](file:///d:/ALL%20PROJECT/AI/Price%20action/core/security.py). 

- The author attribution (`Arafat Islam`) is cryptographically verified against SHA-256 hash assertions at runtime.
- The engines (`PriceActionEngine`, `TradeSetupEngine`, `PositionStateManager`, and the live terminal) automatically verify author integrity.
- Any attempt to remove, alter, or strip the developer attribution will trigger an immediate graceful security shutdown.

---

## 📜 License & Disclaimer

### License
Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

### Disclaimer
*This software is designed for analytical and educational purposes only. Cryptocurrency trading and financial market speculation involve significant financial risk. No analytical tool or machine learning model can guarantee future market behavior. Always manage your risk responsibly and never trade with capital you cannot afford to lose.*

---

<p align="center">
  <b>Built with ❤️ and dedication by <a href="https://github.com/Arafat1slam">Arafat Islam</a></b>
</p>
