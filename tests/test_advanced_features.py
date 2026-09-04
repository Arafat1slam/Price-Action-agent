"""
tests/test_advanced_features.py - Rigorous Verification Suite for Institutional Trading Features
Covers:
  1. SMC OTE Zone (Fib 0.618 - 0.786) & Dealing Range (Premium vs. Discount)
  2. TTM Squeeze Volatility Compression Engine
  3. Choppiness Index (CHOP) Engine
  4. Quasimodo & Wyckoff Spring Structural Reversal Patterns
  5. Market Regime Engine with CHOP & Squeeze Integration
  6. ML Predictor Temperature Calibration
  7. Confluence Engine ICT Session Killzone Detection
  8. Trade Setup 3-Tier TP Structure & Trailing Stop
  9. Position State Manager Multi-Stage Scale-Out (TP1, TP2, TP3)
  10. Backtest Engine Calmar Ratio & Monte Carlo Resampling Simulation
  11. Binance Client Input Sanitization & Health Heartbeat
"""

import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from core.models import BiasType, SetupType, PositionState, TradeSetup
from engines.smc_engine import SMCEngine, OTEZone, DealingRangeState
from engines.indicators_engine import TTMSqueezeEngine, ChoppinessIndexEngine
from engines.chart_pattern_engine import ChartPatternEngine
from engines.regime_engine import RegimeEngine, MarketRegime
from engines.ml_predictor import MLPredictor
from engines.confluence_engine import ConfluenceEngine
from engines.trade_setup_engine import TradeSetupEngine, PositionStateManager
from engines.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    BacktestMetrics,
    TradeRecord,
    MonteCarloResult,
)
from binance_client import BinanceClient


def _make_dummy_ohlcv(n: int = 100, trend: str = "up") -> pd.DataFrame:
    """Helper to generate consistent OHLCV synthetic data for deterministic testing."""
    base = 100.0
    records = []
    for i in range(n):
        if trend == "up":
            step = 0.5 + 0.1 * math.sin(i / 5.0)
        elif trend == "down":
            step = -0.5 - 0.1 * math.sin(i / 5.0)
        else:
            step = 0.2 * math.sin(i / 3.0)

        open_p = base
        close_p = base + step
        high_p = max(open_p, close_p) + 0.3
        low_p = min(open_p, close_p) - 0.3
        volume = 1000.0 + 100.0 * (i % 5)

        records.append({
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
            "is_closed": True,
        })
        base = close_p

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# 1. SMC OTE Zone & Dealing Range Tests
# ----------------------------------------------------------------------------
def test_smc_ote_zone_calculation():
    smc = SMCEngine()
    # Bullish swing: low 100, high 200
    ote_bull = smc.compute_ote_zone(swing_low=100.0, swing_high=200.0, is_bullish=True)
    assert ote_bull is not None
    assert ote_bull.is_bullish is True
    assert math.isclose(ote_bull.fib_0618, 138.2, abs_tol=0.01)
    assert math.isclose(ote_bull.fib_0786, 121.4, abs_tol=0.01)
    assert ote_bull.ote_high > ote_bull.ote_low

    # Bearish swing: high 200, low 100
    ote_bear = smc.compute_ote_zone(swing_low=100.0, swing_high=200.0, is_bullish=False)
    assert ote_bear is not None
    assert ote_bear.is_bullish is False
    assert math.isclose(ote_bear.fib_0618, 161.8, abs_tol=0.01)
    assert math.isclose(ote_bear.fib_0786, 178.6, abs_tol=0.01)


def test_smc_dealing_range():
    smc = SMCEngine()
    dr = smc.compute_dealing_range(swing_low=100.0, swing_high=200.0, current_price=120.0)
    assert dr.equilibrium == 150.0
    assert dr.is_discount is True
    assert dr.is_premium is False
    assert dr.relative_percent == 20.0

    dr_prem = smc.compute_dealing_range(swing_low=100.0, swing_high=200.0, current_price=180.0)
    assert dr_prem.is_premium is True
    assert dr_prem.is_discount is False
    assert dr_prem.relative_percent == 80.0


# ----------------------------------------------------------------------------
# 2. TTM Squeeze Engine Tests
# ----------------------------------------------------------------------------
def test_ttm_squeeze_engine():
    squeeze_eng = TTMSqueezeEngine(bb_length=20, kc_length=20)
    # Low volatility compression data
    n = 40
    data = {
        "open": [100.0] * n,
        "high": [100.1] * n,
        "low": [99.9] * n,
        "close": [100.0 + 0.001 * i for i in range(n)],
        "volume": [500.0] * n,
    }
    df = pd.DataFrame(data)
    res = squeeze_eng.compute(df)
    assert res is not None
    assert isinstance(res.is_squeeze_on, bool)
    assert isinstance(res.momentum_direction, str)
    assert res.momentum_direction in ["BULLISH", "BEARISH", "NEUTRAL"]


# ----------------------------------------------------------------------------
# 3. Choppiness Index Engine Tests
# ----------------------------------------------------------------------------
def test_choppiness_index_engine():
    chop_eng = ChoppinessIndexEngine(period=14)
    # Highly trending data
    df_trend = _make_dummy_ohlcv(60, trend="up")
    res_trend = chop_eng.compute(df_trend)
    assert res_trend is not None
    assert 0.0 <= res_trend.chop_index <= 100.0

    # Highly oscillating choppy data
    chop_records = []
    base = 100.0
    for i in range(60):
        step = 1.0 if (i % 2 == 0) else -1.0
        chop_records.append({
            "open": base,
            "high": base + 1.5,
            "low": base - 1.5,
            "close": base + step,
            "volume": 1000.0,
        })
    df_chop = pd.DataFrame(chop_records)
    res_chop = chop_eng.compute(df_chop)
    assert res_chop is not None
    assert res_chop.chop_index > 38.2


# ----------------------------------------------------------------------------
# 4. Quasimodo & Wyckoff Spring Pattern Tests
# ----------------------------------------------------------------------------
def test_quasimodo_and_wyckoff_detection():
    pat_eng = ChartPatternEngine()
    df = _make_dummy_ohlcv(80, trend="up")

    qm_patterns = pat_eng.detect_quasimodo(df)
    assert isinstance(qm_patterns, list)

    spring_patterns = pat_eng.detect_wyckoff_spring(df)
    assert isinstance(spring_patterns, list)

    all_patterns = pat_eng.detect_all(df)
    assert isinstance(all_patterns, list)


# ----------------------------------------------------------------------------
# 5. Market Regime Engine with CHOP & Squeeze Tests
# ----------------------------------------------------------------------------
def test_regime_engine_integration():
    regime_eng = RegimeEngine()
    df = _make_dummy_ohlcv(80, trend="up")
    report = regime_eng.analyze(df)
    assert report is not None
    assert report.regime in MarketRegime
    assert 0.0 <= report.confidence <= 100.0
    assert len(report.key_drivers) > 0


# ----------------------------------------------------------------------------
# 6. ML Predictor Temperature Scaling Tests
# ----------------------------------------------------------------------------
def test_ml_predictor_temperature_scaling():
    pred_low_temp = MLPredictor(temperature=0.5)
    pred_high_temp = MLPredictor(temperature=2.0)
    df = _make_dummy_ohlcv(60, trend="up")

    res_low = pred_low_temp.predict(df)
    res_high = pred_high_temp.predict(df)

    assert 0.0 <= res_low.prob_bullish <= 1.0
    assert 0.0 <= res_high.prob_bullish <= 1.0
    diff_from_uniform_low = abs(res_low.prob_bullish - 0.333)
    diff_from_uniform_high = abs(res_high.prob_bullish - 0.333)
    assert diff_from_uniform_high <= diff_from_uniform_low + 0.05


# ----------------------------------------------------------------------------
# 7. Confluence Engine Session Killzones Tests
# ----------------------------------------------------------------------------
def test_confluence_session_killzones():
    dt_london = datetime(2026, 3, 10, 8, 30, 0, tzinfo=timezone.utc)
    in_kz, kz_name, boost = ConfluenceEngine.detect_session_killzone(dt_london)
    assert in_kz is True
    assert kz_name == "LONDON_OPEN"
    assert boost == 1.06

    dt_ny = datetime(2026, 3, 10, 14, 15, 0, tzinfo=timezone.utc)
    in_kz, kz_name, boost = ConfluenceEngine.detect_session_killzone(dt_ny)
    assert in_kz is True
    assert kz_name == "NEW_YORK_OPEN"
    assert boost == 1.06

    dt_asia = datetime(2026, 3, 10, 3, 0, 0, tzinfo=timezone.utc)
    in_kz, kz_name, boost = ConfluenceEngine.detect_session_killzone(dt_asia)
    assert in_kz is False
    assert kz_name == "ASIAN_RANGE"
    assert boost == 1.0


# ----------------------------------------------------------------------------
# 8. Trade Setup Engine 3-Tier TP & Trailing Stop Tests
# ----------------------------------------------------------------------------
def test_trade_setup_3_tier_tp_and_trailing():
    eng = TradeSetupEngine()
    setup = eng._create_trade_setup(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry=50000.0,
        sl=49000.0,
        tp1=52000.0,
        tp2=54000.0,
        confidence=80,
        rationale=["Test 3-Tier TP"],
        invalidation_reason="Below SL",
        current_price=50000.0,
    )

    assert setup.tp3_price >= setup.tp2_price
    assert setup.risk_reward_tp3 > 0
    assert setup.recommended_risk_pct <= 3.0

    atr = 500.0
    ts1 = TradeSetupEngine.compute_trailing_stop(
        direction="LONG",
        entry_price=50000.0,
        current_sl=49000.0,
        current_price=50500.0,
        atr=atr,
        tp1_price=52000.0,
        tp2_price=54000.0,
        extreme_price_reached=50500.0,
    )
    assert ts1 == 49000.0

    ts2 = TradeSetupEngine.compute_trailing_stop(
        direction="LONG",
        entry_price=50000.0,
        current_sl=49000.0,
        current_price=52200.0,
        atr=atr,
        tp1_price=52000.0,
        tp2_price=54000.0,
        extreme_price_reached=52200.0,
    )
    assert ts2 >= 50000.0


# ----------------------------------------------------------------------------
# 9. Position State Manager Multi-Stage Transitions
# ----------------------------------------------------------------------------
def test_position_state_manager_3_tier_lifecycle():
    pm = PositionStateManager()
    setup = TradeSetup(
        setup_id="TEST-001",
        symbol="ETHUSDT",
        timestamp=datetime.now(timezone.utc),
        setup_type=SetupType.SMC_ORDER_BLOCK,
        direction="LONG",
        entry_price=3000.0,
        stop_loss=2900.0,
        tp1_price=3150.0,
        tp2_price=3300.0,
        risk_reward_tp1=1.5,
        risk_reward_tp2=3.0,
        effective_rr=2.25,
        confidence_score=75,
        tp3_price=3500.0,
        risk_reward_tp3=5.0,
    )

    pos = pm.open_position(setup, current_price=3000.0)
    assert pos.state == PositionState.OPEN_LONG

    pm.update_position(current_price=3160.0)
    assert pos.state == PositionState.PARTIAL_TP1
    assert pos.trailing_stop >= 3000.0

    pm.update_position(current_price=3310.0)
    assert pos.state == PositionState.PARTIAL_TP2
    assert pos.trailing_stop >= 3150.0

    closed = pm.update_position(current_price=3510.0)
    assert closed is not None
    assert closed.state == PositionState.FLAT
    assert pm.is_flat is True


# ----------------------------------------------------------------------------
# 10. Backtest Engine Calmar Ratio & Monte Carlo Simulation
# ----------------------------------------------------------------------------
def test_backtest_engine_monte_carlo():
    engine = BacktestEngine(config=BacktestConfig(initial_capital=10000.0))
    df = _make_dummy_ohlcv(100, trend="up")
    res = engine.run(df, symbol="BTCUSDT", timeframe="1h")

    assert res is not None
    assert hasattr(res.metrics, "calmar_ratio")
    assert isinstance(res.metrics.calmar_ratio, float)

    mc_res = engine.run_monte_carlo(res, n_simulations=100, seed=42)
    assert isinstance(mc_res, MonteCarloResult)
    assert mc_res.n_simulations == 100
    assert 0.0 <= mc_res.risk_of_ruin_pct <= 100.0
    assert res.metrics.monte_carlo is not None


# ----------------------------------------------------------------------------
# 11. Binance Client Input Sanitization & Health Check
# ----------------------------------------------------------------------------
def test_binance_client_sanitization_and_health():
    client = BinanceClient("btc/usdt", "1H")
    assert client.symbol == "BTCUSDT"
    assert client.interval == "1h"
    assert client.is_healthy() is False

    client.is_running = True
    assert client.is_healthy(max_stale_seconds=60.0) is True
    client.stop()
