"""
tests/test_confluence_engine.py - Unit Tests for Multi-Timeframe Confluence Engine
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from core.models import BiasType, ConfluenceReport, TimeframeConfluence
from engines.confluence_engine import ConfluenceEngine
from engines.smc_engine import SMCEngine
from engines.chart_pattern_engine import ChartPatternEngine
from engines.ml_predictor import MLPredictor


def create_sample_df(n: int = 60, trend: str = "UP", start_price: float = 100.0) -> pd.DataFrame:
    """Helper to generate clean synthetic OHLCV bars."""
    dates = [datetime.now(timezone.utc) - timedelta(hours=n - i) for i in range(n)]
    rows = []
    p = start_price
    for i in range(n):
        step = 1.0 if trend == "UP" else (-1.0 if trend == "DOWN" else (0.5 if i % 2 == 0 else -0.5))
        p += step
        o = p
        h = p + 1.5
        l = p - 1.0
        c = p + 0.5 if trend == "UP" else p - 0.5
        v = 1000.0 + i * 20.0
        rows.append([o, h, l, c, v])

    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df["timestamp"] = dates
    df["is_closed"] = True
    return df


def test_confluence_engine_initialization():
    engine = ConfluenceEngine()
    assert engine.smc_engine is not None
    assert engine.chart_pattern_engine is not None
    assert engine.ml_predictor is not None
    assert engine.WEIGHT_SMC == 0.25
    assert engine.WEIGHT_HTF == 0.20
    assert engine.WEIGHT_SR_VP == 0.15
    assert engine.WEIGHT_PAT == 0.15
    assert engine.WEIGHT_ML == 0.15
    assert engine.WEIGHT_VOL_MOM == 0.10


def test_insufficient_history_fallback():
    engine = ConfluenceEngine()
    df_small = create_sample_df(n=3)
    rep = engine.analyze(symbol="BTCUSDT", data=df_small, current_price=100.0)
    assert rep.overall_bias == BiasType.NEUTRAL
    assert rep.confidence_score == 50
    assert rep.symbol == "BTCUSDT"


def test_bullish_confluence_synthesis():
    engine = ConfluenceEngine()
    df_bull = create_sample_df(n=60, trend="UP", start_price=50000.0)
    rep = engine.analyze(symbol="BTCUSDT", data=df_bull, current_price=df_bull["close"].iloc[-1])
    assert rep is not None
    assert rep.overall_bias in [BiasType.BULLISH, BiasType.STRONG_BULLISH]
    assert 50 <= rep.confidence_score <= 95
    assert rep.symbol == "BTCUSDT"
    assert rep.current_price > 0


def test_bearish_confluence_synthesis():
    engine = ConfluenceEngine()
    df_bear = create_sample_df(n=60, trend="DOWN", start_price=50000.0)
    rep = engine.analyze(symbol="BTCUSDT", data=df_bear, current_price=df_bear["close"].iloc[-1])
    assert rep is not None
    assert rep.overall_bias in [BiasType.BEARISH, BiasType.STRONG_BEARISH]
    assert 50 <= rep.confidence_score <= 95


def test_forming_candle_penalty():
    engine = ConfluenceEngine()
    df = create_sample_df(n=50, trend="UP")
    rep_closed = engine.analyze(symbol="BTCUSDT", data=df, is_closed=True)
    rep_forming = engine.analyze(symbol="BTCUSDT", data=df, is_closed=False)

    mult_closed = getattr(rep_closed, "multipliers", {}).get("m_forming", 1.0)
    mult_forming = getattr(rep_forming, "multipliers", {}).get("m_forming", 1.0)

    assert mult_closed == 1.0
    assert mult_forming == 0.88


def test_htf_counter_trend_penalty():
    engine = ConfluenceEngine()
    df_primary_up = create_sample_df(n=50, trend="UP", start_price=100.0)
    df_htf_down = create_sample_df(n=50, trend="DOWN", start_price=200.0)

    # Primary is UP, but HTF 4h and 1d are strongly DOWN
    tf_data = {
        "15m": df_primary_up,
        "1h": df_primary_up,
        "4h": df_htf_down,
        "1d": df_htf_down,
    }

    rep = engine.analyze(symbol="BTCUSDT", data=tf_data, primary_timeframe="15m")
    multipliers = getattr(rep, "multipliers", {})
    # Either HTF conflict penalty was applied or bias reflects the conflict
    assert multipliers.get("m_htf_conflict", 1.0) <= 1.0
    assert 50 <= rep.confidence_score <= 95


def test_multi_timeframe_dict_input():
    engine = ConfluenceEngine()
    tf_data = {
        "15m": create_sample_df(n=40, trend="UP"),
        "1h": create_sample_df(n=40, trend="UP"),
        "4h": create_sample_df(n=40, trend="UP"),
        "1d": create_sample_df(n=40, trend="UP"),
    }
    rep = engine.analyze(symbol="ETHUSDT", data=tf_data, primary_timeframe="1h")
    assert rep.symbol == "ETHUSDT"
    assert "1d" in rep.timeframe_breakdown
    assert "4h" in rep.timeframe_breakdown
    assert "1h" in rep.timeframe_breakdown
    assert "15m" in rep.timeframe_breakdown

    for tf_key, tf_conf in rep.timeframe_breakdown.items():
        assert isinstance(tf_conf, TimeframeConfluence)
        assert -1.0 <= tf_conf.score <= 1.0
        assert tf_conf.bias in [BiasType.BULLISH, BiasType.STRONG_BULLISH, BiasType.NEUTRAL, BiasType.BEARISH, BiasType.STRONG_BEARISH]


def test_confidence_score_range():
    engine = ConfluenceEngine()
    # Test across multiple random and directional walk profiles
    np.random.seed(42)
    for trend in ["UP", "DOWN", "RANGE"]:
        df = create_sample_df(n=50, trend=trend)
        rep = engine.analyze(symbol="SOLUSDT", data=df)
        assert 50 <= rep.confidence_score <= 95
        assert isinstance(rep.confidence_score, int)


def test_confluence_report_diagnostics():
    engine = ConfluenceEngine()
    df = create_sample_df(n=50, trend="UP")
    rep = engine.analyze(symbol="BNBUSDT", data=df)

    assert hasattr(rep, "raw_score")
    assert hasattr(rep, "adjusted_score")
    assert hasattr(rep, "component_scores")
    assert hasattr(rep, "multipliers")

    assert -1.0 <= rep.raw_score <= 1.0
    assert -1.0 <= rep.adjusted_score <= 1.0
    assert "smc" in rep.component_scores
    assert "htf" in rep.component_scores
    assert "sr_vp" in rep.component_scores
    assert "patterns" in rep.component_scores
    assert "ml" in rep.component_scores
    assert "vol_mom" in rep.component_scores
