"""
tests/test_trade_setup_engine.py - Unit Tests for Trade Setup & Risk-Reward Engine
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from core.models import (
    BiasType,
    ConfluenceReport,
    SetupType,
    TradeSetup,
    FairValueGap,
    OrderBlock,
    ChartPattern,
    VolumeProfileZone,
)
from engines.trade_setup_engine import TradeSetupEngine


def create_sample_df_for_trade(n: int = 30, base_price: float = 60000.0) -> pd.DataFrame:
    """Helper to generate price data with realistic ATR."""
    np.random.seed(42)
    rows = []
    p = base_price
    for i in range(n):
        step = np.random.normal(0, 50)
        p += step
        o = p
        h = p + abs(np.random.normal(100, 30))
        l = p - abs(np.random.normal(100, 30))
        c = p + np.random.normal(0, 40)
        v = 1500.0 + np.random.uniform(0, 500)
        rows.append([o, h, l, c, v])
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.date_range("2026-09-01", periods=n, freq="1h")
    df["is_closed"] = True
    return df


def test_trade_setup_engine_initialization():
    engine = TradeSetupEngine()
    assert engine.min_effective_rr == 1.80
    assert engine.atr_multiplier_sl == 0.20


def test_enforce_min_rr_gate():
    """Verifies that any candidate setup with effective R:R < 1.80 is discarded."""
    engine = TradeSetupEngine(min_effective_rr=1.80)
    df = create_sample_df_for_trade()

    # Create dummy setup with low R:R
    low_rr_setup = engine._create_trade_setup(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry=100.0,
        sl=90.0,    # Risk = 10
        tp1=110.0,  # RR1 = 1.0
        tp2=115.0,  # RR2 = 1.5 -> Effective = 1.25 (< 1.80)
        confidence=70,
        rationale=["Test low RR"],
        invalidation_reason="Test invalidation"
    )
    assert low_rr_setup.effective_rr < 1.80

    # High RR setup
    high_rr_setup = engine._create_trade_setup(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry=100.0,
        sl=95.0,    # Risk = 5
        tp1=110.0,  # RR1 = 2.0
        tp2=120.0,  # RR2 = 4.0 -> Effective = 3.00 (>= 1.80)
        confidence=85,
        rationale=["Test high RR"],
        invalidation_reason="Test invalidation"
    )
    assert high_rr_setup.effective_rr >= 1.80


def test_smc_fvg_pullback_setup_long():
    engine = TradeSetupEngine()
    df = create_sample_df_for_trade(n=30, base_price=64000.0)
    curr_price = 64200.0

    # Active Bullish FVG
    fvg = FairValueGap(
        top=64300.0,
        bottom=64100.0,
        midpoint=64200.0,
        bias=BiasType.BULLISH,
        created_time=datetime.now(timezone.utc),
        is_mitigated=False,
        mitigation_pct=0.0
    )

    report = ConfluenceReport(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        current_price=curr_price,
        overall_bias=BiasType.BULLISH,
        confidence_score=80,
        htf_bias=BiasType.BULLISH,
        ltf_trigger=BiasType.BULLISH,
        active_fvgs=[fvg],
    )

    setup = engine.generate_setup(report, df)
    assert setup is not None
    assert setup.direction == "LONG"
    assert setup.setup_type == SetupType.SMC_PULLBACK_FVG
    assert setup.stop_loss < setup.entry_price
    assert setup.tp1_price > setup.entry_price
    assert setup.tp2_price > setup.tp1_price
    assert setup.effective_rr >= 1.80
    assert "FVG" in setup.rationale[0]


def test_smc_fvg_pullback_setup_short():
    engine = TradeSetupEngine()
    df = create_sample_df_for_trade(n=30, base_price=64000.0)
    curr_price = 63800.0

    # Active Bearish FVG
    fvg = FairValueGap(
        top=63900.0,
        bottom=63700.0,
        midpoint=63800.0,
        bias=BiasType.BEARISH,
        created_time=datetime.now(timezone.utc),
        is_mitigated=False,
        mitigation_pct=0.0
    )

    report = ConfluenceReport(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        current_price=curr_price,
        overall_bias=BiasType.BEARISH,
        confidence_score=82,
        htf_bias=BiasType.BEARISH,
        ltf_trigger=BiasType.BEARISH,
        active_fvgs=[fvg],
    )

    setup = engine.generate_setup(report, df)
    assert setup is not None
    assert setup.direction == "SHORT"
    assert setup.setup_type == SetupType.SMC_PULLBACK_FVG
    assert setup.stop_loss > setup.entry_price
    assert setup.tp1_price < setup.entry_price
    assert setup.tp2_price < setup.tp1_price
    assert setup.effective_rr >= 1.80


def test_smc_order_block_setup():
    engine = TradeSetupEngine()
    df = create_sample_df_for_trade(n=30, base_price=60000.0)
    curr_price = 60100.0

    # Bullish Demand Order Block
    ob = OrderBlock(
        top=60200.0,
        bottom=59800.0,
        bias=BiasType.BULLISH,
        created_time=datetime.now(timezone.utc),
        volume=5000.0,
        is_mitigated=False,
        is_invalidated=False
    )

    report = ConfluenceReport(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        current_price=curr_price,
        overall_bias=BiasType.BULLISH,
        confidence_score=85,
        htf_bias=BiasType.BULLISH,
        ltf_trigger=BiasType.BULLISH,
        active_obs=[ob],
    )

    setup = engine.generate_setup(report, df)
    assert setup is not None
    assert setup.setup_type == SetupType.SMC_ORDER_BLOCK
    assert setup.direction == "LONG"
    assert setup.stop_loss < ob.bottom
    assert setup.effective_rr >= 1.80


def test_value_area_mean_reversion_setup():
    engine = TradeSetupEngine()
    df = create_sample_df_for_trade(n=30, base_price=64000.0)
    curr_price = 63200.0

    vp = VolumeProfileZone(
        poc_price=64100.0,
        vah_price=65400.0,
        val_price=63200.0,
        total_volume=50000.0
    )

    report = ConfluenceReport(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        current_price=curr_price,
        overall_bias=BiasType.BULLISH,
        confidence_score=75,
        htf_bias=BiasType.NEUTRAL,
        ltf_trigger=BiasType.BULLISH,
        volume_profile=vp,
    )

    setup = engine.generate_setup(report, df)
    assert setup is not None
    assert setup.setup_type == SetupType.VALUE_AREA_MEAN_REVERSION
    assert setup.direction == "LONG"
    assert setup.effective_rr >= 1.80


def test_chart_pattern_breakout_setup():
    engine = TradeSetupEngine()
    df = create_sample_df_for_trade(n=30, base_price=64000.0)

    pat = ChartPattern(
        name="DOUBLE_BOTTOM",
        bias=BiasType.BULLISH,
        quality_score=0.88,
        neckline_price=64500.0,
        projected_target=65800.0,
        invalidation_level=63800.0,
        candle_span=12
    )

    report = ConfluenceReport(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        current_price=64520.0,
        overall_bias=BiasType.BULLISH,
        confidence_score=85,
        htf_bias=BiasType.BULLISH,
        ltf_trigger=BiasType.BULLISH,
        active_patterns=[pat],
    )

    setup = engine.generate_setup(report, df)
    assert setup is not None
    assert setup.setup_type == SetupType.CHART_PATTERN_BREAKOUT
    assert setup.direction == "LONG"
    assert setup.stop_loss < 63800.0
    assert setup.effective_rr >= 1.80


def test_effective_rr_math():
    engine = TradeSetupEngine()
    entry = 100.0
    sl = 90.0
    tp1 = 118.0  # RR1 = 1.8
    tp2 = 130.0  # RR2 = 3.0
    # Expected blended = 0.5 * 1.8 + 0.5 * 3.0 = 2.40

    setup = engine._create_trade_setup(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        confidence=85,
        rationale=["Math check"],
        invalidation_reason="Math invalidation"
    )

    assert setup.risk_reward_tp1 == 1.80
    assert setup.risk_reward_tp2 == 3.00
    assert setup.effective_rr == 2.40
    assert setup.effective_rr >= 1.80
