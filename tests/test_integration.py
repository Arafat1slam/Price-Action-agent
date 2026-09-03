"""
tests/test_integration.py - Full System Integration & Cockpit Rendering Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from price_action_engine import PriceActionEngine, AnalysisResult
from engines.confluence_engine import ConfluenceEngine
from engines.trade_setup_engine import TradeSetupEngine
from engines.smc_engine import SMCEngine
from engines.chart_pattern_engine import ChartPatternEngine
from engines.ml_predictor import MLPredictor
from core.models import BiasType, SetupType, TradeSetup

from main import (
    create_header_panel,
    create_mtf_panel,
    create_smc_panel,
    create_srp_panel,
    create_trade_setup_card,
    create_logs_panel,
    render_cockpit,
    render_text_cockpit,
    make_progress_bar,
    get_bias_color,
)


def create_mock_history(n: int = 50, symbol: str = "BTCUSDT") -> pd.DataFrame:
    dates = pd.date_range("2026-09-01", periods=n, freq="1h")
    p = 60000.0
    rows = []
    for i in range(n):
        p += 50.0 if i % 2 == 0 else -20.0
        rows.append([p, p + 100.0, p - 80.0, p + 20.0, 2000.0 + i * 50])
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df["timestamp"] = dates
    df["is_closed"] = True
    df["symbol"] = symbol
    return df


def test_full_price_action_engine_integration():
    """Verifies that PriceActionEngine integrates all 5 sub-engines properly."""
    engine = PriceActionEngine(max_candles=100)
    assert isinstance(engine.confluence_engine, ConfluenceEngine)
    assert isinstance(engine.trade_setup_engine, TradeSetupEngine)
    assert isinstance(engine.smc_engine, SMCEngine)
    assert isinstance(engine.chart_pattern_engine, ChartPatternEngine)
    assert isinstance(engine.ml_predictor, MLPredictor)

    df = create_mock_history(n=60, symbol="BTCUSDT")
    engine.set_history(df)

    res = engine.analyze(symbol="BTCUSDT", timeframe="1h")
    assert res is not None
    assert isinstance(res, AnalysisResult)

    # 1. Confluence Report
    assert res.confluence_report is not None
    assert res.confluence_report.symbol == "BTCUSDT"
    assert 50 <= res.confluence_report.confidence_score <= 95
    assert res.confluence_report.overall_bias in [
        BiasType.STRONG_BULLISH, BiasType.BULLISH, BiasType.NEUTRAL,
        BiasType.BEARISH, BiasType.STRONG_BEARISH
    ]

    # 2. SMC Report
    assert res.smc_report is not None

    # 3. Chart Patterns
    assert isinstance(res.chart_patterns, list)

    # 4. ML Result
    assert res.ml_result is not None
    assert 0.0 <= res.ml_result.prob_bullish <= 1.0
    assert 0.0 <= res.ml_result.prob_bearish <= 1.0

    # 5. Trade Setup (if generated, must strictly enforce >= 1.80 R:R)
    if res.trade_setup is not None:
        assert isinstance(res.trade_setup, TradeSetup)
        assert res.trade_setup.effective_rr >= 1.80
        assert res.trade_setup.direction in ["LONG", "SHORT"]
        assert res.trade_setup.entry_price > 0
        assert res.trade_setup.stop_loss > 0


def test_rich_cockpit_rendering_components():
    """Tests that all Rich dashboard widget panels render without exceptions."""
    engine = PriceActionEngine(max_candles=60)
    df = create_mock_history(n=50, symbol="ETHUSDT")
    engine.set_history(df)

    res = engine.analyze(symbol="ETHUSDT", timeframe="15m")
    assert res is not None

    logs = [
        "[05:00:00] Initialized scanner",
        "[05:01:00] Bullish FVG detected",
        "[05:02:00] Trade setup evaluated"
    ]

    # Test each panel function
    p_header = create_header_panel(res)
    assert p_header is not None

    p_mtf = create_mtf_panel(res)
    assert p_mtf is not None

    p_smc = create_smc_panel(res)
    assert p_smc is not None

    p_srp = create_srp_panel(res)
    assert p_srp is not None

    p_setup = create_trade_setup_card(res)
    assert p_setup is not None

    p_logs = create_logs_panel(logs)
    assert p_logs is not None

    cockpit_layout = render_cockpit(res, logs)
    assert cockpit_layout is not None

    # Test fallback text rendering
    render_text_cockpit(res, logs)


def test_progress_bar_and_color_helpers():
    assert make_progress_bar(0, width=10) == "----------"
    assert make_progress_bar(100, width=10) == "##########"
    assert make_progress_bar(50, width=10) == "#####-----"

    assert get_bias_color(BiasType.STRONG_BULLISH) == "bold green"
    assert get_bias_color(BiasType.BULLISH) == "green"
    assert get_bias_color(BiasType.STRONG_BEARISH) == "bold red"
    assert get_bias_color(BiasType.BEARISH) == "red"
    assert get_bias_color(BiasType.NEUTRAL) == "yellow"
