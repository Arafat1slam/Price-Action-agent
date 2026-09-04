"""
tests/test_mtf_engine.py
========================
Unit and integration tests for the synchronized 5m/15m/1h/4h Live Multi-Timeframe Engine.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from core.models import BiasType, ConfluenceReport, TimeframeConfluence
from engines.confluence_engine import ConfluenceEngine
from price_action_engine import PriceActionEngine, AnalysisResult
from binance_client import BinanceClient


def generate_synthetic_ohlcv(n: int, trend: str = "UP", interval_min: int = 15, base_p: float = 60000.0) -> pd.DataFrame:
    records = []
    curr = base_p
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    for i in range(n):
        drift = 50.0 if trend == "UP" else (-50.0 if trend == "DOWN" else 0.0)
        noise = np.random.normal(0, 15.0)
        o = curr
        c = o + drift + noise
        h = max(o, c) + abs(np.random.normal(20, 5))
        l = min(o, c) - abs(np.random.normal(20, 5))
        v = float(np.random.uniform(100, 500))
        t = t0 + timedelta(minutes=i * interval_min)
        records.append({
            "timestamp": t,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": round(v, 2),
            "is_closed": True,
        })
        curr = c
    return pd.DataFrame(records)


def test_confluence_engine_mtf_all_four_timeframes():
    """Verify ConfluenceEngine analyzes 5m, 15m, 1h, 4h concurrently."""
    engine = ConfluenceEngine()
    
    df_5m = generate_synthetic_ohlcv(60, trend="UP", interval_min=5, base_p=60000.0)
    df_15m = generate_synthetic_ohlcv(60, trend="UP", interval_min=15, base_p=60000.0)
    df_1h = generate_synthetic_ohlcv(60, trend="UP", interval_min=60, base_p=60000.0)
    df_4h = generate_synthetic_ohlcv(60, trend="UP", interval_min=240, base_p=60000.0)

    mtf_data = {
        "5m": df_5m,
        "15m": df_15m,
        "1h": df_1h,
        "4h": df_4h,
    }

    report = engine.analyze(
        symbol="BTCUSDT",
        data=mtf_data,
        current_price=float(df_1h["close"].iloc[-1]),
        primary_timeframe="1h",
    )

    assert isinstance(report, ConfluenceReport)
    assert report.confidence_score >= 50
    # Verify all 4 required timeframes are present in breakdown
    for tf in ["5m", "15m", "1h", "4h"]:
        assert tf in report.timeframe_breakdown, f"Missing timeframe {tf} in report!"
        tf_conf = report.timeframe_breakdown[tf]
        assert isinstance(tf_conf, TimeframeConfluence)
        assert tf_conf.timeframe == tf
        assert isinstance(tf_conf.bias, BiasType)
        assert -1.0 <= tf_conf.score <= 1.0


def test_price_action_engine_set_multi_history_and_analyze():
    """Verify PriceActionEngine stores multi-timeframe buffers and analyzes correctly."""
    pa_engine = PriceActionEngine(max_candles=80)

    df_5m = generate_synthetic_ohlcv(50, trend="UP", interval_min=5, base_p=3000.0)
    df_15m = generate_synthetic_ohlcv(50, trend="UP", interval_min=15, base_p=3000.0)
    df_1h = generate_synthetic_ohlcv(50, trend="UP", interval_min=60, base_p=3000.0)
    df_4h = generate_synthetic_ohlcv(50, trend="UP", interval_min=240, base_p=3000.0)

    mtf_dfs = {
        "5m": df_5m,
        "15m": df_15m,
        "1h": df_1h,
        "4h": df_4h,
    }

    pa_engine.set_multi_history(mtf_dfs, primary_tf="1h")

    assert "5m" in pa_engine.mtf_buffers
    assert "15m" in pa_engine.mtf_buffers
    assert "1h" in pa_engine.mtf_buffers
    assert "4h" in pa_engine.mtf_buffers
    assert len(pa_engine.df) == 50

    res = pa_engine.analyze(symbol="ETHUSDT", timeframe="1h")
    assert res is not None
    assert isinstance(res, AnalysisResult)
    assert res.confluence_report is not None
    assert "5m" in res.confluence_report.timeframe_breakdown
    assert "15m" in res.confluence_report.timeframe_breakdown
    assert "1h" in res.confluence_report.timeframe_breakdown
    assert "4h" in res.confluence_report.timeframe_breakdown


def test_multi_candle_update_routing():
    """Verify incoming streaming candles are routed to their respective timeframe buffer."""
    pa_engine = PriceActionEngine(max_candles=50)
    df_1h = generate_synthetic_ohlcv(30, trend="UP", interval_min=60, base_p=100.0)
    df_5m = generate_synthetic_ohlcv(30, trend="UP", interval_min=5, base_p=100.0)

    pa_engine.set_multi_history({"1h": df_1h, "5m": df_5m}, primary_tf="1h")

    # Send 5m candle update
    new_5m = {
        "timestamp": datetime.now(timezone.utc),
        "open": 105.0,
        "high": 106.0,
        "low": 104.5,
        "close": 105.8,
        "volume": 250.0,
        "is_closed": False,
        "interval": "5m",
        "symbol": "SOLUSDT",
    }
    pa_engine.update_candle(new_5m)
    assert len(pa_engine.mtf_buffers["5m"]) == 31
    assert pa_engine.mtf_buffers["5m"].iloc[-1]["close"] == 105.8
