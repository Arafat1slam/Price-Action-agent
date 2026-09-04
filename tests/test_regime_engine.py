"""
tests/test_regime_engine.py
===========================
Tests for MarketRegimeEngine: Trend / Range / Breakout / High-Volatility Regime Detection.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from core.models import MarketRegime, RegimeReport
from engines.regime_engine import MarketRegimeEngine
from price_action_engine import PriceActionEngine, AnalysisResult


def make_trending_bull_data(n: int = 60, start_price: float = 50000.0) -> pd.DataFrame:
    records = []
    p = start_price
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        drift = 80.0
        noise = np.random.normal(0, 5)
        o = p
        c = o + drift + noise
        h = max(o, c) + 20
        l = min(o, c) - 10
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 500.0, "is_closed": True
        })
        p = c
    return pd.DataFrame(records)


def make_trending_bear_data(n: int = 60, start_price: float = 50000.0) -> pd.DataFrame:
    records = []
    p = start_price
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        drift = -80.0
        noise = np.random.normal(0, 5)
        o = p
        c = o + drift + noise
        h = max(o, c) + 10
        l = min(o, c) - 20
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 500.0, "is_closed": True
        })
        p = c
    return pd.DataFrame(records)


def make_ranging_data(n: int = 60, base_price: float = 50000.0) -> pd.DataFrame:
    records = []
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        # Oscillates tightly around base_price
        c = base_price + 30.0 * np.sin(i * 0.4)
        o = base_price + 30.0 * np.sin((i - 0.5) * 0.4)
        h = max(o, c) + 15.0
        l = min(o, c) - 15.0
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 200.0, "is_closed": True
        })
    return pd.DataFrame(records)


def make_breakout_data(n: int = 60, base_price: float = 50000.0) -> pd.DataFrame:
    records = []
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # First 50 bars tight range
    for i in range(50):
        c = base_price + 10.0 * np.sin(i * 0.3)
        o = c - 2.0
        h = max(o, c) + 5.0
        l = min(o, c) - 5.0
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 100.0, "is_closed": True
        })
    # Last 10 bars explosive breakout
    p = base_price
    for i in range(50, n):
        o = p
        c = o + 300.0  # Huge jump
        h = c + 50.0
        l = o - 10.0
        v = 1500.0     # 15x volume surge
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": v, "is_closed": True
        })
        p = c
    return pd.DataFrame(records)


def make_chop_data(n: int = 60, base_price: float = 50000.0) -> pd.DataFrame:
    records = []
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # First 30 bars normal baseline volatility (range ~50)
    for i in range(30):
        c = base_price + 20.0 * np.sin(i * 0.3)
        o = c - 5.0
        h = max(o, c) + 10.0
        l = min(o, c) - 10.0
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 200.0, "is_closed": True
        })
    # Last 30 bars violent alternate swings with huge wicks (chop expansion)
    for i in range(30, n):
        sign = 1 if i % 2 == 0 else -1
        o = base_price
        c = base_price + sign * 150.0
        h = max(o, c) + 200.0
        l = min(o, c) - 200.0
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 400.0, "is_closed": True
        })
    return pd.DataFrame(records)


def test_trending_bull_regime_detection():
    engine = MarketRegimeEngine()
    df = make_trending_bull_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime == MarketRegime.TRENDING_BULL
    assert rep.plus_di > rep.minus_di
    assert rep.adx >= 20.0
    assert "Trend Following" in rep.recommended_strategy


def test_trending_bear_regime_detection():
    engine = MarketRegimeEngine()
    df = make_trending_bear_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime == MarketRegime.TRENDING_BEAR
    assert rep.minus_di > rep.plus_di
    assert rep.adx >= 20.0
    assert "Trend Following" in rep.recommended_strategy


def test_ranging_regime_detection():
    engine = MarketRegimeEngine()
    df = make_ranging_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime == MarketRegime.RANGING
    assert rep.adx < 22.0
    assert "Mean Reversion" in rep.recommended_strategy


def test_breakout_regime_detection():
    engine = MarketRegimeEngine()
    df = make_breakout_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime == MarketRegime.BREAKOUT
    assert rep.atr_ratio >= 1.2
    assert "Breakout" in rep.recommended_strategy


def test_chop_regime_detection():
    engine = MarketRegimeEngine()
    df = make_chop_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime in [MarketRegime.HIGH_VOLATILITY_CHOP, MarketRegime.RANGING]
    assert rep.atr_ratio > 1.0


def test_price_action_engine_integrates_regime():
    """Verify PriceActionEngine produces valid regime_report in AnalysisResult."""
    pa = PriceActionEngine(max_candles=70)
    df = make_trending_bull_data(60)
    pa.set_history(df)

    res = pa.analyze("BTCUSDT", "1h")
    assert res is not None
    assert res.regime_report is not None
    assert isinstance(res.regime_report, RegimeReport)
    assert res.regime_report.regime == MarketRegime.TRENDING_BULL
    assert res.regime_report.confidence >= 50
