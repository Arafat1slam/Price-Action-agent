import pytest
import pandas as pd
import numpy as np
from price_action_engine import PriceActionEngine

def create_candle_df(candles):
    """Helper to convert list of [open, high, low, close, volume] to DataFrame."""
    df = pd.DataFrame(candles, columns=["open", "high", "low", "close", "volume"], dtype=float)
    df["timestamp"] = pd.date_range(start="2026-01-01", periods=len(df), freq="1h")
    df["is_closed"] = True
    return df

def test_engine_initialization():
    engine = PriceActionEngine(max_candles=50)
    assert engine.max_candles == 50
    assert engine.df.empty

def test_bullish_engulfing_detection():
    engine = PriceActionEngine(max_candles=50)
    # 5 neutral candles followed by red candle then engulfing green candle
    candles = [
        [100, 102, 99, 101, 1000],
        [101, 103, 100, 102, 1000],
        [102, 104, 101, 103, 1000],
        [103, 105, 102, 104, 1000],
        [104, 106, 103, 105, 1000],
        # Red candle: open 105, close 100
        [105, 105.5, 99.5, 100, 1200],
        # Bullish Engulfing: open 99, close 107 (engulfs 105 to 100)
        [99, 107.5, 98.5, 107, 2500]
    ]
    df = create_candle_df(candles)
    engine.set_history(df)

    res = engine.analyze(symbol="BTCUSDT", timeframe="1h")
    assert res is not None
    assert any("Bullish Engulfing" in p for p in res.patterns)
    assert res.bias == "BULLISH"
    assert res.confidence >= 55

def test_bearish_engulfing_detection():
    engine = PriceActionEngine(max_candles=50)
    candles = [
        [100, 102, 99, 101, 1000],
        [101, 103, 100, 102, 1000],
        [102, 104, 101, 103, 1000],
        [103, 105, 102, 104, 1000],
        [104, 106, 103, 105, 1000],
        # Green candle: open 100, close 105
        [100, 105.5, 99.5, 105, 1200],
        # Bearish Engulfing: open 106, close 98
        [106, 106.5, 97.5, 98, 2500]
    ]
    df = create_candle_df(candles)
    engine.set_history(df)

    res = engine.analyze(symbol="BTCUSDT", timeframe="1h")
    assert res is not None
    assert any("Bearish Engulfing" in p for p in res.patterns)
    assert res.bias == "BEARISH"

def test_doji_detection():
    engine = PriceActionEngine(max_candles=50)
    candles = [
        [100, 102, 99, 101, 1000],
        [101, 103, 100, 102, 1000],
        [102, 104, 101, 103, 1000],
        [103, 105, 102, 104, 1000],
        [104, 106, 103, 105, 1000],
        # Doji: open 100.0, close 100.1, high 105, low 95
        [100.0, 105.0, 95.0, 100.1, 800]
    ]
    df = create_candle_df(candles)
    engine.set_history(df)

    res = engine.analyze(symbol="BTCUSDT", timeframe="1h")
    assert res is not None
    assert any("Doji" in p for p in res.patterns)

def test_pin_bar_detection():
    engine = PriceActionEngine(max_candles=50)
    candles = [
        [100, 102, 99, 101, 1000],
        [101, 103, 100, 102, 1000],
        [102, 104, 101, 103, 1000],
        [103, 105, 102, 104, 1000],
        [104, 106, 103, 105, 1000],
        # Bullish Pin bar: High 100, Open 98, Close 99, Low 90 (lower wick = 8, range = 10, wick is 80%)
        [98.0, 100.0, 90.0, 99.0, 1500]
    ]
    df = create_candle_df(candles)
    engine.set_history(df)

    res = engine.analyze(symbol="ETHUSDT", timeframe="15m")
    assert res is not None
    assert any("Bullish Pin Bar" in p for p in res.patterns)

def test_uptrend_detection():
    engine = PriceActionEngine(max_candles=100)
    # Generate an explicit higher highs & higher lows structure
    candles = []
    price = 100.0
    for i in range(30):
        # Stepping upwards
        price += 2.0
        candles.append([price, price + 3.0, price - 1.0, price + 2.0, 1000 + i * 10])

    df = create_candle_df(candles)
    engine.set_history(df)

    res = engine.analyze(symbol="BTCUSDT", timeframe="1h")
    assert res is not None
    assert "Uptrend" in res.trend_detail

def test_live_candle_update():
    engine = PriceActionEngine(max_candles=50)
    candles = [
        [100, 102, 99, 101, 1000],
        [101, 103, 100, 102, 1000],
        [102, 104, 101, 103, 1000],
        [103, 105, 102, 104, 1000],
        [104, 106, 103, 105, 1000]
    ]
    df = create_candle_df(candles)
    engine.set_history(df)

    # Send a live forming update with same timestamp as last candle
    last_ts = engine.df.iloc[-1]["timestamp"]
    live_update = {
        "timestamp": last_ts,
        "open": 104.0,
        "high": 110.0,
        "low": 103.0,
        "close": 109.5,
        "volume": 3500.0,
        "is_closed": False
    }
    engine.update_candle(live_update)
    assert len(engine.df) == 5
    assert engine.df.iloc[-1]["close"] == 109.5
    assert engine.df.iloc[-1]["is_closed"] == False

    res = engine.analyze(symbol="SOLUSDT", timeframe="1h")
    assert res is not None
    assert not res.is_candle_closed
