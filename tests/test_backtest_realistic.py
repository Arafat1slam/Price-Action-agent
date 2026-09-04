"""
tests/test_backtest_realistic.py
================================
Unit tests for institutional realistic backtesting features:
- Dynamic volatility-dependent spread expansion
- Non-linear volume participation market impact slippage
- Execution latency modeling
- Chronological intracandle price path simulation (O->L->H->C / O->H->L->C)
- Partial TP1 scale-out and Breakeven stop adjustment
"""

import math
from datetime import datetime, timezone
import pytest
import numpy as np
import pandas as pd

from engines.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    Position,
    TradeRecord,
    TradeSignal,
    TradeSide,
    ExitReason,
    compute_dynamic_spread,
    compute_dynamic_slippage,
)


def test_dynamic_spread_calculation():
    """Verifies that spread widens when ATR/Price volatility expands."""
    base_spread = 0.0001  # 1 bps
    price = 50000.0

    # Low volatility (0.2% ATR/Price)
    spread_low = compute_dynamic_spread(base_spread, price, atr_val=100.0, enabled=True)
    assert spread_low == base_spread

    # Normal volatility (0.5% ATR/Price)
    spread_norm = compute_dynamic_spread(base_spread, price, atr_val=250.0, enabled=True)
    assert spread_norm >= base_spread

    # High volatility surge (2.0% ATR/Price)
    spread_high = compute_dynamic_spread(base_spread, price, atr_val=1000.0, enabled=True)
    assert spread_high > spread_norm
    assert spread_high <= 0.0025

    # Disabled flag returns base spread
    assert compute_dynamic_spread(base_spread, price, atr_val=1000.0, enabled=False) == base_spread


def test_dynamic_slippage_and_latency():
    """Verifies volume participation impact and latency penalty."""
    base_slip = 0.0002
    price = 100.0
    bar_vol = 50000.0
    atr_val = 2.0

    # Small retail order (negligible participation)
    slip_small, impact_small, latency_drift = compute_dynamic_slippage(
        base_slippage=base_slip,
        price=price,
        order_size=10.0,
        bar_volume=bar_vol,
        atr_val=atr_val,
        latency_ms=85.0,
        volume_impact_enabled=True,
    )
    assert slip_small >= base_slip

    # Large institutional order (heavy participation)
    slip_large, impact_large, _ = compute_dynamic_slippage(
        base_slippage=base_slip,
        price=price,
        order_size=5000.0,
        bar_volume=bar_vol,
        atr_val=atr_val,
        latency_ms=85.0,
        volume_impact_enabled=True,
    )
    assert impact_large > impact_small
    assert slip_large > slip_small

    # Latency ms variation
    _, _, drift_low = compute_dynamic_slippage(base_slip, price, 10.0, bar_vol, atr_val, latency_ms=10.0)
    _, _, drift_high = compute_dynamic_slippage(base_slip, price, 10.0, bar_vol, atr_val, latency_ms=250.0)
    assert drift_high > drift_low


def test_intracandle_path_bullish_candle_sl_first():
    """
    On a Green/Bullish candle (O -> L -> H -> C):
    If price dips to L <= SL first, LONG trade stops out before H (TP) is ever reached.
    """
    cfg = BacktestConfig(intracandle_mode="path", slippage=0.0, spread=0.0)
    engine = BacktestEngine(config=cfg)

    pos = Position(
        trade_id=1,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_index=1,
        entry_time=datetime(2026, 1, 1),
        entry_price=100.0,
        initial_sl=95.0,
        current_sl=95.0,
        tp1_price=110.0,
        tp2_price=115.0,
        initial_size=10.0,
        remaining_size=10.0,
        risk_amount=50.0,
        setup_type="TEST",
    )

    # Bullish candle: Open=98.0, Low=94.0 (triggers SL), High=112.0 (would hit TP1), Close=111.0
    rec = engine._process_active_position(
        pos=pos,
        t=2,
        open_price=98.0,
        high_price=112.0,
        low_price=94.0,
        close_price=111.0,
        timestamp=datetime(2026, 1, 1, 1),
        cfg=cfg,
        equity_before=10000.0,
    )

    assert rec is not None
    assert rec.exit_reason == ExitReason.STOP_LOSS.value
    assert rec.tp1_hit is False
    assert rec.net_pnl < 0


def test_intracandle_path_bearish_candle_tp_first_for_long():
    """
    On a Red/Bearish candle (O -> H -> L -> C):
    For LONG: Price first rallies to H >= TP1, locks in partial scale-out, moves SL to BE,
    before dropping to L.
    """
    cfg = BacktestConfig(intracandle_mode="path", move_be_at_tp1=True, slippage=0.0, spread=0.0)
    engine = BacktestEngine(config=cfg)

    pos = Position(
        trade_id=2,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_index=1,
        entry_time=datetime(2026, 1, 1),
        entry_price=100.0,
        initial_sl=95.0,
        current_sl=95.0,
        tp1_price=110.0,
        tp2_price=120.0,
        initial_size=10.0,
        remaining_size=10.0,
        risk_amount=50.0,
        setup_type="TEST",
    )

    # Bearish candle: Open=105.0, High=112.0 (hits TP1), Low=94.0 (hits BE stop), Close=96.0
    rec = engine._process_active_position(
        pos=pos,
        t=2,
        open_price=105.0,
        high_price=112.0,
        low_price=94.0,
        close_price=96.0,
        timestamp=datetime(2026, 1, 1, 1),
        cfg=cfg,
        equity_before=10000.0,
    )

    assert rec is not None
    assert rec.tp1_hit is True
    # Stopped out at Breakeven on the runner
    assert rec.exit_reason == ExitReason.BREAKEVEN.value
    # Net PnL should be strictly positive because 50% was closed at TP1 (+10.0 pts)!
    assert rec.net_pnl > 0


def test_intracandle_path_short_execution():
    """
    On a Red/Bearish candle (O -> H -> L -> C):
    For SHORT: If price spikes up to H >= SL first, it stops out before dropping to L (TP).
    """
    cfg = BacktestConfig(intracandle_mode="path", slippage=0.0, spread=0.0)
    engine = BacktestEngine(config=cfg)

    pos = Position(
        trade_id=3,
        symbol="BTCUSDT",
        side=TradeSide.SHORT,
        entry_index=1,
        entry_time=datetime(2026, 1, 1),
        entry_price=100.0,
        initial_sl=105.0,
        current_sl=105.0,
        tp1_price=90.0,
        tp2_price=80.0,
        initial_size=10.0,
        remaining_size=10.0,
        risk_amount=50.0,
        setup_type="TEST",
    )

    # Bearish candle: Open=102.0, High=106.0 (hits SL), Low=88.0 (would hit TP1), Close=89.0
    rec = engine._process_active_position(
        pos=pos,
        t=2,
        open_price=102.0,
        high_price=106.0,
        low_price=88.0,
        close_price=89.0,
        timestamp=datetime(2026, 1, 1, 1),
        cfg=cfg,
        equity_before=10000.0,
    )

    assert rec is not None
    assert rec.exit_reason == ExitReason.STOP_LOSS.value
    assert rec.tp1_hit is False
    assert rec.net_pnl < 0
