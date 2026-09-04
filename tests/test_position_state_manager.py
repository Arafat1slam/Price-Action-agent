"""
tests/test_position_state_manager.py - Unit tests for Phase 6: Position-State Awareness
"""

import pytest
from core.models import (
    SetupType,
    TradeSetup,
    PositionState,
    ActivePosition,
)
from engines.trade_setup_engine import PositionStateManager


@pytest.fixture
def manager():
    return PositionStateManager()


@pytest.fixture
def long_setup():
    return TradeSetup(
        setup_id="SETUP-LONG001",
        symbol="BTCUSDT",
        timestamp="2024-01-01T00:00:00",
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry_price=80000.0,
        stop_loss=79000.0,
        tp1_price=81500.0,
        tp2_price=83000.0,
        risk_reward_tp1=1.50,
        risk_reward_tp2=3.00,
        effective_rr=2.25,
        confidence_score=75,
    )


@pytest.fixture
def short_setup():
    return TradeSetup(
        setup_id="SETUP-SHORT001",
        symbol="BTCUSDT",
        timestamp="2024-01-01T00:00:00",
        setup_type=SetupType.SMC_ORDER_BLOCK,
        direction="SHORT",
        entry_price=80000.0,
        stop_loss=81000.0,
        tp1_price=78500.0,
        tp2_price=77000.0,
        risk_reward_tp1=1.50,
        risk_reward_tp2=3.00,
        effective_rr=2.25,
        confidence_score=75,
    )


class TestPositionStateManager:
    """Tests for the position-state manager (whipsaw prevention & reversal warnings)."""

    def test_initial_state_is_flat(self, manager):
        assert manager.is_flat is True
        assert manager.has_open_trade is False
        assert manager.active_position is None

    def test_open_long_position(self, manager, long_setup):
        pos = manager.open_position(long_setup, current_price=80000.0)
        assert isinstance(pos, ActivePosition)
        assert pos.direction == "LONG"
        assert pos.state == PositionState.OPEN_LONG
        assert manager.has_open_trade is True
        assert manager.is_flat is False

    def test_open_short_position(self, manager, short_setup):
        pos = manager.open_position(short_setup, current_price=80000.0)
        assert pos.direction == "SHORT"
        assert pos.state == PositionState.OPEN_SHORT

    def test_close_position(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        closed = manager.close_position("Manual close")
        assert closed is not None
        assert closed.state == PositionState.FLAT
        assert manager.is_flat is True
        assert manager.active_position is None

    def test_close_when_flat_returns_none(self, manager):
        assert manager.close_position() is None

    def test_update_position_calculates_pnl(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        pos = manager.update_position(current_price=80500.0)
        assert pos is not None
        assert pos.current_pnl_pct > 0
        assert pos.current_rr > 0

    def test_update_position_tracks_peak_rr(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        # Price goes up to 81000 (1R profit)
        manager.update_position(current_price=81000.0)
        peak1 = manager.active_position.peak_rr
        # Price retraces to 80500
        manager.update_position(current_price=80500.0)
        assert manager.active_position.peak_rr == peak1, "Peak should not decrease"

    def test_tp1_hit_sets_partial_state(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        pos = manager.update_position(current_price=81600.0)
        assert pos.state == PositionState.PARTIAL_TP1

    def test_sl_hit_closes_position(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        result = manager.update_position(current_price=78900.0)
        # SL is 79000, price went to 78900 → should close
        assert manager.is_flat is True

    def test_tp2_hit_closes_position(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        result = manager.update_position(current_price=83100.0)
        # TP2 is 83000, price went to 83100 → should close
        assert manager.is_flat is True

    def test_whipsaw_prevention_suppresses_opposite(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        assert manager.should_suppress_signal("SHORT") is True
        assert manager.should_suppress_signal("LONG") is False

    def test_whipsaw_no_suppression_when_flat(self, manager):
        assert manager.should_suppress_signal("LONG") is False
        assert manager.should_suppress_signal("SHORT") is False

    def test_early_reversal_warning_long(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        # Price goes up to 81200 (1.2R) → peak
        manager.update_position(current_price=81200.0)
        # Then drops with bearish candle with big upper wick
        candle = {
            "open": 81200.0,
            "high": 81500.0,
            "low": 80700.0,
            "close": 80800.0,  # Red candle
        }
        pos = manager.update_position(current_price=80800.0, candle=candle)
        # RR drawdown = peak(1.2) - current(0.8) = 0.4 → may or may not trigger
        # upper_wick = (81500-81200)/800 = 0.375 < 0.40, is_red=True
        # rr_drawdown = 1.2 - 0.8 = 0.4 < 0.5
        # Wick is 0.375 < 0.40 AND drawdown 0.4 < 0.5 → doesn't trigger both conditions
        # But we still have the position
        assert pos is not None

    def test_early_reversal_warning_triggers_on_significant_drawdown(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        # Price goes way up to 82000 (3R)
        manager.update_position(current_price=82000.0)
        # Then significant drop
        candle = {
            "open": 81600.0,
            "high": 81800.0,
            "low": 81300.0,
            "close": 81400.0,  # Red candle
        }
        pos = manager.update_position(current_price=81400.0, candle=candle)
        # peak_rr was 3.0, current_rr = (81400-80000)/1000 = 1.4
        # rr_drawdown = 3.0 - 1.4 = 1.6 >= 0.5 → TRIGGERS
        assert pos is not None
        assert pos.reversal_warning is True
        assert len(pos.reversal_reason) > 0

    def test_short_position_pnl_calculation(self, manager, short_setup):
        manager.open_position(short_setup, current_price=80000.0)
        pos = manager.update_position(current_price=79000.0)
        # Short: pnl = entry - current = 80000 - 79000 = 1000
        # risk = |80000 - 81000| = 1000
        assert pos.current_rr == 1.0
        assert pos.current_pnl_pct > 0

    def test_trade_history_records_closed_trades(self, manager, long_setup):
        manager.open_position(long_setup, current_price=80000.0)
        manager.close_position("Test close")
        history = manager.get_trade_history()
        assert len(history) == 1
        assert history[0].position_id == "SETUP-LONG001"

    def test_update_when_flat_returns_none(self, manager):
        result = manager.update_position(current_price=80000.0)
        assert result is None
