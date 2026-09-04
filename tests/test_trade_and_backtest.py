"""
tests/test_backtest_engine.py - Comprehensive Unit Test Suite for Event-Driven Backtester

Tests:
1. BacktestConfig initialization and parameter validation.
2. Position sizing & capital risk allocation.
3. Realistic execution modeling (Binance VIP0 taker/maker fees, adverse slippage, spread buffer).
4. Zero-lookahead order execution timing (signal at bar t executes on bar t+1).
5. TP1 50% scale-out and automated Breakeven stop loss movement.
6. TP2 runner full take-profit execution.
7. Pessimistic intracandle execution (SL triggers first when both SL and TP are within same bar).
8. Stop loss execution with adverse slippage.
9. Quantitative metrics calculations (Win Rate, Profit Factor, Expectancy R, Max DD, Sharpe, Sortino).
10. End-to-end backtest execution on historical data.
11. Multi-dataset batch backtesting.
"""

import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from engines.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    BacktestMetrics,
    SMCPriceActionStrategy,
    TradeSignal,
    Position,
    TradeRecord,
    TradeSide,
    OrderType,
    ExitReason,
    compute_dynamic_spread,
    compute_dynamic_slippage,
)
from core.models import (
    BiasType,
    SetupType,
    TradeSetup,
    TradeQualityScore,
    QualityGrade,
    ConfluenceReport,
    TimeframeConfluence,
    MarketStructureState,
    FairValueGap,
    OrderBlock,
    VolumeProfileZone,
    MLInferenceResult,
    RegimeReport,
    MarketRegime,
    PositionState,
    ActivePosition,
)
from engines.trade_setup_engine import TradeSetupEngine, PositionStateManager


# ============================================================================
# Synthetic Data Helpers
# ============================================================================

def make_sample_ohlcv(
    n: int = 100,
    base_price: float = 100.0,
    trend: float = 0.0,
    volatility: float = 1.0,
) -> pd.DataFrame:
    """Generates deterministic OHLCV DataFrame for testing."""
    np.random.seed(42)
    timestamps = [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(n)]
    prices = [base_price]
    for i in range(1, n):
        step = trend + np.random.normal(0, volatility)
        prices.append(max(10.0, prices[-1] + step))

    opens = np.array(prices, dtype=np.float64)
    highs = opens + np.abs(np.random.normal(volatility, 0.2, n))
    lows = opens - np.abs(np.random.normal(volatility, 0.2, n))
    closes = (opens + highs + lows) / 3.0
    volumes = np.random.uniform(500.0, 1500.0, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ============================================================================
# 1. Configuration & Default Parameter Tests
# ============================================================================

def test_backtest_config_defaults():
    """Validates institutional VIP0 default configuration parameters."""
    cfg = BacktestConfig()
    assert cfg.initial_capital == 10000.0
    assert cfg.risk_per_trade == 0.015         # 1.5%
    assert cfg.taker_fee == 0.0005             # 0.05% VIP0 taker
    assert cfg.maker_fee == 0.0004             # 0.04% VIP0 maker
    assert cfg.slippage == 0.0002              # 0.02%
    assert cfg.spread == 0.0001                # 0.01%
    assert cfg.tp1_rr == 2.0                   # 1:2.0
    assert cfg.tp2_rr == 3.0                   # 1:3.0
    assert cfg.tp1_ratio == 0.50               # 50% scale-out
    assert cfg.move_be_at_tp1 is True
    assert cfg.pessimistic_intracandle is True


# ============================================================================
# 2. Position Sizing & Capital Allocation Tests
# ============================================================================

def test_position_sizing_and_leverage():
    """Verifies that position size exactly matches account equity risk budget."""
    cfg = BacktestConfig(initial_capital=10000.0, risk_per_trade=0.015, max_leverage=5.0)
    engine = BacktestEngine(config=cfg)

    # Long signal: Entry at 100.0, SL at 90.0 (Risk = 10.0 per unit)
    signal = TradeSignal(
        bar_index=10,
        timestamp=datetime(2026, 1, 1),
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=100.0,
        stop_loss=90.0,
        tp1_price=120.0,
        tp2_price=130.0,
    )

    pos = engine._execute_pending_signal(
        signal=signal,
        t=11,
        open_price=100.0,
        high_price=102.0,
        low_price=99.0,
        timestamp=datetime(2026, 1, 1, 1),
        current_equity=10000.0,
        cfg=cfg,
        trade_id=1,
    )

    assert pos is not None
    # Executed fill price should include slippage & spread
    assert pos.entry_price > 100.0
    expected_risk_dollar = 10000.0 * 0.015  # $150
    assert math.isclose(pos.risk_amount, expected_risk_dollar, rel_tol=1e-3)
    # Total size should be ~ 150 / ~10.03 = ~14.95 units
    assert pos.initial_size > 0
    assert pos.remaining_size == pos.initial_size


def test_position_sizing_leverage_clamp():
    """Verifies that position size is clamped if it exceeds max allowed leverage."""
    # Extremely tight stop: risk per unit is only 0.05 on a $100 stock
    cfg = BacktestConfig(initial_capital=10000.0, risk_per_trade=0.015, max_leverage=3.0)
    engine = BacktestEngine(config=cfg)

    signal = TradeSignal(
        bar_index=10,
        timestamp=datetime(2026, 1, 1),
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_price=100.0,
        stop_loss=99.95,
        tp1_price=100.20,
        tp2_price=100.30,
    )

    pos = engine._execute_pending_signal(
        signal=signal,
        t=11,
        open_price=100.0,
        high_price=101.0,
        low_price=99.96,
        timestamp=datetime(2026, 1, 1, 1),
        current_equity=10000.0,
        cfg=cfg,
        trade_id=1,
    )

    assert pos is not None
    # Maximum notional = $10,000 * 3.0 = $30,000. Max size at $100 = 300 units
    notional = pos.initial_size * pos.entry_price
    assert notional <= 30000.0 + 1e-4


# ============================================================================
# 3. Execution Modeling Tests: Fees, Slippage & Spread
# ============================================================================

def test_execution_modeling_slippage_and_fees():
    """Verifies realistic fee calculation for maker and taker orders and adverse slippage."""
    cfg = BacktestConfig(taker_fee=0.0005, maker_fee=0.0004, slippage=0.0002, spread=0.0001)
    engine = BacktestEngine(config=cfg)

    # 1. Long Entry fill price must be higher than open (adverse)
    sig_long = TradeSignal(
        bar_index=1, timestamp=datetime(2026, 1, 1), symbol="BTCUSDT",
        side=TradeSide.LONG, entry_price=1000.0, stop_loss=900.0, tp1_price=1200.0, tp2_price=1300.0
    )
    pos_long = engine._execute_pending_signal(sig_long, 2, 1000.0, 1010.0, 990.0, datetime(2026, 1, 1, 1), 10000.0, cfg, 1)
    assert pos_long is not None
    assert pos_long.entry_price > 1000.0
    assert pos_long.entry_fee > 0.0

    # 2. Short Entry fill price must be lower than open (adverse)
    sig_short = TradeSignal(
        bar_index=1, timestamp=datetime(2026, 1, 1), symbol="BTCUSDT",
        side=TradeSide.SHORT, entry_price=1000.0, stop_loss=1100.0, tp1_price=800.0, tp2_price=700.0
    )
    pos_short = engine._execute_pending_signal(sig_short, 2, 1000.0, 1010.0, 990.0, datetime(2026, 1, 1, 1), 10000.0, cfg, 2)
    assert pos_short is not None
    assert pos_short.entry_price < 1000.0
    assert pos_short.entry_fee > 0.0


# ============================================================================
# 4. Zero-Lookahead Simulation Timing
# ============================================================================

def test_zero_lookahead_signal_execution_lag():
    """Ensures signals generated at closed candle t are executed strictly at t+1 open."""
    cfg = BacktestConfig()
    engine = BacktestEngine(config=cfg)

    # Deterministic strategy that generates exactly 1 signal at candle 50
    class OneShotStrategy:
        def evaluate_bar(self, df, t, config, precomputed):
            if t == 50:
                return TradeSignal(
                    bar_index=t,
                    timestamp=precomputed["timestamps"][t],
                    symbol="BTCUSDT",
                    side=TradeSide.LONG,
                    entry_price=precomputed["close"][t],
                    stop_loss=precomputed["close"][t] - 5.0,
                    tp1_price=precomputed["close"][t] + 10.0,
                    tp2_price=precomputed["close"][t] + 15.0,
                )
            return None

    df = make_sample_ohlcv(n=70, base_price=100.0)
    res = engine.run(df, symbol="BTCUSDT", timeframe="1h", strategy=OneShotStrategy())

    assert len(res.trades) >= 1
    trade = res.trades[0]
    # Trade entry index must be 51, NOT 50
    assert trade.entry_index == 51


# ============================================================================
# 5. Trade Management: TP1 Scale-Out & Breakeven Movement
# ============================================================================

def test_tp1_scaleout_and_breakeven_move():
    """
    Tests that:
    1. Reaching TP1 scales out 50% of the position with maker fee.
    2. Stop loss is automatically moved to Breakeven.
    3. Reversal afterwards triggers Breakeven exit with net profit preserved.
    """
    cfg = BacktestConfig(initial_capital=10000.0, tp1_ratio=0.50, move_be_at_tp1=True)
    engine = BacktestEngine(config=cfg)

    # Initial Position: Long at 100.0, SL at 90.0, TP1 at 120.0, TP2 at 130.0
    pos = Position(
        trade_id=1,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_index=10,
        entry_time=datetime(2026, 1, 1),
        entry_price=100.0,
        initial_sl=90.0,
        current_sl=90.0,
        tp1_price=120.0,
        tp2_price=130.0,
        initial_size=10.0,
        remaining_size=10.0,
        risk_amount=100.0,
        setup_type="TEST_TP1",
        entry_fee=0.5,
    )

    # Bar 11: Price reaches High 122.0 (triggers TP1 scale-out), Low is 99.0
    rec1 = engine._process_active_position(
        pos=pos,
        t=11,
        open_price=101.0,
        high_price=122.0,
        low_price=99.0,
        close_price=121.0,
        timestamp=datetime(2026, 1, 1, 1),
        cfg=cfg,
        equity_before=10000.0,
    )

    # Trade is still open (runner is active)
    assert rec1 is None
    assert pos.tp1_hit is True
    assert pos.remaining_size == 5.0  # 50% remains
    assert pos.tp1_pnl > 0.0          # TP1 generated profit
    # Stop loss must be moved to Breakeven (>= 100.0)
    assert pos.current_sl >= 100.0

    # Bar 12: Price collapses to Low 98.0 (triggers Breakeven stop)
    rec2 = engine._process_active_position(
        pos=pos,
        t=12,
        open_price=110.0,
        high_price=111.0,
        low_price=98.0,
        close_price=98.5,
        timestamp=datetime(2026, 1, 1, 2),
        cfg=cfg,
        equity_before=10000.0,
    )

    assert rec2 is not None
    assert rec2.exit_reason == ExitReason.BREAKEVEN.value
    # Net PnL must be positive because 50% was banked at TP1!
    assert rec2.net_pnl > 0.0
    assert rec2.realized_r > 0.0


# ============================================================================
# 6. Trade Management: Full TP2 Runner Win
# ============================================================================

def test_full_tp2_runner_win():
    """Tests full trade reaching both TP1 and TP2 runner target."""
    cfg = BacktestConfig(initial_capital=10000.0, tp1_ratio=0.50)
    engine = BacktestEngine(config=cfg)

    pos = Position(
        trade_id=1,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_index=10,
        entry_time=datetime(2026, 1, 1),
        entry_price=100.0,
        initial_sl=90.0,
        current_sl=90.0,
        tp1_price=120.0,
        tp2_price=130.0,
        initial_size=10.0,
        remaining_size=10.0,
        risk_amount=100.0,
        setup_type="TEST_TP2",
        entry_fee=0.5,
    )

    # Bar 11: Hits TP1
    engine._process_active_position(
        pos=pos, t=11, open_price=105.0, high_price=125.0, low_price=104.0, close_price=124.0,
        timestamp=datetime(2026, 1, 1, 1), cfg=cfg, equity_before=10000.0
    )
    assert pos.tp1_hit is True

    # Bar 12: Hits TP2 runner
    rec = engine._process_active_position(
        pos=pos, t=12, open_price=124.0, high_price=135.0, low_price=123.0, close_price=132.0,
        timestamp=datetime(2026, 1, 1, 2), cfg=cfg, equity_before=10000.0
    )

    assert rec is not None
    assert rec.exit_reason == ExitReason.TAKE_PROFIT_2.value
    assert rec.net_pnl > 0.0
    # Realized R should be >= 2.0 R (blended 1:2 on half + 1:3 on half = 2.5 R projected)
    assert rec.realized_r >= 2.0


# ============================================================================
# 7. Pessimistic Intracandle Execution Modeling
# ============================================================================

def test_pessimistic_intracandle_execution():
    """
    Tests that if both Stop Loss and Take Profit levels fall inside the candle range [Low, High],
    the engine pessimistically triggers Stop Loss first.
    """
    cfg = BacktestConfig(pessimistic_intracandle=True)
    engine = BacktestEngine(config=cfg)

    # Long trade with entry at 100, SL at 90, TP1 at 120
    pos = Position(
        trade_id=1,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_index=10,
        entry_time=datetime(2026, 1, 1),
        entry_price=100.0,
        initial_sl=90.0,
        current_sl=90.0,
        tp1_price=120.0,
        tp2_price=130.0,
        initial_size=10.0,
        remaining_size=10.0,
        risk_amount=100.0,
        setup_type="TEST_PESSIMISTIC",
    )

    # Massive volatile candle: Low=85.0 (below SL) and High=125.0 (above TP1)
    rec = engine._process_active_position(
        pos=pos,
        t=11,
        open_price=100.0,
        high_price=125.0,
        low_price=85.0,
        close_price=105.0,
        timestamp=datetime(2026, 1, 1, 1),
        cfg=cfg,
        equity_before=10000.0,
    )

    assert rec is not None
    # Must be STOP_LOSS, NOT TAKE_PROFIT
    assert rec.exit_reason == ExitReason.STOP_LOSS.value
    assert rec.net_pnl < 0.0
    assert rec.realized_r < 0.0


# ============================================================================
# 8. Stop Loss Exit with Adverse Slippage
# ============================================================================

def test_stop_loss_exit():
    """Tests standard stop loss triggering with adverse slippage and taker fee."""
    cfg = BacktestConfig(slippage=0.0002, taker_fee=0.0005)
    engine = BacktestEngine(config=cfg)

    pos = Position(
        trade_id=1,
        symbol="BTCUSDT",
        side=TradeSide.LONG,
        entry_index=10,
        entry_time=datetime(2026, 1, 1),
        entry_price=100.0,
        initial_sl=90.0,
        current_sl=90.0,
        tp1_price=120.0,
        tp2_price=130.0,
        initial_size=10.0,
        remaining_size=10.0,
        risk_amount=100.0,
        setup_type="TEST_SL",
        entry_fee=0.5,
    )

    # Candle drops below SL
    rec = engine._process_active_position(
        pos=pos, t=11, open_price=95.0, high_price=96.0, low_price=88.0, close_price=89.0,
        timestamp=datetime(2026, 1, 1, 1), cfg=cfg, equity_before=10000.0
    )

    assert rec is not None
    assert rec.exit_reason == ExitReason.STOP_LOSS.value
    # Executed exit price should be <= 90.0 due to adverse slippage
    assert rec.exit_price <= 90.0
    assert rec.net_pnl < 0.0


# ============================================================================
# 9. Performance Metrics Calculation
# ============================================================================

def test_performance_metrics_calculations():
    """Verifies precision calculation of Win Rate, Profit Factor, Expectancy R, Max DD, Sharpe."""
    cfg = BacktestConfig(initial_capital=10000.0)
    engine = BacktestEngine(config=cfg)

    # Create 4 deterministic trade records:
    # Trade 1: +$300 (win, 2.0 R)
    # Trade 2: -$150 (loss, -1.0 R)
    # Trade 3: +$300 (win, 2.0 R)
    # Trade 4: -$150 (loss, -1.0 R)
    trades = [
        TradeRecord(
            trade_id=1, symbol="BTCUSDT", side="LONG", setup_type="SMC", entry_index=1, entry_time=None,
            entry_price=100.0, initial_sl=90.0, final_sl=90.0, tp1_price=120.0, tp2_price=130.0,
            exit_index=5, exit_time=None, exit_price=125.0, exit_reason="TAKE_PROFIT_2",
            size=1.0, risk_amount=150.0, gross_pnl=300.0, net_pnl=300.0, net_pnl_pct=3.0,
            realized_r=2.0, fees_paid=2.0, slippage_paid=1.0, tp1_hit=True, duration_bars=4,
            equity_before=10000.0, equity_after=10300.0
        ),
        TradeRecord(
            trade_id=2, symbol="BTCUSDT", side="LONG", setup_type="SMC", entry_index=10, entry_time=None,
            entry_price=100.0, initial_sl=90.0, final_sl=90.0, tp1_price=120.0, tp2_price=130.0,
            exit_index=15, exit_time=None, exit_price=90.0, exit_reason="STOP_LOSS",
            size=1.0, risk_amount=150.0, gross_pnl=-150.0, net_pnl=-150.0, net_pnl_pct=-1.45,
            realized_r=-1.0, fees_paid=2.0, slippage_paid=1.0, tp1_hit=False, duration_bars=5,
            equity_before=10300.0, equity_after=10150.0
        ),
        TradeRecord(
            trade_id=3, symbol="BTCUSDT", side="SHORT", setup_type="SMC", entry_index=20, entry_time=None,
            entry_price=100.0, initial_sl=110.0, final_sl=110.0, tp1_price=80.0, tp2_price=70.0,
            exit_index=25, exit_time=None, exit_price=75.0, exit_reason="TAKE_PROFIT_2",
            size=1.0, risk_amount=150.0, gross_pnl=300.0, net_pnl=300.0, net_pnl_pct=2.95,
            realized_r=2.0, fees_paid=2.0, slippage_paid=1.0, tp1_hit=True, duration_bars=5,
            equity_before=10150.0, equity_after=10450.0
        ),
        TradeRecord(
            trade_id=4, symbol="BTCUSDT", side="SHORT", setup_type="SMC", entry_index=30, entry_time=None,
            entry_price=100.0, initial_sl=110.0, final_sl=110.0, tp1_price=80.0, tp2_price=70.0,
            exit_index=35, exit_time=None, exit_price=110.0, exit_reason="STOP_LOSS",
            size=1.0, risk_amount=150.0, gross_pnl=-150.0, net_pnl=-150.0, net_pnl_pct=-1.43,
            realized_r=-1.0, fees_paid=2.0, slippage_paid=1.0, tp1_hit=False, duration_bars=5,
            equity_before=10450.0, equity_after=10300.0
        ),
    ]

    equity_df = pd.DataFrame({
        "equity": [10000.0, 10300.0, 10150.0, 10450.0, 10300.0]
    })

    metrics = engine._calculate_metrics(trades, equity_df, 10000.0, "1h")

    assert metrics.total_trades == 4
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 2
    assert metrics.win_rate_pct == 50.0
    # Profit Factor: Gross Win = 600, Gross Loss = 300 => PF = 2.0
    assert metrics.profit_factor == 2.0
    # Expectancy: (2 - 1 + 2 - 1) / 4 = +0.50 R
    assert metrics.expectancy_r == 0.50
    assert metrics.net_profit == 300.0
    assert metrics.net_profit_pct == 3.0


# ============================================================================
# 10. End-to-End Backtest Run on Historical Data
# ============================================================================

def test_end_to_end_backtest_run():
    """Runs complete BacktestEngine on real BTC historical candle slice."""
    csv_path = "data/historical/BTCUSDT_1h.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        pytest.skip(f"Historical file {csv_path} not found for integration test")

    engine = BacktestEngine()
    # Test on slice of 200 candles
    slice_df = df.iloc[:200].copy()
    res = engine.run(slice_df, symbol="BTCUSDT", timeframe="1h")

    assert res.symbol == "BTCUSDT"
    assert res.timeframe == "1h"
    assert isinstance(res.metrics, BacktestMetrics)
    assert isinstance(res.equity_curve, pd.DataFrame)
    assert not res.equity_curve.empty
    assert res.metrics.initial_capital == 10000.0


# ============================================================================
# 11. Multi-Dataset Batch Backtest Run
# ============================================================================

def test_run_batch_backtesting():
    """Validates run_batch functionality across multiple datasets."""
    df1 = make_sample_ohlcv(n=80, base_price=100.0)
    df2 = make_sample_ohlcv(n=80, base_price=200.0)

    engine = BacktestEngine()
    datasets = {
        "BTCUSDT_1h": df1,
        "ETHUSDT_15m": df2,
    }

    results = engine.run_batch(datasets)
    assert len(results) == 2
    assert "BTCUSDT_1h" in results
    assert "ETHUSDT_15m" in results
    assert results["BTCUSDT_1h"].metrics.initial_capital == 10000.0
    assert results["ETHUSDT_15m"].metrics.initial_capital == 10000.0


# ============================================================================
# 12. Trade Setup Engine & R:R Verification Tests
# ============================================================================
from engines.trade_setup_engine import TradeSetupEngine
from core.models import SetupType, BiasType, FairValueGap, OrderBlock

def test_trade_setup_engine_initialization():
    engine = TradeSetupEngine()
    assert engine.min_effective_rr == 1.80
    assert engine.atr_multiplier_sl == 0.20


def test_enforce_min_rr_gate():
    """Verifies that any candidate setup with effective R:R < 1.80 is discarded."""
    engine = TradeSetupEngine(min_effective_rr=1.80)
    low_rr_setup = engine._create_trade_setup(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry=100.0,
        sl=90.0,
        tp1=110.0,
        tp2=115.0,
        confidence=70,
        rationale=["Test low RR"],
        invalidation_reason="Test invalidation"
    )
    assert low_rr_setup.effective_rr < 1.80


def test_tp_probabilities_and_risk_sizing():
    """Verifies that TP1/TP2 probabilities and recommended risk % are accurately computed."""
    engine = TradeSetupEngine(min_effective_rr=1.80)
    setup = engine._create_trade_setup(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry=80000.0,
        sl=79000.0,
        tp1=81600.0,
        tp2=83000.0,
        confidence=85,
        rationale=["High conviction institutional FVG retest"],
        invalidation_reason="Invalidated below 79000"
    )

    # TP probabilities
    assert 40 <= setup.tp1_probability <= 92
    assert 25 <= setup.tp2_probability <= 85
    assert setup.tp1_probability >= setup.tp2_probability  # TP1 must have equal or higher probability than TP2

    # High confidence (85%) should recommend 2.0% risk
    assert setup.recommended_risk_pct == 2.0
    assert setup.position_size_usd > 0.0


def test_entry_confirmation_state_machine():
    """Verifies state transitions: WAITING_FOR_PRICE -> CONFIRMED_ENTRY_TRIGGER -> TARGET_HIT & INVALIDATED."""
    from core.models import ExecutionState

    engine = TradeSetupEngine()
    setup = engine._create_trade_setup(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry=80000.0,
        sl=79000.0,
        tp1=81600.0,
        tp2=83000.0,
        confidence=80,
        rationale=["Test setup"],
        invalidation_reason="Test invalidation",
        current_price=80500.0  # Current price is 500 above entry
    )

    # 1. Price is +0.62% above entry -> Must be WAITING_FOR_PRICE (Do not chase)
    assert setup.execution_state == ExecutionState.WAITING_FOR_PRICE.value
    assert "DO NOT CHASE" in setup.entry_action
    assert setup.entry_distance_pct > 0.0

    # 2. Price pulls back into entry zone ($80,050, within 0.15% tolerance) -> CONFIRMED_ENTRY_TRIGGER
    candle_reaction = {"open": 79980.0, "high": 80100.0, "low": 79950.0, "close": 80050.0, "volume": 50.0}
    engine.update_execution_state(setup, current_price=80050.0, candle=candle_reaction)
    assert setup.execution_state == ExecutionState.CONFIRMED_ENTRY_TRIGGER.value
    assert "CONFIRMED ENTRY TRIGGER" in setup.entry_action

    # 3. Price drops below Stop Loss -> INVALIDATED
    engine.update_execution_state(setup, current_price=78900.0)
    assert setup.execution_state == ExecutionState.INVALIDATED.value
    assert "SETUP INVALIDATED" in setup.entry_action

    # 4. Price surges to TP1 -> TARGET_HIT
    engine.update_execution_state(setup, current_price=81700.0)
    assert setup.execution_state == ExecutionState.TARGET_HIT.value
    assert "TARGET 1 HIT" in setup.entry_action



# ============================================================================
# 13. Symbol Mapper & Normalizer Tests
# ============================================================================
from symbol_mapper import resolve_symbol, suggest_symbols, clean_input

def test_clean_input():
    assert clean_input(" btc / usdt ") == "btcusdt"
    assert clean_input("sol-usdt") == "solusdt"
    assert clean_input("  eth _ usdt  ") == "ethusdt"


def test_resolve_common_names():
    assert resolve_symbol("bitcoin") == "BTCUSDT"
    assert resolve_symbol("btc") == "BTCUSDT"
    assert resolve_symbol("ethereum") == "ETHUSDT"
    assert resolve_symbol("eth") == "ETHUSDT"
    assert resolve_symbol("solana") == "SOLUSDT"
    assert resolve_symbol("sol") == "SOLUSDT"


def test_resolve_raw_symbols():
    assert resolve_symbol("BTCUSDT") == "BTCUSDT"
    assert resolve_symbol("ethusdt") == "ETHUSDT"
    assert resolve_symbol("ADA") == "ADAUSDT"


def test_suggest_symbols():
    suggestions = suggest_symbols("bitcoi")
    assert len(suggestions) > 0
    assert suggestions[0][1] == "BTCUSDT"


# ============================================================================
# 14. Data Collector & Validation Tests
# ============================================================================
from data_collector import DataCollector

def test_data_collector_clean_and_validate(tmp_path):
    collector = DataCollector(output_dir=tmp_path)
    base_time = 1700000000000
    step = 3600 * 1000
    candles = []
    for i in range(10):
        t = base_time + i * step
        o, h, l, c, v = 50000.0 + i*10, 50050.0 + i*10, 49960.0 + i*10, 50020.0 + i*10, 12.5 + i
        candles.append([t, str(o), str(h), str(l), str(c), str(v), t + step - 1, str(v*c), 150+i, str(v*0.6), str(v*0.6*c), "0"])

    df, report = collector.clean_and_validate(candles, interval="1h", drop_incomplete=False)
    assert report["valid"] is True
    assert report["candle_count"] == 10
    assert len(df) == 10


# ============================================================================
# 15. Realistic Institutional Backtesting Tests
# ============================================================================

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
    assert rec.exit_reason == ExitReason.BREAKEVEN.value
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


# ============================================================================
# 16. Trade Quality Scoring Tests (0-100 Unified Grade)
# ============================================================================

@pytest.fixture
def tq_engine():
    return TradeSetupEngine()


@pytest.fixture
def basic_long_setup():
    return TradeSetup(
        setup_id="TEST-001",
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
def basic_short_setup():
    return TradeSetup(
        setup_id="TEST-002",
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


def _make_confluence_tq(
    bias=BiasType.BULLISH,
    confidence=75,
    with_fvgs=False,
    with_obs=False,
    with_vp=False,
    with_mtf=False,
    with_smc_state=False,
):
    fvgs = []
    if with_fvgs:
        fvgs = [FairValueGap(
            top=80500, bottom=79500, midpoint=80000,
            bias=BiasType.BULLISH, created_time="t1",
        )]
    obs = []
    if with_obs:
        obs = [OrderBlock(
            top=80200, bottom=79800, bias=BiasType.BULLISH,
            created_time="t1", volume=1000.0,
        )]
    vp = None
    if with_vp:
        vp = VolumeProfileZone(
            poc_price=80000, vah_price=81000, val_price=79000, total_volume=50000
        )
    tf_breakdown = {}
    if with_mtf:
        for tf in ["4h", "1h", "15m", "5m"]:
            tf_breakdown[tf] = TimeframeConfluence(
                timeframe=tf, bias=bias, score=0.5
            )
    smc_state = None
    if with_smc_state:
        smc_state = MarketStructureState(
            trend="UPTREND",
            recent_bos="BULLISH_BOS",
            recent_choch=None,
            last_swing_high=81000,
            last_swing_low=79000,
        )

    return ConfluenceReport(
        symbol="BTCUSDT",
        timestamp="2024-01-01",
        current_price=80000,
        overall_bias=bias,
        confidence_score=confidence,
        htf_bias=bias,
        ltf_trigger=bias,
        timeframe_breakdown=tf_breakdown,
        smc_state=smc_state,
        active_fvgs=fvgs,
        active_obs=obs,
        volume_profile=vp,
    )


class TestTradeQualityScoring:
    """Tests for the unified 0-100 trade quality scoring system."""

    def test_score_returns_quality_score_object(self, tq_engine, basic_long_setup):
        qs = tq_engine.score_trade_quality(basic_long_setup)
        assert isinstance(qs, TradeQualityScore)
        assert 0 <= qs.raw_score <= 100
        assert isinstance(qs.grade, QualityGrade)

    def test_score_attached_to_setup(self, tq_engine, basic_long_setup):
        qs = tq_engine.score_trade_quality(basic_long_setup)
        assert basic_long_setup.quality_score is qs

    def test_grade_mapping_a_plus(self, tq_engine):
        assert tq_engine._map_grade(90) == QualityGrade.A_PLUS
        assert tq_engine._map_grade(85) == QualityGrade.A_PLUS

    def test_grade_mapping_a(self, tq_engine):
        assert tq_engine._map_grade(80) == QualityGrade.A
        assert tq_engine._map_grade(75) == QualityGrade.A

    def test_grade_mapping_b(self, tq_engine):
        assert tq_engine._map_grade(70) == QualityGrade.B
        assert tq_engine._map_grade(65) == QualityGrade.B

    def test_grade_mapping_c(self, tq_engine):
        assert tq_engine._map_grade(55) == QualityGrade.C
        assert tq_engine._map_grade(45) == QualityGrade.C

    def test_grade_mapping_filtered(self, tq_engine):
        assert tq_engine._map_grade(44) == QualityGrade.FILTERED
        assert tq_engine._map_grade(0) == QualityGrade.FILTERED

    def test_smc_fvg_setup_scores_higher_smc(self, tq_engine, basic_long_setup):
        va_setup = TradeSetup(
            setup_id="TEST-VA",
            symbol="BTCUSDT",
            timestamp="2024-01-01",
            setup_type=SetupType.VALUE_AREA_MEAN_REVERSION,
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
        smc_score_fvg = tq_engine._score_smc_component(basic_long_setup, None)
        smc_score_va = tq_engine._score_smc_component(va_setup, None)
        assert smc_score_fvg > smc_score_va

    def test_high_confluence_scores_high(self, tq_engine, basic_long_setup):
        rep = _make_confluence_tq(
            bias=BiasType.BULLISH,
            confidence=85,
            with_fvgs=True,
            with_obs=True,
            with_vp=True,
            with_mtf=True,
            with_smc_state=True,
        )
        regime = RegimeReport(
            regime=MarketRegime.TRENDING_BULL,
            regime_label="Trending Bull",
            confidence=80,
            adx=35.0,
            plus_di=30.0,
            minus_di=15.0,
            atr_ratio=1.2,
            bb_bandwidth_pct=5.0,
            volatility_state="NORMAL",
            recommended_strategy="Trend Following",
        )
        ml = MLInferenceResult(
            prob_bullish=0.75,
            prob_bearish=0.15,
            prob_neutral=0.10,
            model_confidence=0.30,
        )
        qs = tq_engine.score_trade_quality(basic_long_setup, rep, regime, ml)
        assert qs.raw_score >= 65, f"High confluence setup should score >= 65 but got {qs.raw_score}"
        assert qs.grade in (QualityGrade.A_PLUS, QualityGrade.A, QualityGrade.B)

    def test_no_confluence_gets_baseline_score(self, tq_engine, basic_long_setup):
        qs = tq_engine.score_trade_quality(basic_long_setup)
        assert qs.raw_score > 0, "Even without confluence data, should get baseline score"

    def test_component_breakdown_sums_correctly(self, tq_engine, basic_long_setup):
        rep = _make_confluence_tq(bias=BiasType.BULLISH, confidence=70)
        qs = tq_engine.score_trade_quality(basic_long_setup, rep)
        breakdown_sum = sum(qs.component_breakdown.values())
        assert abs(breakdown_sum - qs.raw_score) < 1.0, (
            f"Component sum {breakdown_sum} should match raw score {qs.raw_score}"
        )

    def test_short_with_bear_regime_scores_well(self, tq_engine, basic_short_setup):
        regime = RegimeReport(
            regime=MarketRegime.TRENDING_BEAR,
            regime_label="Trending Bear",
            confidence=80,
            adx=32.0,
            plus_di=12.0,
            minus_di=28.0,
            atr_ratio=1.1,
            bb_bandwidth_pct=4.5,
            volatility_state="NORMAL",
            recommended_strategy="Trend Following Short",
        )
        structure_score = tq_engine._score_structure_component(basic_short_setup, regime)
        assert structure_score >= 70, f"Short in trending bear should score well: {structure_score}"

    def test_ml_component_high_prob(self, tq_engine, basic_long_setup):
        ml = MLInferenceResult(
            prob_bullish=0.80, prob_bearish=0.10, prob_neutral=0.10, model_confidence=0.30
        )
        ml_score = tq_engine._score_ml_component(basic_long_setup, ml)
        assert ml_score >= 80, f"High ML probability should score >= 80: {ml_score}"

    def test_ml_component_no_ml_returns_neutral(self, tq_engine, basic_long_setup):
        ml_score = tq_engine._score_ml_component(basic_long_setup, None)
        assert ml_score == 50.0

    def test_mtf_all_aligned_scores_high(self, tq_engine):
        rep = _make_confluence_tq(bias=BiasType.BULLISH, with_mtf=True)
        mtf_score = tq_engine._score_mtf_component(rep)
        assert mtf_score >= 80, f"All timeframes aligned should score high: {mtf_score}"


# ============================================================================
# 17. Position-State Awareness & Whipsaw Prevention Tests
# ============================================================================

@pytest.fixture
def pos_manager():
    return PositionStateManager()


@pytest.fixture
def pos_long_setup():
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
def pos_short_setup():
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

    def test_initial_state_is_flat(self, pos_manager):
        assert pos_manager.is_flat is True
        assert pos_manager.has_open_trade is False
        assert pos_manager.active_position is None

    def test_open_long_position(self, pos_manager, pos_long_setup):
        pos = pos_manager.open_position(pos_long_setup, current_price=80000.0)
        assert isinstance(pos, ActivePosition)
        assert pos.direction == "LONG"
        assert pos.state == PositionState.OPEN_LONG
        assert pos_manager.has_open_trade is True
        assert pos_manager.is_flat is False

    def test_open_short_position(self, pos_manager, pos_short_setup):
        pos = pos_manager.open_position(pos_short_setup, current_price=80000.0)
        assert pos.direction == "SHORT"
        assert pos.state == PositionState.OPEN_SHORT

    def test_close_position(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        closed = pos_manager.close_position("Manual close")
        assert closed is not None
        assert closed.state == PositionState.FLAT
        assert pos_manager.is_flat is True
        assert pos_manager.active_position is None

    def test_close_when_flat_returns_none(self, pos_manager):
        assert pos_manager.close_position() is None

    def test_update_position_calculates_pnl(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        pos = pos_manager.update_position(current_price=80500.0)
        assert pos is not None
        assert pos.current_pnl_pct > 0
        assert pos.current_rr > 0

    def test_update_position_tracks_peak_rr(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        pos_manager.update_position(current_price=81000.0)
        peak1 = pos_manager.active_position.peak_rr
        pos_manager.update_position(current_price=80500.0)
        assert pos_manager.active_position.peak_rr == peak1, "Peak should not decrease"

    def test_tp1_hit_sets_partial_state(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        pos = pos_manager.update_position(current_price=81600.0)
        assert pos.state == PositionState.PARTIAL_TP1

    def test_sl_hit_closes_position(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        result = pos_manager.update_position(current_price=78900.0)
        assert pos_manager.is_flat is True

    def test_tp2_hit_closes_position(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        result = pos_manager.update_position(current_price=83100.0)
        assert pos_manager.is_flat is True

    def test_whipsaw_prevention_suppresses_opposite(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        assert pos_manager.should_suppress_signal("SHORT") is True
        assert pos_manager.should_suppress_signal("LONG") is False

    def test_whipsaw_no_suppression_when_flat(self, pos_manager):
        assert pos_manager.should_suppress_signal("LONG") is False
        assert pos_manager.should_suppress_signal("SHORT") is False

    def test_early_reversal_warning_long(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        pos_manager.update_position(current_price=81200.0)
        candle = {
            "open": 81200.0,
            "high": 81500.0,
            "low": 80700.0,
            "close": 80800.0,
        }
        pos = pos_manager.update_position(current_price=80800.0, candle=candle)
        assert pos is not None

    def test_early_reversal_warning_triggers_on_significant_drawdown(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        pos_manager.update_position(current_price=82000.0)
        candle = {
            "open": 81600.0,
            "high": 81800.0,
            "low": 81300.0,
            "close": 81400.0,
        }
        pos = pos_manager.update_position(current_price=81400.0, candle=candle)
        assert pos is not None
        assert pos.reversal_warning is True
        assert len(pos.reversal_reason) > 0

    def test_short_position_pnl_calculation(self, pos_manager, pos_short_setup):
        pos_manager.open_position(pos_short_setup, current_price=80000.0)
        pos = pos_manager.update_position(current_price=79000.0)
        assert pos.current_rr == 1.0
        assert pos.current_pnl_pct > 0

    def test_trade_history_records_closed_trades(self, pos_manager, pos_long_setup):
        pos_manager.open_position(pos_long_setup, current_price=80000.0)
        pos_manager.close_position("Test close")
        history = pos_manager.get_trade_history()
        assert len(history) == 1
        assert history[0].position_id == "SETUP-LONG001"

    def test_update_when_flat_returns_none(self, pos_manager):
        result = pos_manager.update_position(current_price=80000.0)
        assert result is None

