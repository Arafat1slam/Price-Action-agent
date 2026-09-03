"""
engines/backtest_engine.py - Event-Driven Quantitative Backtesting Simulator & Performance Benchmarker

Architectural Specification:
1. Zero-Lookahead Simulation:
   - Bar-by-bar chronological iteration through historical OHLCV data.
   - Signals generated at candle close t are queued and executed strictly at candle t+1 open (or limit fill).
   - Multi-timeframe and fractal swing confirmations enforce causal confirmation lag (k + radius).

2. Realistic Institutional Execution Modeling:
   - Exchange Fees: Binance VIP0 fee tier (0.05% taker, 0.04% maker).
   - Slippage Modeling: Realistic 0.02% adverse slippage on market entries, stop losses, and market exits.
   - Spread Buffers: 0.01% bid-ask spread simulation.
   - Intracandle Pessimistic Execution: If both Stop Loss and Take Profit price levels fall within
     the [Low, High] range of the same bar, Stop Loss is modeled as hit first.

3. Position Sizing & Capital Management:
   - Dynamic account balance and equity tracking starting at $10,000.
   - 1.5% (configurable 1-2%) risk per trade based on account equity and distance to structural stop loss.
   - Capital & leverage constraints to prevent over-allocation.

4. Trade Management:
   - Invalidation Stop Loss placed beyond structural swing, Order Block (OB), or Liquidity Sweep (+- 0.2 ATR buffer).
   - Take Profit 1 (TP1): 50% scale-out at 1:2.0 R:R (maker fee).
   - Breakeven Adjustment: Stop loss immediately advanced to Breakeven (entry price + spread buffer) once TP1 is hit.
   - Take Profit 2 (TP2): Runner (remaining 50%) closed at 1:3.0 R:R (or trailing / next structural zone).

5. Quantitative Metrics Suite:
   - Total Trades, Long/Short breakdown, Win Rate %, Profit Factor, Max Drawdown %, Net Profit %,
     Expectancy (R-multiple), Sharpe Ratio (annualized crypto 24/7), Sortino Ratio, Avg Trade Duration,
     Realized vs Projected R:R, Win/Loss Ratio, and Consecutive Streak Analytics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table


# ============================================================================
# Enums & Configurations
# ============================================================================

class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    BREAKEVEN = "BREAKEVEN"
    TAKE_PROFIT_1 = "TAKE_PROFIT_1"
    TAKE_PROFIT_2 = "TAKE_PROFIT_2"
    TIMEOUT = "TIMEOUT"
    END_OF_DATA = "END_OF_DATA"


@dataclass
class BacktestConfig:
    """Institutional execution and simulation parameter configuration."""
    initial_capital: float = 10000.0
    risk_per_trade: float = 0.015       # 1.5% equity risk per trade (within 1-2% institutional range)
    taker_fee: float = 0.0005           # Binance VIP0 spot/futures taker fee (0.05%)
    maker_fee: float = 0.0004           # Binance VIP0 spot/futures maker fee (0.04%)
    slippage: float = 0.0002            # 0.02% adverse slippage on market/stop fills
    spread: float = 0.0001              # 0.01% bid-ask spread simulation buffer
    tp1_rr: float = 2.0                 # TP1 target at 1:2.0 R:R
    tp2_rr: float = 3.0                 # TP2 runner at 1:3.0 R:R
    tp1_ratio: float = 0.50             # 50% scale-out at TP1
    move_be_at_tp1: bool = True         # Advance stop loss to breakeven after TP1 hit
    pessimistic_intracandle: bool = True# SL executed first if both SL and TP in same candle
    max_active_trades: int = 1          # Single concurrent position per symbol
    max_holding_bars: int = 80          # Maximum bars before timeout exit
    min_risk_atr: float = 0.35          # Minimum stop distance in ATR multiples
    max_risk_atr: float = 3.5           # Maximum stop distance in ATR multiples
    atr_buffer_sl: float = 0.20         # 0.20 ATR buffer beyond structural invalidation level
    max_leverage: float = 5.0           # Maximum allowed notional leverage cap


# ============================================================================
# Trade & Order Dataclasses
# ============================================================================

@dataclass
class TradeSignal:
    """Trade setup signal generated at closed bar t for execution at t+1."""
    bar_index: int
    timestamp: Any
    symbol: str
    side: TradeSide
    entry_price: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    setup_type: str = "SMC_PRICE_ACTION"
    confidence: float = 50.0
    order_type: OrderType = OrderType.MARKET
    rationale: List[str] = field(default_factory=list)


@dataclass
class Position:
    """Active open position tracking intra-trade state and partial scale-outs."""
    trade_id: int
    symbol: str
    side: TradeSide
    entry_index: int
    entry_time: Any
    entry_price: float                  # Executed entry price (including slippage)
    initial_sl: float                   # Initial structural invalidation stop loss
    current_sl: float                   # Dynamic stop loss (updated to BE after TP1)
    tp1_price: float                    # 1:2 R:R target price
    tp2_price: float                    # 1:3 R:R target price
    initial_size: float                 # Total initial position size in coin/contract units
    remaining_size: float               # Remaining size (50% after TP1 scale-out)
    risk_amount: float                  # Initial dollar risk: size * abs(entry - initial_sl)
    setup_type: str
    tp1_hit: bool = False
    tp1_exit_time: Any = None
    tp1_exit_price: Optional[float] = None
    tp1_pnl: float = 0.0
    entry_fee: float = 0.0
    tp1_fee: float = 0.0
    runner_fee: float = 0.0
    slippage_paid: float = 0.0


@dataclass
class TradeRecord:
    """Canonical historical record of a completed trade."""
    trade_id: int
    symbol: str
    side: str                           # "LONG" or "SHORT"
    setup_type: str
    entry_index: int
    entry_time: Any
    entry_price: float
    initial_sl: float
    final_sl: float
    tp1_price: float
    tp2_price: float
    exit_index: int
    exit_time: Any
    exit_price: float
    exit_reason: str                    # "TAKE_PROFIT_2", "STOP_LOSS", "BREAKEVEN", etc.
    size: float
    risk_amount: float
    gross_pnl: float
    net_pnl: float
    net_pnl_pct: float
    realized_r: float                   # Realized R-multiple: net_pnl / risk_amount
    fees_paid: float
    slippage_paid: float
    tp1_hit: bool
    duration_bars: int
    equity_before: float
    equity_after: float


@dataclass
class BacktestMetrics:
    """Comprehensive institutional quantitative performance analytics."""
    initial_capital: float
    final_equity: float
    net_profit: float
    net_profit_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_r: float                 # Average realized R-multiple per trade
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    long_trades: int
    long_win_rate_pct: float
    short_trades: int
    short_win_rate_pct: float
    avg_trade_duration_bars: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_fees: float
    total_slippage: float


@dataclass
class BacktestResult:
    """Complete container for backtest output artifacts."""
    symbol: str
    timeframe: str
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: List[TradeRecord]
    equity_curve: pd.DataFrame


# ============================================================================
# Built-in Institutional Strategy: SMCPriceActionStrategy
# ============================================================================

class SMCPriceActionStrategy:
    """
    Unified Institutional Smart Money Concepts (SMC) & Structural Strategy:
    1. Liquidity Sweeps (BSL/SSL): Stop runs past recent swing points with wick rejection & volume.
    2. Order Block Retests (+OB/-OB): Pullbacks into validated order blocks with structural invalidation.
    3. FVG Consequent Encroachment (50%) Retests: Pullbacks into imbalances in trend direction.
    4. Confirmed Structural Invalidation Stop Loss: Set beyond swing extreme/OB boundary +- 0.2 ATR.
    5. Take Profit 1 at 1:2 R:R (50% scale-out, BE move) and Take Profit 2 at 1:3 R:R (runner).
    """

    def __init__(
        self,
        swing_radius: int = 3,
        lookback_swings: int = 50,
        min_wick_ratio: float = 0.35,
        min_rvol: float = 1.10,
        use_trend_filter: bool = True,
        trend_ema_span: int = 50,
    ):
        self.swing_radius = swing_radius
        self.lookback_swings = lookback_swings
        self.min_wick_ratio = min_wick_ratio
        self.min_rvol = min_rvol
        self.use_trend_filter = use_trend_filter
        self.trend_ema_span = trend_ema_span

    def evaluate_bar(
        self,
        df: pd.DataFrame,
        t: int,
        config: BacktestConfig,
        precomputed: Dict[str, Any],
    ) -> Optional[TradeSignal]:
        """
        Evaluates candle t with zero lookahead bias.
        All indicator values and swing confirmations at index t use only data up to t.
        """
        if t < 50 or t >= len(df) - 1:
            return None

        symbol = precomputed["symbol"]
        high = precomputed["high"]
        low = precomputed["low"]
        close = precomputed["close"]
        open_p = precomputed["open"]
        vol = precomputed["volume"]
        atr = precomputed["atr"]
        vol_sma = precomputed["vol_sma"]
        ema_trend = precomputed["ema_trend"]
        sh_list = precomputed["sh_list"]
        sl_list = precomputed["sl_list"]
        ts = precomputed["timestamps"]

        c_rng = max(high[t] - low[t], 1e-9)
        u_wick = (high[t] - max(open_p[t], close[t])) / c_rng
        l_wick = (min(open_p[t], close[t]) - low[t]) / c_rng
        rvol = vol[t] / (vol_sma[t] + 1e-9)
        curr_atr = atr[t]
        bar_ts = ts[t]

        # Recent confirmed swing points (confirmed strictly at or before t-1)
        rec_sh = [s for s in sh_list if s[2] <= t - 1 and s[0] >= t - self.lookback_swings]
        rec_sl = [s for s in sl_list if s[2] <= t - 1 and s[0] >= t - self.lookback_swings]

        # --------------------------------------------------------------------
        # 1. Liquidity Sweeps (SSL & BSL)
        # --------------------------------------------------------------------
        # SSL Sweep: Pierces swing low, closes back above, lower wick rejection
        for _, s_px, _ in rec_sl:
            if low[t] < s_px and close[t] > s_px and l_wick >= self.min_wick_ratio and rvol >= self.min_rvol:
                sl_price = round(low[t] - config.atr_buffer_sl * curr_atr, 4)
                risk = close[t] - sl_price
                if config.min_risk_atr * curr_atr <= risk <= config.max_risk_atr * curr_atr:
                    if not self.use_trend_filter or close[t] > ema_trend[t] or l_wick >= 0.45:
                        tp1 = round(close[t] + config.tp1_rr * risk, 4)
                        tp2 = round(close[t] + config.tp2_rr * risk, 4)
                        return TradeSignal(
                            bar_index=t,
                            timestamp=bar_ts,
                            symbol=symbol,
                            side=TradeSide.LONG,
                            entry_price=close[t],
                            stop_loss=sl_price,
                            tp1_price=tp1,
                            tp2_price=tp2,
                            setup_type="SSL_LIQUIDITY_SWEEP",
                            confidence=75.0,
                            rationale=[
                                f"SSL Sweep of swing low {s_px:.2f}",
                                f"Lower wick rejection {l_wick:.1%}",
                                f"Relative volume RVOL {rvol:.2f}x",
                            ],
                        )

        # BSL Sweep: Pierces swing high, closes back below, upper wick rejection
        for _, s_px, _ in rec_sh:
            if high[t] > s_px and close[t] < s_px and u_wick >= self.min_wick_ratio and rvol >= self.min_rvol:
                sl_price = round(high[t] + config.atr_buffer_sl * curr_atr, 4)
                risk = sl_price - close[t]
                if config.min_risk_atr * curr_atr <= risk <= config.max_risk_atr * curr_atr:
                    if not self.use_trend_filter or close[t] < ema_trend[t] or u_wick >= 0.45:
                        tp1 = round(close[t] - config.tp1_rr * risk, 4)
                        tp2 = round(close[t] - config.tp2_rr * risk, 4)
                        return TradeSignal(
                            bar_index=t,
                            timestamp=bar_ts,
                            symbol=symbol,
                            side=TradeSide.SHORT,
                            entry_price=close[t],
                            stop_loss=sl_price,
                            tp1_price=tp1,
                            tp2_price=tp2,
                            setup_type="BSL_LIQUIDITY_SWEEP",
                            confidence=75.0,
                            rationale=[
                                f"BSL Sweep of swing high {s_px:.2f}",
                                f"Upper wick rejection {u_wick:.1%}",
                                f"Relative volume RVOL {rvol:.2f}x",
                            ],
                        )

        # --------------------------------------------------------------------
        # 2. Order Block / FVG Pullback Confluence
        # --------------------------------------------------------------------
        # Bullish FVG pullback in uptrend
        if self.use_trend_filter and close[t] > ema_trend[t]:
            if t >= 3 and low[t-1] > high[t-3]:
                fvg_gap = low[t-1] - high[t-3]
                fvg_ce = (low[t-1] + high[t-3]) / 2.0
                if fvg_gap >= 0.3 * curr_atr and low[t] <= fvg_ce and close[t] > open_p[t]:
                    sl_price = round(high[t-3] - config.atr_buffer_sl * curr_atr, 4)
                    risk = close[t] - sl_price
                    if config.min_risk_atr * curr_atr <= risk <= config.max_risk_atr * curr_atr:
                        tp1 = round(close[t] + config.tp1_rr * risk, 4)
                        tp2 = round(close[t] + config.tp2_rr * risk, 4)
                        return TradeSignal(
                            bar_index=t,
                            timestamp=bar_ts,
                            symbol=symbol,
                            side=TradeSide.LONG,
                            entry_price=close[t],
                            stop_loss=sl_price,
                            tp1_price=tp1,
                            tp2_price=tp2,
                            setup_type="SMC_FVG_PULLBACK",
                            confidence=70.0,
                            rationale=[
                                "Bullish FVG Consequent Encroachment (50%) retest",
                                f"Trend alignment (Close > EMA{self.trend_ema_span})",
                            ],
                        )

        # Bearish FVG pullback in downtrend
        elif self.use_trend_filter and close[t] < ema_trend[t]:
            if t >= 3 and high[t-1] < low[t-3]:
                fvg_gap = low[t-3] - high[t-1]
                fvg_ce = (low[t-3] + high[t-1]) / 2.0
                if fvg_gap >= 0.3 * curr_atr and high[t] >= fvg_ce and close[t] < open_p[t]:
                    sl_price = round(low[t-3] + config.atr_buffer_sl * curr_atr, 4)
                    risk = sl_price - close[t]
                    if config.min_risk_atr * curr_atr <= risk <= config.max_risk_atr * curr_atr:
                        tp1 = round(close[t] - config.tp1_rr * risk, 4)
                        tp2 = round(close[t] - config.tp2_rr * risk, 4)
                        return TradeSignal(
                            bar_index=t,
                            timestamp=bar_ts,
                            symbol=symbol,
                            side=TradeSide.SHORT,
                            entry_price=close[t],
                            stop_loss=sl_price,
                            tp1_price=tp1,
                            tp2_price=tp2,
                            setup_type="SMC_FVG_PULLBACK",
                            confidence=70.0,
                            rationale=[
                                "Bearish FVG Consequent Encroachment (50%) retest",
                                f"Trend alignment (Close < EMA{self.trend_ema_span})",
                            ],
                        )

        return None


# ============================================================================
# Core Backtest Simulation Engine
# ============================================================================

class BacktestEngine:
    """
    Event-driven bar-by-bar chronological backtest simulation engine.
    Guarantees zero lookahead bias with institutional fee, slippage, and spread modeling.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.console = Console()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        strategy: Optional[Any] = None,
    ) -> BacktestResult:
        """
        Executes an event-driven backtest over historical OHLCV data.
        """
        df_norm = self._normalize_dataframe(df)
        n = len(df_norm)
        if n < 60:
            raise ValueError(f"Insufficient candle history for backtest: {n} bars (minimum 60 required)")

        cfg = self.config
        strat = strategy or SMCPriceActionStrategy()

        # Precompute causal technical arrays (using rolling window / causal operations)
        precomputed = self._precompute_features(df_norm, symbol, timeframe, strat)

        # Simulation state
        capital = cfg.initial_capital
        equity = capital
        trades: List[TradeRecord] = []
        active_position: Optional[Position] = None
        pending_signal: Optional[TradeSignal] = None
        trade_id_counter = 1

        high = precomputed["high"]
        low = precomputed["low"]
        close = precomputed["close"]
        open_p = precomputed["open"]
        ts = precomputed["timestamps"]

        # Track bar-by-bar equity curve
        equity_records = []

        # Chronological bar-by-bar event loop
        for t in range(50, n):
            # ----------------------------------------------------------------
            # Step 1: Manage Active Position on Candle t
            # ----------------------------------------------------------------
            if active_position is not None:
                closed_record = self._process_active_position(
                    pos=active_position,
                    t=t,
                    open_price=open_p[t],
                    high_price=high[t],
                    low_price=low[t],
                    close_price=close[t],
                    timestamp=ts[t],
                    cfg=cfg,
                    equity_before=equity,
                )
                if closed_record is not None:
                    equity += closed_record.net_pnl
                    trades.append(closed_record)
                    active_position = None

            # ----------------------------------------------------------------
            # Step 2: Process Pending Order on Candle t (Zero Lookahead)
            # ----------------------------------------------------------------
            if pending_signal is not None and active_position is None:
                new_pos = self._execute_pending_signal(
                    signal=pending_signal,
                    t=t,
                    open_price=open_p[t],
                    high_price=high[t],
                    low_price=low[t],
                    timestamp=ts[t],
                    current_equity=equity,
                    cfg=cfg,
                    trade_id=trade_id_counter,
                )
                if new_pos is not None:
                    active_position = new_pos
                    trade_id_counter += 1
                pending_signal = None

            # ----------------------------------------------------------------
            # Step 3: Evaluate Strategy at Candle Close t (For Execution at t+1)
            # ----------------------------------------------------------------
            if active_position is None and pending_signal is None and t < n - 1:
                sig = strat.evaluate_bar(df_norm, t, cfg, precomputed)
                if sig is not None:
                    pending_signal = sig

            # ----------------------------------------------------------------
            # Step 4: Record Mark-to-Market Bar Equity
            # ----------------------------------------------------------------
            unrealized_pnl = 0.0
            if active_position is not None:
                unrealized_pnl = self._calculate_unrealized_pnl(active_position, close[t], cfg)

            bar_equity = equity + unrealized_pnl
            equity_records.append({
                "index": t,
                "timestamp": ts[t],
                "close": close[t],
                "equity": bar_equity,
                "cash": equity,
                "in_trade": active_position is not None,
            })

        # Close any remaining position at end of simulation
        if active_position is not None:
            final_record = self._close_position_at_market(
                pos=active_position,
                t=n - 1,
                exit_price=close[-1],
                timestamp=ts[-1],
                reason=ExitReason.END_OF_DATA,
                cfg=cfg,
                equity_before=equity,
            )
            equity += final_record.net_pnl
            trades.append(final_record)
            if equity_records:
                equity_records[-1]["equity"] = equity

        # Compile equity DataFrame & compute comprehensive metrics
        equity_df = pd.DataFrame(equity_records)
        metrics = self._calculate_metrics(trades, equity_df, cfg.initial_capital, timeframe)

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            config=cfg,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_df,
        )

    def run_batch(
        self,
        datasets: Dict[str, pd.DataFrame],
        strategy: Optional[Any] = None,
    ) -> Dict[str, BacktestResult]:
        """
        Executes backtests across multiple symbol/timeframe datasets.
        """
        results: Dict[str, BacktestResult] = {}
        for key, df in datasets.items():
            parts = key.split("_")
            symbol = parts[0]
            tf = parts[1] if len(parts) > 1 else "1h"
            res = self.run(df, symbol=symbol, timeframe=tf, strategy=strategy)
            results[key] = res
        return results

    # ------------------------------------------------------------------------
    # Execution & Position Management Helpers
    # ------------------------------------------------------------------------

    def _execute_pending_signal(
        self,
        signal: TradeSignal,
        t: int,
        open_price: float,
        high_price: float,
        low_price: float,
        timestamp: Any,
        current_equity: float,
        cfg: BacktestConfig,
        trade_id: int,
    ) -> Optional[Position]:
        """
        Executes a queued trade signal strictly at the open of candle t.
        Applies slippage, spread buffer, taker fee, and dynamic position sizing.
        """
        if current_equity <= 0:
            return None

        # Calculate fill price with adverse slippage and spread buffer
        if signal.side == TradeSide.LONG:
            fill_price = open_price * (1.0 + cfg.slippage) * (1.0 + cfg.spread)
            if fill_price <= signal.stop_loss:
                return None  # Invalidation: gapped past stop loss
            risk_per_unit = fill_price - signal.stop_loss
        else:
            fill_price = open_price * (1.0 - cfg.slippage) * (1.0 - cfg.spread)
            if fill_price >= signal.stop_loss:
                return None  # Invalidation: gapped past stop loss
            risk_per_unit = signal.stop_loss - fill_price

        if risk_per_unit <= 1e-8:
            return None

        # Position Sizing: Risk exactly cfg.risk_per_trade of current equity
        risk_dollar = current_equity * cfg.risk_per_trade
        nominal_size = risk_dollar / risk_per_unit

        # Leverage & capital constraint check
        max_notional = current_equity * cfg.max_leverage
        if nominal_size * fill_price > max_notional:
            nominal_size = max_notional / fill_price
            risk_dollar = nominal_size * risk_per_unit

        if nominal_size <= 0:
            return None

        entry_fee = nominal_size * fill_price * cfg.taker_fee
        slippage_cost = nominal_size * fill_price * cfg.slippage

        # Adjust TP targets relative to actual executed fill price
        if signal.side == TradeSide.LONG:
            tp1 = fill_price + cfg.tp1_rr * risk_per_unit
            tp2 = fill_price + cfg.tp2_rr * risk_per_unit
        else:
            tp1 = fill_price - cfg.tp1_rr * risk_per_unit
            tp2 = fill_price - cfg.tp2_rr * risk_per_unit

        return Position(
            trade_id=trade_id,
            symbol=signal.symbol,
            side=signal.side,
            entry_index=t,
            entry_time=timestamp,
            entry_price=round(fill_price, 4),
            initial_sl=round(signal.stop_loss, 4),
            current_sl=round(signal.stop_loss, 4),
            tp1_price=round(tp1, 4),
            tp2_price=round(tp2, 4),
            initial_size=nominal_size,
            remaining_size=nominal_size,
            risk_amount=risk_dollar,
            setup_type=signal.setup_type,
            entry_fee=entry_fee,
            slippage_paid=slippage_cost,
        )

    def _process_active_position(
        self,
        pos: Position,
        t: int,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        timestamp: Any,
        cfg: BacktestConfig,
        equity_before: float,
    ) -> Optional[TradeRecord]:
        """
        Evaluates active position against candle t.
        Enforces conservative pessimistic intracandle execution, TP1 scale-out,
        Breakeven stop movement, and runner TP2 management.
        """
        side = pos.side
        duration = t - pos.entry_index

        # ==================== LONG POSITION ====================
        if side == TradeSide.LONG:
            sl_hit = low_price <= pos.current_sl
            tp1_hit = not pos.tp1_hit and high_price >= pos.tp1_price
            tp2_hit = high_price >= pos.tp2_price

            # Pessimistic execution: if SL and TP both triggered on same bar, SL triggers first!
            if cfg.pessimistic_intracandle and sl_hit and (tp1_hit or tp2_hit):
                return self._close_position_at_stop(
                    pos, t, pos.current_sl, timestamp, cfg, equity_before
                )

            # Standard Stop Loss hit
            if sl_hit:
                return self._close_position_at_stop(
                    pos, t, pos.current_sl, timestamp, cfg, equity_before
                )

            # Take Profit 1 Triggered (50% scale-out)
            if tp1_hit:
                self._execute_tp1_scaleout(pos, t, pos.tp1_price, timestamp, cfg)

                # Check if TP2 also hit on the same candle after TP1 scaleout
                if high_price >= pos.tp2_price:
                    return self._close_runner_at_tp2(
                        pos, t, pos.tp2_price, timestamp, cfg, equity_before
                    )

            # Take Profit 2 Triggered on runner
            elif pos.tp1_hit and tp2_hit:
                return self._close_runner_at_tp2(
                    pos, t, pos.tp2_price, timestamp, cfg, equity_before
                )

            # Timeout exit
            if duration >= cfg.max_holding_bars:
                return self._close_position_at_market(
                    pos, t, close_price, timestamp, ExitReason.TIMEOUT, cfg, equity_before
                )

        # ==================== SHORT POSITION ====================
        else:
            sl_hit = high_price >= pos.current_sl
            tp1_hit = not pos.tp1_hit and low_price <= pos.tp1_price
            tp2_hit = low_price <= pos.tp2_price

            # Pessimistic execution
            if cfg.pessimistic_intracandle and sl_hit and (tp1_hit or tp2_hit):
                return self._close_position_at_stop(
                    pos, t, pos.current_sl, timestamp, cfg, equity_before
                )

            # Standard Stop Loss hit
            if sl_hit:
                return self._close_position_at_stop(
                    pos, t, pos.current_sl, timestamp, cfg, equity_before
                )

            # Take Profit 1 Triggered (50% scale-out)
            if tp1_hit:
                self._execute_tp1_scaleout(pos, t, pos.tp1_price, timestamp, cfg)

                # Check if TP2 also hit on the same candle after TP1 scaleout
                if low_price <= pos.tp2_price:
                    return self._close_runner_at_tp2(
                        pos, t, pos.tp2_price, timestamp, cfg, equity_before
                    )

            # Take Profit 2 Triggered on runner
            elif pos.tp1_hit and tp2_hit:
                return self._close_runner_at_tp2(
                    pos, t, pos.tp2_price, timestamp, cfg, equity_before
                )

            # Timeout exit
            if duration >= cfg.max_holding_bars:
                return self._close_position_at_market(
                    pos, t, close_price, timestamp, ExitReason.TIMEOUT, cfg, equity_before
                )

        return None

    def _execute_tp1_scaleout(
        self,
        pos: Position,
        t: int,
        tp1_price: float,
        timestamp: Any,
        cfg: BacktestConfig,
    ) -> None:
        """
        Executes 50% scale-out at TP1 (maker fee) and advances stop loss to Breakeven.
        """
        scale_size = pos.initial_size * cfg.tp1_ratio
        if pos.side == TradeSide.LONG:
            gross_gain = scale_size * (tp1_price - pos.entry_price)
            # Advance stop loss to Breakeven plus buffer to cover round-trip fees
            if cfg.move_be_at_tp1:
                pos.current_sl = round(pos.entry_price * (1.0 + cfg.spread), 4)
        else:
            gross_gain = scale_size * (pos.entry_price - tp1_price)
            if cfg.move_be_at_tp1:
                pos.current_sl = round(pos.entry_price * (1.0 - cfg.spread), 4)

        fee = scale_size * tp1_price * cfg.maker_fee
        pos.tp1_fee = fee
        pos.tp1_pnl = gross_gain - fee
        pos.tp1_hit = True
        pos.tp1_exit_time = timestamp
        pos.tp1_exit_price = tp1_price
        pos.remaining_size -= scale_size

    def _close_runner_at_tp2(
        self,
        pos: Position,
        t: int,
        tp2_price: float,
        timestamp: Any,
        cfg: BacktestConfig,
        equity_before: float,
    ) -> TradeRecord:
        """
        Closes runner at TP2 target price (maker fee).
        """
        runner_size = pos.remaining_size
        if pos.side == TradeSide.LONG:
            runner_gross = runner_size * (tp2_price - pos.entry_price)
        else:
            runner_gross = runner_size * (pos.entry_price - tp2_price)

        fee = runner_size * tp2_price * cfg.maker_fee
        pos.runner_fee = fee
        runner_pnl = runner_gross - fee

        total_net_pnl = (pos.tp1_pnl + runner_pnl) - pos.entry_fee
        total_fees = pos.entry_fee + pos.tp1_fee + pos.runner_fee
        realized_r = total_net_pnl / (pos.risk_amount + 1e-9)

        return TradeRecord(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            side=pos.side.value,
            setup_type=pos.setup_type,
            entry_index=pos.entry_index,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            initial_sl=pos.initial_sl,
            final_sl=pos.current_sl,
            tp1_price=pos.tp1_price,
            tp2_price=pos.tp2_price,
            exit_index=t,
            exit_time=timestamp,
            exit_price=tp2_price,
            exit_reason=ExitReason.TAKE_PROFIT_2.value,
            size=pos.initial_size,
            risk_amount=pos.risk_amount,
            gross_pnl=pos.tp1_pnl + runner_gross,
            net_pnl=total_net_pnl,
            net_pnl_pct=(total_net_pnl / equity_before) * 100.0,
            realized_r=round(realized_r, 3),
            fees_paid=total_fees,
            slippage_paid=pos.slippage_paid,
            tp1_hit=True,
            duration_bars=t - pos.entry_index,
            equity_before=equity_before,
            equity_after=equity_before + total_net_pnl,
        )

    def _close_position_at_stop(
        self,
        pos: Position,
        t: int,
        sl_price: float,
        timestamp: Any,
        cfg: BacktestConfig,
        equity_before: float,
    ) -> TradeRecord:
        """
        Executes stop loss exit with adverse slippage and taker fee.
        """
        rem_size = pos.remaining_size
        if pos.side == TradeSide.LONG:
            exit_px = sl_price * (1.0 - cfg.slippage)
            gross = rem_size * (exit_px - pos.entry_price)
        else:
            exit_px = sl_price * (1.0 + cfg.slippage)
            gross = rem_size * (pos.entry_price - exit_px)

        exit_fee = rem_size * exit_px * cfg.taker_fee
        slippage_cost = rem_size * exit_px * cfg.slippage
        pos.runner_fee = exit_fee
        pos.slippage_paid += slippage_cost

        runner_pnl = gross - exit_fee
        total_net_pnl = (pos.tp1_pnl + runner_pnl) - pos.entry_fee
        total_fees = pos.entry_fee + pos.tp1_fee + pos.runner_fee
        realized_r = total_net_pnl / (pos.risk_amount + 1e-9)

        exit_reason = ExitReason.BREAKEVEN.value if pos.tp1_hit else ExitReason.STOP_LOSS.value

        return TradeRecord(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            side=pos.side.value,
            setup_type=pos.setup_type,
            entry_index=pos.entry_index,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            initial_sl=pos.initial_sl,
            final_sl=pos.current_sl,
            tp1_price=pos.tp1_price,
            tp2_price=pos.tp2_price,
            exit_index=t,
            exit_time=timestamp,
            exit_price=round(exit_px, 4),
            exit_reason=exit_reason,
            size=pos.initial_size,
            risk_amount=pos.risk_amount,
            gross_pnl=pos.tp1_pnl + gross,
            net_pnl=total_net_pnl,
            net_pnl_pct=(total_net_pnl / equity_before) * 100.0,
            realized_r=round(realized_r, 3),
            fees_paid=total_fees,
            slippage_paid=pos.slippage_paid,
            tp1_hit=pos.tp1_hit,
            duration_bars=t - pos.entry_index,
            equity_before=equity_before,
            equity_after=equity_before + total_net_pnl,
        )

    def _close_position_at_market(
        self,
        pos: Position,
        t: int,
        market_price: float,
        timestamp: Any,
        reason: ExitReason,
        cfg: BacktestConfig,
        equity_before: float,
    ) -> TradeRecord:
        """
        Executes market exit (timeout or end-of-data) with slippage and taker fee.
        """
        rem_size = pos.remaining_size
        if pos.side == TradeSide.LONG:
            exit_px = market_price * (1.0 - cfg.slippage)
            gross = rem_size * (exit_px - pos.entry_price)
        else:
            exit_px = market_price * (1.0 + cfg.slippage)
            gross = rem_size * (pos.entry_price - exit_px)

        exit_fee = rem_size * exit_px * cfg.taker_fee
        slippage_cost = rem_size * exit_px * cfg.slippage
        pos.runner_fee = exit_fee
        pos.slippage_paid += slippage_cost

        runner_pnl = gross - exit_fee
        total_net_pnl = (pos.tp1_pnl + runner_pnl) - pos.entry_fee
        total_fees = pos.entry_fee + pos.tp1_fee + pos.runner_fee
        realized_r = total_net_pnl / (pos.risk_amount + 1e-9)

        return TradeRecord(
            trade_id=pos.trade_id,
            symbol=pos.symbol,
            side=pos.side.value,
            setup_type=pos.setup_type,
            entry_index=pos.entry_index,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            initial_sl=pos.initial_sl,
            final_sl=pos.current_sl,
            tp1_price=pos.tp1_price,
            tp2_price=pos.tp2_price,
            exit_index=t,
            exit_time=timestamp,
            exit_price=round(exit_px, 4),
            exit_reason=reason.value,
            size=pos.initial_size,
            risk_amount=pos.risk_amount,
            gross_pnl=pos.tp1_pnl + gross,
            net_pnl=total_net_pnl,
            net_pnl_pct=(total_net_pnl / equity_before) * 100.0,
            realized_r=round(realized_r, 3),
            fees_paid=total_fees,
            slippage_paid=pos.slippage_paid,
            tp1_hit=pos.tp1_hit,
            duration_bars=t - pos.entry_index,
            equity_before=equity_before,
            equity_after=equity_before + total_net_pnl,
        )

    def _calculate_unrealized_pnl(self, pos: Position, current_price: float, cfg: BacktestConfig) -> float:
        """Calculates marked-to-market unrealized PnL of active position."""
        if pos.side == TradeSide.LONG:
            unrealized_runner = pos.remaining_size * (current_price - pos.entry_price)
        else:
            unrealized_runner = pos.remaining_size * (pos.entry_price - current_price)
        return pos.tp1_pnl + unrealized_runner - pos.entry_fee

    # ------------------------------------------------------------------------
    # Quantitative Analytics & Performance Metrics
    # ------------------------------------------------------------------------

    def _calculate_metrics(
        self,
        trades: List[TradeRecord],
        equity_df: pd.DataFrame,
        initial_capital: float,
        timeframe: str,
    ) -> BacktestMetrics:
        """
        Computes institutional-grade quantitative metrics from trade journal and equity curve.
        """
        final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else initial_capital
        net_profit = final_equity - initial_capital
        net_profit_pct = (net_profit / initial_capital) * 100.0

        total_trades = len(trades)
        if total_trades == 0:
            return BacktestMetrics(
                initial_capital=initial_capital,
                final_equity=final_equity,
                net_profit=0.0,
                net_profit_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                breakeven_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                expectancy_r=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                long_trades=0,
                long_win_rate_pct=0.0,
                short_trades=0,
                short_win_rate_pct=0.0,
                avg_trade_duration_bars=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                win_loss_ratio=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                total_fees=0.0,
                total_slippage=0.0,
            )

        wins = [tr for tr in trades if tr.net_pnl > 0]
        losses = [tr for tr in trades if tr.net_pnl < 0]
        breakevens = [tr for tr in trades if tr.net_pnl == 0]

        win_rate_pct = (len(wins) / total_trades) * 100.0
        gross_profit = sum(tr.net_pnl for tr in wins)
        gross_loss = abs(sum(tr.net_pnl for tr in losses))
        profit_factor = round(gross_profit / (gross_loss + 1e-9), 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        # Realized R-Multiple Expectancy
        expectancy_r = round(float(np.mean([tr.realized_r for tr in trades])), 3)

        # Max Drawdown
        equity_series = equity_df["equity"].values
        peak = np.maximum.accumulate(equity_series)
        drawdown_pct = (peak - equity_series) / np.maximum(peak, 1e-9) * 100.0
        max_drawdown_pct = round(float(np.max(drawdown_pct)), 2)
        max_drawdown = round(float(np.max(peak - equity_series)), 2)

        # Sharpe & Sortino Ratio (annualized for 24/7 crypto markets)
        ann_factor = math.sqrt(8760) if timeframe in ["1h", "60m"] else (
            math.sqrt(35040) if timeframe in ["15m"] else math.sqrt(2190)
        )
        bar_returns = np.diff(equity_series) / np.maximum(equity_series[:-1], 1e-9)
        mean_ret = float(np.mean(bar_returns)) if len(bar_returns) > 0 else 0.0
        std_ret = float(np.std(bar_returns)) if len(bar_returns) > 0 else 1e-9
        sharpe_ratio = round((mean_ret / (std_ret + 1e-9)) * ann_factor, 2)

        # Sortino (penalizes only downside volatility)
        downside_returns = bar_returns[bar_returns < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 1e-9
        sortino_ratio = round((mean_ret / (downside_std + 1e-9)) * ann_factor, 2)

        # Long vs Short
        longs = [tr for tr in trades if tr.side == "LONG"]
        shorts = [tr for tr in trades if tr.side == "SHORT"]
        long_wr = (len([tr for tr in longs if tr.net_pnl > 0]) / len(longs) * 100.0) if longs else 0.0
        short_wr = (len([tr for tr in shorts if tr.net_pnl > 0]) / len(shorts) * 100.0) if shorts else 0.0

        # Trade Durations & Averages
        avg_dur = float(np.mean([tr.duration_bars for tr in trades]))
        avg_win = float(np.mean([tr.net_pnl for tr in wins])) if wins else 0.0
        avg_loss = abs(float(np.mean([tr.net_pnl for tr in losses]))) if losses else 0.0
        win_loss_ratio = round(avg_win / (avg_loss + 1e-9), 2)

        # Consecutive Streaks
        cur_w, max_w, cur_l, max_l = 0, 0, 0, 0
        for tr in trades:
            if tr.net_pnl > 0:
                cur_w += 1
                cur_l = 0
                max_w = max(max_w, cur_w)
            elif tr.net_pnl < 0:
                cur_l += 1
                cur_w = 0
                max_l = max(max_l, cur_l)

        total_fees = round(sum(tr.fees_paid for tr in trades), 2)
        total_slip = round(sum(tr.slippage_paid for tr in trades), 2)

        return BacktestMetrics(
            initial_capital=initial_capital,
            final_equity=round(final_equity, 2),
            net_profit=round(net_profit, 2),
            net_profit_pct=round(net_profit_pct, 2),
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            breakeven_trades=len(breakevens),
            win_rate_pct=round(win_rate_pct, 2),
            profit_factor=profit_factor,
            expectancy_r=expectancy_r,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            long_trades=len(longs),
            long_win_rate_pct=round(long_wr, 2),
            short_trades=len(shorts),
            short_win_rate_pct=round(short_wr, 2),
            avg_trade_duration_bars=round(avg_dur, 1),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            win_loss_ratio=win_loss_ratio,
            max_consecutive_wins=max_w,
            max_consecutive_losses=max_l,
            total_fees=total_fees,
            total_slippage=total_slip,
        )

    # ------------------------------------------------------------------------
    # Internal Helpers: Data & Features
    # ------------------------------------------------------------------------

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes columns to lowercase and sorts chronologically."""
        clean_df = df.copy()
        clean_df.columns = [c.lower() for c in clean_df.columns]
        req = ["open", "high", "low", "close", "volume"]
        for col in req:
            if col not in clean_df.columns:
                raise ValueError(f"Missing required OHLCV column: '{col}'")
        if "timestamp" in clean_df.columns:
            clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])
            clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)
        return clean_df

    def _precompute_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strat: SMCPriceActionStrategy,
    ) -> Dict[str, Any]:
        """
        Precalculates technical indicators and confirmed fractal swings causally.
        Zero lookahead guarantee: Swing points confirmed at k + radius are strictly tagged.
        """
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)
        open_p = df["open"].values.astype(np.float64)
        vol = df["volume"].values.astype(np.float64)
        ts = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

        # Vectorized True Range & ATR(14)
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        tr = np.insert(tr, 0, high[0] - low[0])
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
        atr = np.maximum(atr, 1e-6)

        # Volume SMA(20)
        vol_sma = pd.Series(vol).rolling(20, min_periods=1).mean().values

        # Trend Filter EMA
        ema_trend = pd.Series(close).ewm(span=strat.trend_ema_span, adjust=False).mean().values

        # Fractal Swings (radius=3, confirmed at k + 3)
        r = strat.swing_radius
        n = len(df)
        sh_list: List[Tuple[int, float, int]] = []
        sl_list: List[Tuple[int, float, int]] = []

        for k in range(r, n - r):
            p_h = high[k]
            if all(high[k - j] < p_h and high[k + j] < p_h for j in range(1, r + 1)):
                sh_list.append((k, float(p_h), k + r))
            p_l = low[k]
            if all(low[k - j] > p_l and low[k + j] > p_l for j in range(1, r + 1)):
                sl_list.append((k, float(p_l), k + r))

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "high": high,
            "low": low,
            "close": close,
            "open": open_p,
            "volume": vol,
            "atr": atr,
            "vol_sma": vol_sma,
            "ema_trend": ema_trend,
            "sh_list": sh_list,
            "sl_list": sl_list,
            "timestamps": ts,
        }

    # ------------------------------------------------------------------------
    # Terminal Display & Report Generation
    # ------------------------------------------------------------------------

    def build_summary_table(self, res: BacktestResult) -> Table:
        """Constructs an institutional rich summary table for an individual benchmark run."""
        m = res.metrics
        table = Table(
            title=f"[Backtest Benchmark: {res.symbol} ({res.timeframe}) — 5,000 Candles]",
            border_style="cyan",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="bold white", width=26)
        table.add_column("Value", justify="right", width=22)
        table.add_column("Benchmark Context / Institutional Target", style="dim", width=42)

        net_profit_color = "green" if m.net_profit >= 0 else "red"
        pf_color = "green" if m.profit_factor >= 1.0 else "red"
        exp_color = "green" if m.expectancy_r >= 0 else "red"

        table.add_row("Starting Capital", f"${m.initial_capital:,.2f}", "Initial allocated balance")
        table.add_row("Final Account Equity", f"${m.final_equity:,.2f}", "Net equity after fees & slippage")
        table.add_row("Net Profit ($ / %)", f"[{net_profit_color}]${m.net_profit:+,.2f} ({m.net_profit_pct:+.2f}%)[/]", "Target: positive alpha")
        table.add_row("Total Trades Closed", f"{m.total_trades}", f"Long: {m.long_trades} | Short: {m.short_trades}")
        table.add_row("Win Rate (%)", f"{m.win_rate_pct:.1f}%", f"{m.winning_trades}W / {m.losing_trades}L / {m.breakeven_trades}BE")
        table.add_row("Profit Factor", f"[{pf_color}]{m.profit_factor:.2f}[/]", "Target: >= 1.20 (Gross Win / Gross Loss)")
        table.add_row("Expectancy (R-Multiple)", f"[{exp_color}]{m.expectancy_r:+.3f} R[/]", "Average R realized per trade")
        table.add_row("Max Drawdown (%)", f"[yellow]-{m.max_drawdown_pct:.2f}%[/] (${m.max_drawdown:,.2f})", "Peak-to-trough drawdown")
        table.add_row("Sharpe Ratio (24/7 Crypto)", f"{m.sharpe_ratio:.2f}", "Annualized risk-adjusted return")
        table.add_row("Sortino Ratio", f"{m.sortino_ratio:.2f}", "Penalizes only downside risk")
        table.add_row("Win / Loss Ratio", f"{m.win_loss_ratio:.2f}", f"Avg Win: ${m.avg_win:,.2f} | Avg Loss: ${m.avg_loss:,.2f}")
        table.add_row("Exchange Fees & Slippage", f"${m.total_fees + m.total_slippage:,.2f}", f"Fees: ${m.total_fees:,.2f} | Slip: ${m.total_slippage:,.2f}")
        table.add_row("Avg Holding Duration", f"{m.avg_trade_duration_bars:.1f} bars", "Chronological trade holding span")
        table.add_row("Max Consecutive Streaks", f"{m.max_consecutive_wins} Wins / {m.max_consecutive_losses} Losses", "Risk management stress metric")

        return table

    def build_comparison_table(self, results: List[BacktestResult]) -> Table:
        """Constructs a consolidated multi-asset benchmark table."""
        table = Table(
            title="[Institutional Strategy Benchmark Suite — Multi-Asset Summary]",
            border_style="bright_blue",
            show_header=True,
            header_style="bold yellow",
        )
        table.add_column("Symbol", style="bold cyan", width=10)
        table.add_column("Timeframe", justify="center", width=10)
        table.add_column("Trades", justify="right", width=8)
        table.add_column("Win Rate", justify="right", width=10)
        table.add_column("Profit Factor", justify="right", width=14)
        table.add_column("Expectancy", justify="right", width=12)
        table.add_column("Net Profit", justify="right", width=16)
        table.add_column("Max DD", justify="right", width=10)
        table.add_column("Sharpe", justify="right", width=9)
        table.add_column("Sortino", justify="right", width=9)

        for res in results:
            m = res.metrics
            np_col = "green" if m.net_profit >= 0 else "red"
            pf_col = "green" if m.profit_factor >= 1.0 else "red"
            exp_col = "green" if m.expectancy_r >= 0 else "red"

            table.add_row(
                res.symbol,
                res.timeframe,
                str(m.total_trades),
                f"{m.win_rate_pct:.1f}%",
                f"[{pf_col}]{m.profit_factor:.2f}[/]",
                f"[{exp_col}]{m.expectancy_r:+.2f} R[/]",
                f"[{np_col}]${m.net_profit:+,.2f} ({m.net_profit_pct:+.1f}%)[/]",
                f"-{m.max_drawdown_pct:.1f}%",
                f"{m.sharpe_ratio:.2f}",
                f"{m.sortino_ratio:.2f}",
            )

        return table
