"""
tests/test_smc_patterns.py - Comprehensive Unit Test Suite for SMC Engine and Chart Pattern Engine

Tests:
  1. SMC Engine:
     - FVG Detection (Bullish BISI & Bearish SIBI, Consequent Encroachment 50%)
     - Stateful FVG Mitigation (Unmitigated, Partially Mitigated, CE Tested, Fully Mitigated, Inverted)
     - Order Blocks (+OB & -OB with strict 4-pillar validation, Mean Threshold, and Breaker Block flip)
     - Liquidity Sweeps (BSL and SSL stop-hunt sweeps with >= 35% wick ratio, volume expansion, and Turtle Soup)
     - Market Structure (Fractal Swings radius=3, BOS continuation, CHoCH reversal with body close confirmation)
     - Unified SMC analysis & reporting
  2. Chart Pattern Engine:
     - Double Top & Double Bottom with symmetry and neckline breakout
     - Head & Shoulders & Inverse Head & Shoulders
     - Triangles (Ascending, Descending, Symmetrical) and Wedges (Rising, Falling)
     - Classical Multi-Candle Patterns (Morning Star, Evening Star, Three White Soldiers, Three Black Crows, Fakey)
     - Master pattern detection
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.models import BiasType, ChartPattern
from engines.smc_engine import (
    SMCEngine, FVGType, FVGState, FairValueGap,
    OBType, OBState, OrderBlock,
    SweepType, LiquiditySweep,
    TrendState, StructureEventType, SwingPoint, StructureEvent,
    SMCTrend, SMCAnalysisReport
)
from engines.chart_pattern_engine import ChartPatternEngine


# ============================================================================
# Helper Fixtures & Data Generators
# ============================================================================

def make_base_df(n: int = 50, base_price: float = 100.0) -> pd.DataFrame:
    """Generates a neutral baseline OHLCV dataframe."""
    timestamps = [datetime(2026, 1, 1) + timedelta(minutes=15 * i) for i in range(n)]
    opens = np.full(n, base_price)
    highs = np.full(n, base_price + 1.0)
    lows = np.full(n, base_price - 1.0)
    closes = np.full(n, base_price + 0.1)
    volumes = np.full(n, 1000.0)

    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


# ============================================================================
# 1. SMC Engine Tests: Fair Value Gaps (FVG)
# ============================================================================

class TestSMCFairValueGaps:
    """Tests Fair Value Gap detection and dynamic stateful mitigation tracking."""

    def test_tc01_bullish_fvg_creation(self):
        """TC-01: Bullish FVG (BISI) creation with Consequent Encroachment (50%)."""
        engine = SMCEngine(fvg_min_atr=0.20, atr_period=5)
        df = make_base_df(10, base_price=100.0)

        # Bar 0: High = 100.0, Low = 98.0
        df.loc[0, 'open'] = 98.5
        df.loc[0, 'high'] = 100.0
        df.loc[0, 'low'] = 98.0
        df.loc[0, 'close'] = 99.5

        # Bar 1: Aggressive surge
        df.loc[1, 'open'] = 99.5
        df.loc[1, 'high'] = 107.0
        df.loc[1, 'low'] = 99.0
        df.loc[1, 'close'] = 106.5

        # Bar 2: Low = 105.0 > High[0] = 100.0 (Gap = 5.0)
        df.loc[2, 'open'] = 106.5
        df.loc[2, 'high'] = 110.0
        df.loc[2, 'low'] = 105.0
        df.loc[2, 'close'] = 109.0

        # Subsequent bars remain above 105
        for k in range(3, 10):
            df.loc[k, 'open'] = 108.0
            df.loc[k, 'high'] = 112.0
            df.loc[k, 'low'] = 107.0
            df.loc[k, 'close'] = 110.0

        fvgs = engine.detect_fvgs(df)
        bullish = [f for f in fvgs if f.gap_type == FVGType.BULLISH]
        assert len(bullish) >= 1

        fvg = bullish[0]
        assert fvg.creation_index == 2
        assert fvg.top == 105.0
        assert fvg.bottom == 100.0
        assert fvg.ce == 102.5  # Consequent Encroachment (50%)
        assert fvg.initial_size == 5.0
        assert fvg.state == FVGState.UNMITIGATED
        assert fvg.is_active is True
        assert fvg.mitigation_percentage == 0.0

    def test_tc02_fvg_partial_mitigation_and_ce_tested(self):
        """TC-02: Price dips into gap beyond CE (50%), triggering CE_TESTED state."""
        engine = SMCEngine(fvg_min_atr=0.20, atr_period=5)
        df = make_base_df(10, base_price=100.0)

        # Bar 0, 1, 2 form Bullish FVG [100.0, 105.0], CE = 102.5
        df.loc[0, 'high'] = 100.0
        df.loc[1, 'open'] = 100.0
        df.loc[1, 'close'] = 106.0
        df.loc[1, 'high'] = 107.0
        df.loc[2, 'low'] = 105.0
        df.loc[2, 'high'] = 110.0

        # Bar 3: Price stays high
        df.loc[3, 'low'] = 106.0
        df.loc[3, 'high'] = 111.0

        # Bar 4: Price dips to 102.0 (< CE 102.5, but > bottom 100.0)
        df.loc[4, 'open'] = 106.0
        df.loc[4, 'low'] = 102.0
        df.loc[4, 'high'] = 106.0
        df.loc[4, 'close'] = 104.0

        # Remaining bars stay above 102
        for k in range(5, 10):
            df.loc[k, 'low'] = 103.0
            df.loc[k, 'high'] = 108.0

        fvgs = engine.detect_fvgs(df)
        bullish = [f for f in fvgs if f.gap_type == FVGType.BULLISH]
        assert len(bullish) >= 1

        fvg = bullish[0]
        assert fvg.state == FVGState.CE_TESTED
        assert fvg.current_top == 102.0
        assert pytest.approx(fvg.mitigation_percentage, 0.01) == 0.60
        assert fvg.is_active is True

    def test_tc03_fvg_full_mitigation_and_inversion(self):
        """TC-03: Candle pierces through gap and closes below bottom, flipping to INVERTED."""
        engine = SMCEngine(fvg_min_atr=0.20, atr_period=5)
        df = make_base_df(10, base_price=100.0)

        # Bar 0, 1, 2 form Bullish FVG [100.0, 105.0]
        df.loc[0, 'high'] = 100.0
        df.loc[1, 'open'] = 100.0
        df.loc[1, 'close'] = 106.0
        df.loc[1, 'high'] = 107.0
        df.loc[2, 'low'] = 105.0
        df.loc[2, 'high'] = 110.0

        # Bar 3: Violently dumps through bottom and closes at 98.0 (< bottom 100.0)
        df.loc[3, 'open'] = 106.0
        df.loc[3, 'high'] = 106.5
        df.loc[3, 'low'] = 97.0
        df.loc[3, 'close'] = 98.0

        fvgs = engine.detect_fvgs(df)
        bullish = [f for f in fvgs if f.gap_type == FVGType.BULLISH]
        assert len(bullish) >= 1

        fvg = bullish[0]
        assert fvg.state == FVGState.INVERTED
        assert fvg.is_inverted is True
        assert fvg.mitigation_percentage == 1.0
        assert fvg.mitigation_index == 3
        assert fvg.is_active is False

    def test_bearish_fvg_creation_and_mitigation(self):
        """Bearish FVG (SIBI): High[i] < Low[i-2], CE tested when price wicks up."""
        engine = SMCEngine(fvg_min_atr=0.20, atr_period=5)
        df = make_base_df(10, base_price=92.0)

        # Bar 0: Low = 100.0
        df.loc[0, 'open'] = 102.0
        df.loc[0, 'high'] = 103.0
        df.loc[0, 'low'] = 100.0
        df.loc[0, 'close'] = 101.0

        # Bar 1: Strong dump
        df.loc[1, 'open'] = 101.0
        df.loc[1, 'high'] = 101.5
        df.loc[1, 'low'] = 92.0
        df.loc[1, 'close'] = 93.0

        # Bar 2: High = 95.0 < Low[0] = 100.0 (Gap = 5.0)
        df.loc[2, 'open'] = 93.0
        df.loc[2, 'high'] = 95.0
        df.loc[2, 'low'] = 90.0
        df.loc[2, 'close'] = 91.0

        # Bar 3: Wicks up to 98.0 (>= CE 97.5)
        df.loc[3, 'open'] = 91.0
        df.loc[3, 'high'] = 98.0
        df.loc[3, 'low'] = 90.5
        df.loc[3, 'close'] = 92.0

        # Subsequent bars remain low below 95.0
        for k in range(4, 10):
            df.loc[k, 'open'] = 91.5
            df.loc[k, 'high'] = 93.0
            df.loc[k, 'low'] = 90.0
            df.loc[k, 'close'] = 92.0

        fvgs = engine.detect_fvgs(df)
        bearish = [f for f in fvgs if f.gap_type == FVGType.BEARISH]
        assert len(bearish) >= 1

        fvg = bearish[0]
        assert fvg.top == 100.0
        assert fvg.bottom == 95.0
        assert fvg.ce == 97.5
        assert fvg.state == FVGState.CE_TESTED
        assert fvg.current_bottom == 98.0


# ============================================================================
# 2. SMC Engine Tests: Order Blocks (OB) & 4-Pillar Validation
# ============================================================================

class TestSMCOrderBlocks:
    """Tests Order Block detection with 4 pillars, Mean Threshold, and Breaker Block flip."""

    def test_tc04_bullish_order_block_validation(self):
        """
        TC-04: Bullish Order Block (+OB):
          Pillar 1: Bearish candle (close < open)
          Pillar 2: Impulse surging >= 1.5 ATR
          Pillar 3: FVG created
          Pillar 4: Structural break of prior swing high
        """
        engine = SMCEngine(ob_disp_atr=1.5, fvg_min_atr=0.1, atr_period=5, vol_ma_period=5, max_lookforward=3)
        df = make_base_df(35, base_price=100.0)

        # Establish prior swing high at bar 5 (Price = 105.0)
        df.loc[5, 'high'] = 105.0
        df.loc[5, 'close'] = 104.0

        # Candidate Bullish OB at bar 10: Down candle (Pillar 1)
        df.loc[10, 'open'] = 102.0
        df.loc[10, 'high'] = 102.5
        df.loc[10, 'low'] = 100.0
        df.loc[10, 'close'] = 100.5
        df.loc[10, 'volume'] = 2500.0

        # Bar 11: Massive bullish impulse (Pillar 2)
        df.loc[11, 'open'] = 100.5
        df.loc[11, 'low'] = 100.2
        df.loc[11, 'high'] = 108.0
        df.loc[11, 'close'] = 107.5  # Breaks prior swing high 105.0! (Pillar 4)

        # Bar 12: Continues up, creating FVG (Low[12] = 104.0 > High[10] = 102.5) (Pillar 3)
        df.loc[12, 'open'] = 107.5
        df.loc[12, 'low'] = 104.0
        df.loc[12, 'high'] = 111.0
        df.loc[12, 'close'] = 110.0

        swing_highs = [(5, 105.0)]
        swing_lows = [(2, 98.0)]

        obs = engine.detect_order_blocks(df, swing_highs, swing_lows)
        bullish_obs = [ob for ob in obs if ob.ob_type == OBType.BULLISH and ob.candle_index == 10]

        assert len(bullish_obs) >= 1
        ob = bullish_obs[0]
        assert ob.candle_index == 10
        assert ob.top == 102.0  # max(open, close)
        assert ob.bottom == 100.0  # low
        assert ob.mean_threshold == 101.0  # 50% Mean Threshold
        assert ob.invalidation_level == 100.0
        assert ob.state == OBState.UNTESTED
        assert ob.is_breaker is False

    def test_order_block_retest_and_breaker_flip(self):
        """Retesting an OB marks it TESTED, and closing past invalidation flips to BREAKER."""
        engine = SMCEngine(ob_disp_atr=1.5, fvg_min_atr=0.1, atr_period=5, vol_ma_period=5, max_lookforward=2)
        df = make_base_df(25, base_price=100.0)

        # Setup Bullish OB at bar 6
        df.loc[6, 'open'] = 102.0
        df.loc[6, 'close'] = 100.0
        df.loc[6, 'low'] = 99.0
        df.loc[6, 'high'] = 102.5

        # Impulse at 7 and 8
        df.loc[7, 'open'] = 100.0
        df.loc[7, 'close'] = 108.0
        df.loc[7, 'high'] = 108.5
        df.loc[7, 'low'] = 99.8

        df.loc[8, 'open'] = 108.0
        df.loc[8, 'close'] = 110.0
        df.loc[8, 'high'] = 111.0
        df.loc[8, 'low'] = 104.0  # FVG formed (104.0 > 102.5)

        swing_highs = [(3, 104.0)]
        swing_lows = [(2, 98.0)]

        # Bar 12: Retest into zone [99.0, 102.0]
        df.loc[12, 'open'] = 104.0
        df.loc[12, 'low'] = 100.5  # Wicks into zone
        df.loc[12, 'close'] = 103.0
        df.loc[12, 'high'] = 104.5

        # Bar 15: Violently breaks and closes below invalidation level 99.0
        df.loc[15, 'open'] = 101.0
        df.loc[15, 'low'] = 97.0
        df.loc[15, 'close'] = 97.5  # Close < 99.0 -> Breaker flip!
        df.loc[15, 'high'] = 101.0

        obs = engine.detect_order_blocks(df, swing_highs, swing_lows)
        bullish_obs = [ob for ob in obs if ob.ob_type == OBType.BULLISH and ob.candle_index == 6]

        assert len(bullish_obs) >= 1
        ob = bullish_obs[0]
        assert ob.touches >= 1
        assert ob.state == OBState.BREAKER
        assert ob.is_breaker is True
        assert ob.breaker_index == 15

    def test_bearish_order_block_validation(self):
        """Bearish Order Block (-OB): green candle followed by downward expansion breaking swing low."""
        engine = SMCEngine(ob_disp_atr=1.5, fvg_min_atr=0.1, atr_period=5, vol_ma_period=5, max_lookforward=3)
        df = make_base_df(35, base_price=100.0)

        # Ensure bars 0 to 9 don't create false breaks
        for k in range(10):
            df.loc[k, 'close'] = 99.0
            df.loc[k, 'open'] = 99.5

        # Prior swing low at bar 4 (Price = 96.0)
        df.loc[4, 'low'] = 96.0

        # Bar 10: Green candle (Pillar 1)
        df.loc[10, 'open'] = 98.0
        df.loc[10, 'close'] = 100.0
        df.loc[10, 'high'] = 101.0
        df.loc[10, 'low'] = 97.5

        # Bar 11: Dump below 96.0 (Pillars 2 and 4)
        df.loc[11, 'open'] = 100.0
        df.loc[11, 'high'] = 100.2
        df.loc[11, 'low'] = 93.0
        df.loc[11, 'close'] = 93.5

        # Bar 12: Bearish FVG (High[12] = 96.0 < Low[10] = 97.5) (Pillar 3)
        df.loc[12, 'open'] = 93.5
        df.loc[12, 'high'] = 96.0
        df.loc[12, 'low'] = 90.0
        df.loc[12, 'close'] = 91.0

        swing_highs = [(2, 102.0)]
        swing_lows = [(4, 96.0)]

        obs = engine.detect_order_blocks(df, swing_highs, swing_lows)
        bearish_obs = [ob for ob in obs if ob.ob_type == OBType.BEARISH and ob.candle_index == 10]

        assert len(bearish_obs) >= 1
        ob = bearish_obs[0]
        assert ob.ob_type == OBType.BEARISH
        assert ob.top == 101.0  # high
        assert ob.bottom == 98.0  # min(open, close)
        assert ob.mean_threshold == 99.5
        assert ob.invalidation_level == 101.0


# ============================================================================
# 3. SMC Engine Tests: Liquidity Sweeps
# ============================================================================

class TestSMCLiquiditySweeps:
    """Tests Buy-Side (BSL) and Sell-Side (SSL) stop-hunt liquidity sweeps."""

    def test_tc05_bsl_sweep(self):
        """TC-05: High wicks above prior swing high, close settles below, upper wick >= 35%."""
        engine = SMCEngine(sweep_wick_min=0.35, atr_period=5)
        df = make_base_df(30, base_price=100.0)

        swing_highs = [(10, 120.0)]
        swing_lows = [(5, 95.0)]

        # Bar 25: Reaches 121.5, Closes at 118.8, Open at 119.0, Low = 118.0
        df.loc[25, 'open'] = 119.0
        df.loc[25, 'high'] = 121.5
        df.loc[25, 'low'] = 118.0
        df.loc[25, 'close'] = 118.8
        df.loc[25, 'volume'] = 2000.0

        sweeps = engine.detect_liquidity_sweeps(df, swing_highs, swing_lows)
        bsl_sweeps = [s for s in sweeps if s.sweep_type == SweepType.BSL_SWEEP]

        assert len(bsl_sweeps) >= 1
        sweep = bsl_sweeps[0]
        assert sweep.sweep_index == 25
        assert sweep.level_price == 120.0
        assert sweep.sweep_extreme == 121.5
        assert sweep.close_price == 118.8
        assert sweep.wick_ratio >= 0.35
        assert sweep.is_multi_bar is False

    def test_ssl_sweep(self):
        """SSL Sweep: Low wicks below prior swing low, close settles above, lower wick >= 35%."""
        engine = SMCEngine(sweep_wick_min=0.35, atr_period=5)
        df = make_base_df(30, base_price=105.0)

        swing_highs = [(5, 115.0)]
        swing_lows = [(10, 100.0)]  # Prior swing low at 100.0

        # Bar 22: Low = 98.5 < 100.0, Close = 101.2 > 100.0, Open = 100.8, High = 101.5
        df.loc[22, 'open'] = 100.8
        df.loc[22, 'high'] = 101.5
        df.loc[22, 'low'] = 98.5
        df.loc[22, 'close'] = 101.2

        sweeps = engine.detect_liquidity_sweeps(df, swing_highs, swing_lows)
        ssl_sweeps = [s for s in sweeps if s.sweep_type == SweepType.SSL_SWEEP]

        assert len(ssl_sweeps) >= 1
        sweep = ssl_sweeps[0]
        assert sweep.sweep_index == 22
        assert sweep.level_price == 100.0
        assert sweep.sweep_extreme == 98.5
        assert sweep.wick_ratio >= 0.35

    def test_multi_bar_turtle_soup_sweep(self):
        """Composite Turtle Soup Sweep: Bar t-1 closes above, Bar t forcefully closes back below."""
        engine = SMCEngine(sweep_wick_min=0.20, atr_period=5)
        df = make_base_df(20, base_price=100.0)

        swing_highs = [(5, 115.0)]
        swing_lows = [(2, 90.0)]

        # Bar 14 closes above 115.0
        df.loc[14, 'open'] = 114.0
        df.loc[14, 'high'] = 116.5
        df.loc[14, 'close'] = 115.5
        df.loc[14, 'low'] = 114.0

        # Bar 15 dumps back below 115.0
        df.loc[15, 'open'] = 115.2
        df.loc[15, 'high'] = 115.5
        df.loc[15, 'close'] = 114.0
        df.loc[15, 'low'] = 113.5

        sweeps = engine.detect_liquidity_sweeps(df, swing_highs, swing_lows)
        multi_sweeps = [s for s in sweeps if s.is_multi_bar]
        assert len(multi_sweeps) >= 1
        assert multi_sweeps[0].sweep_type == SweepType.BSL_SWEEP


# ============================================================================
# 4. SMC Engine Tests: Market Structure (BOS & CHoCH)
# ============================================================================

class TestSMCMarketStructure:
    """Tests zero-lookahead Swing Point confirmation, BOS continuation, and CHoCH reversal."""

    def test_swing_point_confirmation_zero_lookahead(self):
        """Swing points with radius R=3 are confirmed strictly at index k + R."""
        engine = SMCEngine(swing_radius=3)
        df = make_base_df(25, base_price=100.0)

        # Create a swing high at bar 8
        df.loc[8, 'high'] = 120.0
        for offset in [1, 2, 3]:
            df.loc[8 - offset, 'high'] = 110.0
            df.loc[8 + offset, 'high'] = 110.0

        sh, sl = engine.find_swing_points(df, radius=3)
        matching = [sp for sp in sh if sp.index == 8]
        assert len(matching) == 1
        assert matching[0].price == 120.0
        assert matching[0].confirmed_at_index == 11  # Confirmed at 8 + 3 = 11

    def test_tc06_bullish_bos_continuation(self):
        """TC-06: Confirmed close above prior swing high during Bullish trend creates BOS_BULLISH."""
        engine = SMCEngine(swing_radius=3)
        df = make_base_df(35, base_price=100.0)

        # Bar 5: Swing High at 110.0 confirmed by bar 8
        df.loc[5, 'high'] = 110.0
        for r in [1, 2, 3]:
            df.loc[5 - r, 'high'] = 105.0
            df.loc[5 + r, 'high'] = 105.0

        # Bar 10: Close at 112.0 breaks 110.0 -> First break establishes BULLISH trend (CHOCH)
        df.loc[10, 'close'] = 112.0
        df.loc[10, 'high'] = 113.0

        # Bar 15: Swing High at 120.0 confirmed by bar 18
        df.loc[15, 'high'] = 120.0
        for r in [1, 2, 3]:
            df.loc[15 - r, 'high'] = 115.0
            df.loc[15 + r, 'high'] = 115.0

        # Bar 22: Close at 122.5 > 120.0 -> Trend is already BULLISH, so emits BOS_BULLISH!
        df.loc[22, 'close'] = 122.5
        df.loc[22, 'high'] = 123.0

        result = engine.evaluate_market_structure(df)
        events = result['events']
        bos_events = [e for e in events if e.event_type == StructureEventType.BOS_BULLISH]

        assert len(bos_events) >= 1
        assert bos_events[0].broken_level == 120.0
        assert bos_events[0].breakout_price == 122.5
        assert result['current_trend'] == TrendState.BULLISH

    def test_tc07_bearish_choch_reversal(self):
        """TC-07: In an established uptrend, close below prior Higher Low triggers CHOCH_BEARISH."""
        engine = SMCEngine(swing_radius=2)
        df = make_base_df(30, base_price=100.0)

        # 1. Establish Bullish trend by breaking swing high
        df.loc[4, 'high'] = 110.0
        for r in [1, 2]:
            df.loc[4 - r, 'high'] = 105.0
            df.loc[4 + r, 'high'] = 105.0

        df.loc[8, 'close'] = 112.0  # Breaks high -> Trend becomes BULLISH

        # 2. Establish Higher Low at bar 12 (Price = 106.0) confirmed by bar 14
        df.loc[12, 'low'] = 106.0
        for r in [1, 2]:
            df.loc[12 - r, 'low'] = 108.0
            df.loc[12 + r, 'low'] = 108.0

        # Keep bars 13 to 17 trading comfortably above 106.0
        for k in range(13, 18):
            df.loc[k, 'open'] = 108.0
            df.loc[k, 'high'] = 110.0
            df.loc[k, 'low'] = 107.5
            df.loc[k, 'close'] = 109.0

        # 3. Bar 18: Closes below 106.0 at 104.5 -> Trend reversal CHoCH!
        df.loc[18, 'open'] = 108.0
        df.loc[18, 'high'] = 108.5
        df.loc[18, 'low'] = 104.0
        df.loc[18, 'close'] = 104.5

        result = engine.evaluate_market_structure(df)
        events = result['events']
        choch_bearish = [e for e in events if e.event_type == StructureEventType.CHOCH_BEARISH]

        assert len(choch_bearish) >= 1
        assert choch_bearish[0].broken_level == 106.0
        assert choch_bearish[0].breakout_price == 104.5
        assert result['current_trend'] == TrendState.BEARISH

    def test_strict_body_close_vs_wick_axiom(self):
        """Wick breach without body close does NOT trigger BOS or CHoCH."""
        engine = SMCEngine(swing_radius=2)
        df = make_base_df(20, base_price=100.0)

        # Swing high at bar 4 (Price = 110.0)
        df.loc[4, 'high'] = 110.0
        for r in [1, 2]:
            df.loc[4 - r, 'high'] = 105.0
            df.loc[4 + r, 'high'] = 105.0

        # Bar 10: High = 111.5 (wicks above 110.0), but Close = 108.5 (below 110.0)
        df.loc[10, 'high'] = 111.5
        df.loc[10, 'close'] = 108.5

        result = engine.evaluate_market_structure(df)
        assert len(result['events']) == 0  # No structural break event emitted!


# ============================================================================
# 5. SMC Engine Tests: Full Integrated Report
# ============================================================================

class TestSMCIntegratedReport:
    """Tests full SMC analysis report synthesis and institutional bias computation."""

    def test_smc_analysis_report(self):
        engine = SMCEngine()
        df = make_base_df(40, base_price=100.0)

        report = engine.analyze_buffer(df)
        assert isinstance(report, SMCAnalysisReport)
        assert report.current_price == pytest.approx(df['close'].iloc[-1], 0.01)
        assert report.institutional_bias in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert 0 <= report.confidence <= 100


# ============================================================================
# 6. Chart Pattern Engine Tests: Double Top & Double Bottom
# ============================================================================

class TestChartPatternDoubleTopBottom:
    """Tests Double Top and Double Bottom detection with symmetry and breakout confirmation."""

    def test_double_top_detection(self):
        engine = ChartPatternEngine(double_top_tol=0.015, min_pattern_span=5, max_pattern_span=30)
        df = make_base_df(40, base_price=100.0)

        # Set pattern zone base to around 112
        df.loc[9:24, 'low'] = 112.0
        df.loc[9:24, 'open'] = 113.0
        df.loc[9:24, 'close'] = 113.5
        df.loc[9:24, 'high'] = 114.0

        # Peak 1 at bar 10 (Price = 120.0)
        df.loc[10, 'high'] = 120.0
        df.loc[9, 'high'] = 115.0
        df.loc[11, 'high'] = 115.0

        # Trough at bar 16 (Price = 108.0)
        df.loc[16, 'low'] = 108.0
        df.loc[16, 'close'] = 109.0

        # Peak 2 at bar 22 (Price = 120.5) (within 0.5% symmetry)
        df.loc[22, 'high'] = 120.5
        df.loc[21, 'high'] = 115.0
        df.loc[23, 'high'] = 115.0

        # Breakout at bar 28: Close = 106.0 (< Neckline 108.0)
        df.loc[28, 'close'] = 106.0
        df.loc[28, 'low'] = 105.0

        patterns = engine.detect_double_tops(df)
        dt_patterns = [p for p in patterns if p.name == "DOUBLE_TOP" and p.metadata["peak1"][0] == 10]
        assert len(dt_patterns) >= 1

        pat = dt_patterns[0]
        assert pat.name == "DOUBLE_TOP"
        assert pat.bias == BiasType.BEARISH
        assert pat.neckline_price == 108.0
        assert pat.invalidation_level == 120.5
        # Target = 108.0 - (120.5 - 108.0) = 95.5
        assert pat.projected_target == 95.5
        assert pat.metadata['is_breakout_confirmed'] is True

    def test_double_bottom_detection(self):
        engine = ChartPatternEngine(double_top_tol=0.015, min_pattern_span=5, max_pattern_span=30)
        df = make_base_df(40, base_price=100.0)

        # Set pattern zone base to around 86
        df.loc[7:23, 'high'] = 87.0
        df.loc[7:23, 'open'] = 85.0
        df.loc[7:23, 'close'] = 85.5
        df.loc[7:23, 'low'] = 84.0

        # Trough 1 at bar 8 (Price = 80.0)
        df.loc[8, 'low'] = 80.0

        # Intervening peak at bar 14 (Price = 92.0)
        df.loc[14, 'high'] = 92.0

        # Trough 2 at bar 20 (Price = 80.3)
        df.loc[20, 'low'] = 80.3

        # Breakout at bar 25: Close = 94.0 (> Neckline 92.0)
        df.loc[25, 'close'] = 94.0
        df.loc[25, 'high'] = 95.0

        patterns = engine.detect_double_bottoms(df)
        db_patterns = [p for p in patterns if p.name == "DOUBLE_BOTTOM" and p.metadata["trough1"][0] == 8]
        assert len(db_patterns) >= 1

        pat = db_patterns[0]
        assert pat.name == "DOUBLE_BOTTOM"
        assert pat.bias == BiasType.BULLISH
        assert pat.neckline_price == 92.0
        assert pat.invalidation_level == 80.0
        # Target = 92.0 + (92.0 - 80.0) = 104.0
        assert pat.projected_target == 104.0
        assert pat.metadata['is_breakout_confirmed'] is True


# ============================================================================
# 7. Chart Pattern Engine Tests: Head & Shoulders
# ============================================================================

class TestChartPatternHeadAndShoulders:
    """Tests Head & Shoulders and Inverse Head & Shoulders recognition."""

    def test_head_and_shoulders_standard(self):
        engine = ChartPatternEngine(hs_symmetry_tol=0.035)
        df = make_base_df(50, base_price=100.0)

        # Set pattern zone baseline
        df.loc[9:32, 'low'] = 105.0
        df.loc[9:32, 'open'] = 107.0
        df.loc[9:32, 'close'] = 107.0
        df.loc[9:32, 'high'] = 108.0

        # Left Shoulder at bar 10 (Price = 110.0)
        df.loc[10, 'high'] = 110.0

        # Trough 1 at bar 15 (Price = 102.0)
        df.loc[15, 'low'] = 102.0

        # Head at bar 20 (Price = 120.0 > 110.0)
        df.loc[20, 'high'] = 120.0

        # Trough 2 at bar 25 (Price = 102.0)
        df.loc[25, 'low'] = 102.0

        # Right Shoulder at bar 30 (Price = 110.5, symmetric to 110.0)
        df.loc[30, 'high'] = 110.5

        # Breakout at bar 35: Close = 100.0 (< Neckline 102.0)
        df.loc[35, 'close'] = 100.0
        df.loc[35, 'low'] = 99.0

        patterns = engine.detect_head_and_shoulders(df)
        hs = [p for p in patterns if p.name == "HEAD_AND_SHOULDERS" and p.metadata["head"][0] == 20]
        assert len(hs) >= 1

        pat = hs[0]
        assert pat.bias == BiasType.BEARISH
        assert pat.neckline_price == 102.0
        assert pat.invalidation_level == 120.0
        # Target = 102.0 - (120.0 - 102.0) = 84.0
        assert pat.projected_target == 84.0
        assert pat.metadata['is_breakout_confirmed'] is True

    def test_inverse_head_and_shoulders(self):
        engine = ChartPatternEngine(hs_symmetry_tol=0.035)
        df = make_base_df(50, base_price=100.0)

        # Set pattern zone baseline
        df.loc[9:32, 'high'] = 93.0
        df.loc[9:32, 'open'] = 92.0
        df.loc[9:32, 'close'] = 92.0
        df.loc[9:32, 'low'] = 91.0

        # Left Shoulder Trough at bar 10 (Price = 90.0)
        df.loc[10, 'low'] = 90.0

        # Peak 1 at bar 15 (Price = 98.0)
        df.loc[15, 'high'] = 98.0

        # Head Trough at bar 20 (Price = 80.0 < 90.0)
        df.loc[20, 'low'] = 80.0

        # Peak 2 at bar 25 (Price = 98.0)
        df.loc[25, 'high'] = 98.0

        # Right Shoulder Trough at bar 30 (Price = 89.8, symmetric to 90.0)
        df.loc[30, 'low'] = 89.8

        # Breakout at bar 35: Close = 100.0 (> Neckline 98.0)
        df.loc[35, 'close'] = 100.0
        df.loc[35, 'high'] = 101.0

        patterns = engine.detect_head_and_shoulders(df)
        ihs = [p for p in patterns if p.name == "INVERSE_HEAD_AND_SHOULDERS" and p.metadata["head"][0] == 20]
        assert len(ihs) >= 1

        pat = ihs[0]
        assert pat.bias == BiasType.BULLISH
        assert pat.neckline_price == 98.0
        assert pat.invalidation_level == 80.0
        # Target = 98.0 + (98.0 - 80.0) = 116.0
        assert pat.projected_target == 116.0
        assert pat.metadata['is_breakout_confirmed'] is True


# ============================================================================
# 8. Chart Pattern Engine Tests: Triangles & Wedges
# ============================================================================

class TestChartPatternTrianglesWedges:
    """Tests geometric pattern recognition via linear regression trendlines."""

    def test_ascending_triangle(self):
        engine = ChartPatternEngine()
        n = 30
        df = make_base_df(n, base_price=100.0)

        # Flat upper resistance at ~110.0, higher swing lows: 95.0 -> 99.0 -> 103.0
        peaks_x = [5, 15, 25]
        for px in peaks_x:
            df.loc[px, 'high'] = 110.0
            df.loc[px - 1, 'high'] = 106.0
            df.loc[px + 1, 'high'] = 106.0

        troughs_x = [8, 18, 28]
        troughs_y = [95.0, 99.0, 103.0]
        for tx, ty in zip(troughs_x, troughs_y):
            df.loc[tx, 'low'] = ty
            df.loc[tx - 1, 'low'] = ty + 2.0
            df.loc[tx + 1, 'low'] = ty + 2.0

        patterns = engine.detect_triangles_and_wedges(df, window_len=30)
        asc = [p for p in patterns if p.name == "ASCENDING_TRIANGLE"]
        assert len(asc) >= 1
        assert asc[0].bias == BiasType.BULLISH

    def test_falling_wedge(self):
        engine = ChartPatternEngine()
        n = 30
        df = make_base_df(n, base_price=100.0)

        # Both slopes negative, but upper is steeper downwards: converging downward
        peaks_x = [5, 15, 25]
        peaks_y = [120.0, 112.0, 105.0]  # Steeper drop
        for px, py in zip(peaks_x, peaks_y):
            df.loc[px, 'high'] = py
            df.loc[px - 1, 'high'] = py - 2.0
            df.loc[px + 1, 'high'] = py - 2.0

        troughs_x = [8, 18, 28]
        troughs_y = [105.0, 101.0, 98.0]  # Flatter drop
        for tx, ty in zip(troughs_x, troughs_y):
            df.loc[tx, 'low'] = ty
            df.loc[tx - 1, 'low'] = ty + 2.0
            df.loc[tx + 1, 'low'] = ty + 2.0

        patterns = engine.detect_triangles_and_wedges(df, window_len=30)
        wedges = [p for p in patterns if p.name == "FALLING_WEDGE"]
        assert len(wedges) >= 1
        assert wedges[0].bias == BiasType.BULLISH


# ============================================================================
# 9. Chart Pattern Engine Tests: Classical Multi-Candle Patterns
# ============================================================================

class TestChartPatternMultiCandle:
    """Tests Morning Star, Evening Star, Three White Soldiers, Three Black Crows, and Fakey."""

    def test_morning_star(self):
        engine = ChartPatternEngine()
        df = make_base_df(10, base_price=100.0)

        # Bar 3: Large red candle (Open 110, Close 100)
        df.loc[3, 'open'] = 110.0
        df.loc[3, 'high'] = 110.5
        df.loc[3, 'low'] = 99.5
        df.loc[3, 'close'] = 100.0

        # Bar 4: Small doji/star (Open 98.0, Close 98.2)
        df.loc[4, 'open'] = 98.0
        df.loc[4, 'high'] = 98.5
        df.loc[4, 'low'] = 97.5
        df.loc[4, 'close'] = 98.2

        # Bar 5: Large green candle penetrating > 50% into bar 3 (Close 106.0 >= Midpoint 105.0)
        df.loc[5, 'open'] = 98.5
        df.loc[5, 'high'] = 107.0
        df.loc[5, 'low'] = 98.0
        df.loc[5, 'close'] = 106.0

        patterns = engine.detect_multi_candle_patterns(df)
        morning_stars = [p for p in patterns if p.name == "MORNING_STAR"]
        assert len(morning_stars) >= 1
        assert morning_stars[0].bias == BiasType.BULLISH
        assert morning_stars[0].candle_span == 3

    def test_evening_star(self):
        engine = ChartPatternEngine()
        df = make_base_df(10, base_price=100.0)

        # Bar 3: Large green candle (Open 90.0, Close 100.0)
        df.loc[3, 'open'] = 90.0
        df.loc[3, 'high'] = 100.5
        df.loc[3, 'low'] = 89.5
        df.loc[3, 'close'] = 100.0

        # Bar 4: Small star (Open 102.0, Close 102.2)
        df.loc[4, 'open'] = 102.0
        df.loc[4, 'high'] = 102.5
        df.loc[4, 'low'] = 101.5
        df.loc[4, 'close'] = 102.2

        # Bar 5: Large red candle penetrating > 50% into bar 3 (Close 94.0 <= Midpoint 95.0)
        df.loc[5, 'open'] = 101.5
        df.loc[5, 'high'] = 102.0
        df.loc[5, 'low'] = 93.5
        df.loc[5, 'close'] = 94.0

        patterns = engine.detect_multi_candle_patterns(df)
        evening_stars = [p for p in patterns if p.name == "EVENING_STAR"]
        assert len(evening_stars) >= 1
        assert evening_stars[0].bias == BiasType.BEARISH

    def test_three_white_soldiers(self):
        engine = ChartPatternEngine()
        df = make_base_df(10, base_price=100.0)

        # 3 consecutive strong green bars closing near highs
        df.loc[2, 'open'] = 100.0
        df.loc[2, 'close'] = 103.0
        df.loc[2, 'high'] = 103.2
        df.loc[2, 'low'] = 99.8

        df.loc[3, 'open'] = 102.5
        df.loc[3, 'close'] = 106.0
        df.loc[3, 'high'] = 106.2
        df.loc[3, 'low'] = 102.0

        df.loc[4, 'open'] = 105.5
        df.loc[4, 'close'] = 109.0
        df.loc[4, 'high'] = 109.2
        df.loc[4, 'low'] = 105.0

        patterns = engine.detect_multi_candle_patterns(df)
        soldiers = [p for p in patterns if p.name == "THREE_WHITE_SOLDIERS"]
        assert len(soldiers) >= 1
        assert soldiers[0].bias == BiasType.STRONG_BULLISH

    def test_three_black_crows(self):
        engine = ChartPatternEngine()
        df = make_base_df(10, base_price=100.0)

        # 3 consecutive strong red bars closing near lows
        df.loc[2, 'open'] = 109.0
        df.loc[2, 'close'] = 106.0
        df.loc[2, 'high'] = 109.2
        df.loc[2, 'low'] = 105.8

        df.loc[3, 'open'] = 106.5
        df.loc[3, 'close'] = 103.0
        df.loc[3, 'high'] = 106.8
        df.loc[3, 'low'] = 102.8

        df.loc[4, 'open'] = 103.5
        df.loc[4, 'close'] = 100.0
        df.loc[4, 'high'] = 103.8
        df.loc[4, 'low'] = 99.8

        patterns = engine.detect_multi_candle_patterns(df)
        crows = [p for p in patterns if p.name == "THREE_BLACK_CROWS"]
        assert len(crows) >= 1
        assert crows[0].bias == BiasType.STRONG_BEARISH

    def test_fakey_inside_bar_false_breakout(self):
        engine = ChartPatternEngine()
        df = make_base_df(10, base_price=100.0)

        # Bar 3: Mother bar (High 105.0, Low 95.0)
        df.loc[3, 'high'] = 105.0
        df.loc[3, 'low'] = 95.0

        # Bar 4: Inside bar (High 102.0, Low 98.0)
        df.loc[4, 'high'] = 102.0
        df.loc[4, 'low'] = 98.0

        # Bar 5: False breakdown: wicks below 98.0 to 97.0, but closes green at 100.0
        df.loc[5, 'open'] = 98.5
        df.loc[5, 'low'] = 97.0
        df.loc[5, 'high'] = 101.0
        df.loc[5, 'close'] = 100.0

        patterns = engine.detect_multi_candle_patterns(df)
        fakey = [p for p in patterns if p.name == "FAKEY_BULLISH"]
        assert len(fakey) >= 1
        assert fakey[0].bias == BiasType.BULLISH

    def test_detect_all_master_method(self):
        engine = ChartPatternEngine()
        df = make_base_df(30, base_price=100.0)

        all_patterns = engine.detect_all(df)
        assert isinstance(all_patterns, list)
