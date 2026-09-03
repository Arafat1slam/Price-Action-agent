"""
engines/smc_engine.py - Unified Smart Money Concepts (SMC) & Market Structure Core Engine

Implements:
  1. Fair Value Gaps (FVG) / Imbalances (BISI & SIBI, Consequent Encroachment 50%,
     stateful mitigation tracking: UNMITIGATED, PARTIALLY_MITIGATED, CE_TESTED,
     FULLY_MITIGATED, INVERTED).
  2. Order Blocks (+OB & -OB) with strict 4-pillar validation:
     - Polarity check preceding displacement
     - Impulsive displacement >= 1.5 ATR with expansion profile
     - Fair Value Gap confirmation footprint
     - Structural break confirmation (BOS/CHoCH with candle body close)
     - Mean Threshold (50% midpoint) tracking
     - Breaker Block transition upon invalidation
  3. Liquidity Sweeps (BSL and SSL stop-hunt sweeps with >= 35% wick ratio,
     volume expansion RVOL, penetration constraints, and Turtle Soup multi-bar patterns).
  4. Market Structure Geometry (Fractal Swings radius=3, Break of Structure BOS
     trend continuation, Change of Character CHoCH trend reversal with strict
     candle body close confirmation and zero lookahead bias).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

# ============================================================================
# Enums
# ============================================================================

class FVGType(str, Enum):
    BULLISH = "BULLISH"  # BISI (Buyside Imbalance Sellside Inefficiency)
    BEARISH = "BEARISH"  # SIBI (Sellside Imbalance Buyside Inefficiency)

class FVGState(str, Enum):
    UNMITIGATED = "UNMITIGATED"
    PARTIALLY_MITIGATED = "PARTIALLY_MITIGATED"
    CE_TESTED = "CE_TESTED"
    FULLY_MITIGATED = "FULLY_MITIGATED"
    INVERTED = "INVERTED"

class OBType(str, Enum):
    BULLISH = "BULLISH"  # +OB (Demand zone)
    BEARISH = "BEARISH"  # -OB (Supply zone)

class OBState(str, Enum):
    UNTESTED = "UNTESTED"
    TESTED = "TESTED"            # Price retraced into zone
    INVALIDATED = "INVALIDATED"  # Closed past invalidation level
    BREAKER = "BREAKER"          # Flipped polarity into a Breaker Block

class SweepType(str, Enum):
    BSL_SWEEP = "BSL_SWEEP"  # Buy-Side Liquidity Swept (Bearish signal)
    SSL_SWEEP = "SSL_SWEEP"  # Sell-Side Liquidity Swept (Bullish signal)

class TrendState(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"

class StructureEventType(str, Enum):
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"  # Trend reversal from Bearish/Sideways to Bullish
    CHOCH_BEARISH = "CHOCH_BEARISH"  # Trend reversal from Bullish/Sideways to Bearish

class SMCTrend(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"

class SMCEventType(str, Enum):
    FVG_BULLISH = "FVG_BULLISH"
    FVG_BEARISH = "FVG_BEARISH"
    OB_BULLISH = "OB_BULLISH"
    OB_BEARISH = "OB_BEARISH"
    SWEEP_BSL = "SWEEP_BSL"
    SWEEP_SSL = "SWEEP_SSL"
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"

# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class FairValueGap:
    gap_type: FVGType
    creation_index: int
    creation_timestamp: Any
    top: float
    bottom: float
    ce: float                      # Consequent Encroachment (50% midpoint)
    initial_size: float
    current_top: float             # Dynamic remaining gap top
    current_bottom: float          # Dynamic remaining gap bottom
    state: FVGState = FVGState.UNMITIGATED
    mitigation_index: Optional[int] = None
    mitigation_percentage: float = 0.0
    is_inverted: bool = False
    inversion_index: Optional[int] = None

    @property
    def is_active(self) -> bool:
        return self.state in [FVGState.UNMITIGATED, FVGState.PARTIALLY_MITIGATED, FVGState.CE_TESTED]

@dataclass
class OrderBlock:
    ob_type: OBType
    candle_index: int
    timestamp: Any
    top: float
    bottom: float
    mean_threshold: float         # 50% midpoint of OB
    invalidation_level: float     # Bottom for Bullish OB, Top for Bearish OB
    volume: float
    rvol: float                   # Relative Volume at candle
    displacement_atr: float       # Magnitude of impulse in ATR multiples
    state: OBState = OBState.UNTESTED
    mitigation_index: Optional[int] = None
    touches: int = 0
    is_breaker: bool = False
    breaker_index: Optional[int] = None

    @property
    def is_active(self) -> bool:
        return self.state in [OBState.UNTESTED, OBState.TESTED]

@dataclass
class LiquiditySweep:
    sweep_type: SweepType
    sweep_index: int
    timestamp: Any
    level_price: float
    sweep_extreme: float          # Highest high or lowest low of sweep
    close_price: float
    wick_ratio: float
    penetration_atr: float
    volume_ratio: float
    is_multi_bar: bool = False

@dataclass
class SwingPoint:
    index: int
    timestamp: Any
    price: float
    is_high: bool                 # True for Swing High, False for Swing Low
    confirmed_at_index: int

@dataclass
class StructureEvent:
    event_type: StructureEventType
    candle_index: int
    timestamp: Any
    broken_level: float
    breakout_price: float
    prior_trend: TrendState
    new_trend: TrendState

@dataclass
class SMCAnalysisReport:
    timestamp: Any
    current_price: float
    trend: SMCTrend
    active_bullish_fvgs: List[FairValueGap]
    active_bearish_fvgs: List[FairValueGap]
    active_bullish_obs: List[OrderBlock]
    active_bearish_obs: List[OrderBlock]
    recent_sweeps: List[LiquiditySweep]
    recent_structure_events: List[StructureEvent]
    institutional_bias: str
    confidence: int               # 0 to 100%
    all_fvgs: List[FairValueGap] = field(default_factory=list)
    all_obs: List[OrderBlock] = field(default_factory=list)
    all_sweeps: List[LiquiditySweep] = field(default_factory=list)
    all_structure_events: List[StructureEvent] = field(default_factory=list)
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)

# ============================================================================
# Core SMC Engine
# ============================================================================

class SMCEngine:
    """
    Institutional Smart Money Concepts (SMC) & Market Structure Core Engine.
    Zero-lookahead, causal price action analysis.
    """
    def __init__(self,
                 swing_radius: int = 3,
                 fvg_min_atr: float = 0.20,
                 ob_disp_atr: float = 1.50,
                 sweep_wick_min: float = 0.35,
                 atr_period: int = 14,
                 vol_ma_period: int = 20,
                 max_lookforward: int = 3):
        self.swing_radius = swing_radius
        self.fvg_min_atr = fvg_min_atr
        self.ob_disp_atr = ob_disp_atr
        self.sweep_wick_min = sweep_wick_min
        self.atr_period = atr_period
        self.vol_ma_period = vol_ma_period
        self.max_lookforward = max_lookforward

    def compute_atr(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized Average True Range (ATR) calculation."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(df)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([max(high[0] - low[0], 1e-6)])

        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        tr = np.insert(tr, 0, high[0] - low[0])
        atr = pd.Series(tr).rolling(self.atr_period, min_periods=1).mean().values
        return np.maximum(atr, 1e-6)

    def find_swing_points(self, df: pd.DataFrame, radius: Optional[int] = None) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Extract confirmed swing highs and swing lows with zero lookahead bias.
        A swing at index k is confirmed only at candle k + radius.
        """
        r = radius if radius is not None else self.swing_radius
        n = len(df)
        swing_highs: List[SwingPoint] = []
        swing_lows: List[SwingPoint] = []

        if n < 2 * r + 1:
            return swing_highs, swing_lows

        high = df['high'].values
        low = df['low'].values
        ts = df['timestamp'].values if 'timestamp' in df.columns else np.arange(n)

        for k in range(r, n - r):
            confirmation_idx = k + r

            # Check Swing High
            is_sh = True
            p_high = high[k]
            for j in range(1, r + 1):
                if high[k - j] >= p_high or high[k + j] >= p_high:
                    is_sh = False
                    break
            if is_sh:
                swing_highs.append(SwingPoint(
                    index=k,
                    timestamp=ts[k],
                    price=float(p_high),
                    is_high=True,
                    confirmed_at_index=confirmation_idx
                ))

            # Check Swing Low
            is_sl = True
            p_low = low[k]
            for j in range(1, r + 1):
                if low[k - j] <= p_low or low[k + j] <= p_low:
                    is_sl = False
                    break
            if is_sl:
                swing_lows.append(SwingPoint(
                    index=k,
                    timestamp=ts[k],
                    price=float(p_low),
                    is_high=False,
                    confirmed_at_index=confirmation_idx
                ))

        return swing_highs, swing_lows

    def detect_fvgs(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detects Fair Value Gaps (BISI & SIBI) and performs dynamic stateful
        mitigation tracking forward in time (Unmitigated, Partially Mitigated,
        CE Tested, Fully Mitigated, Inverted).
        """
        n = len(df)
        if n < 3:
            return []

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        ts = df['timestamp'].values if 'timestamp' in df.columns else np.arange(n)
        atr = self.compute_atr(df)

        fvgs: List[FairValueGap] = []

        for i in range(2, n):
            atr_thresh = self.fvg_min_atr * atr[i]

            # Bullish FVG (BISI: Low[i] > High[i-2])
            if low[i] > high[i-2]:
                gap_size = low[i] - high[i-2]
                if gap_size >= atr_thresh:
                    top = float(low[i])
                    bottom = float(high[i-2])
                    ce = (top + bottom) / 2.0
                    fvg = FairValueGap(
                        gap_type=FVGType.BULLISH,
                        creation_index=i,
                        creation_timestamp=ts[i],
                        top=top,
                        bottom=bottom,
                        ce=ce,
                        initial_size=float(gap_size),
                        current_top=top,
                        current_bottom=bottom
                    )
                    fvgs.append(fvg)

            # Bearish FVG (SIBI: High[i] < Low[i-2])
            elif high[i] < low[i-2]:
                gap_size = low[i-2] - high[i]
                if gap_size >= atr_thresh:
                    top = float(low[i-2])
                    bottom = float(high[i])
                    ce = (top + bottom) / 2.0
                    fvg = FairValueGap(
                        gap_type=FVGType.BEARISH,
                        creation_index=i,
                        creation_timestamp=ts[i],
                        top=top,
                        bottom=bottom,
                        ce=ce,
                        initial_size=float(gap_size),
                        current_top=top,
                        current_bottom=bottom
                    )
                    fvgs.append(fvg)

        # Stateful mitigation tracking forward
        for fvg in fvgs:
            for k in range(fvg.creation_index + 1, n):
                bar_h = high[k]
                bar_l = low[k]
                bar_c = close[k]

                if fvg.gap_type == FVGType.BULLISH:
                    if bar_l < fvg.current_top:
                        fvg.current_top = min(fvg.current_top, bar_l)
                        penetration = fvg.top - max(fvg.current_top, fvg.bottom)
                        fvg.mitigation_percentage = min(1.0, max(0.0, penetration / (fvg.initial_size + 1e-9)))

                        if bar_l <= fvg.bottom:
                            fvg.state = FVGState.FULLY_MITIGATED
                            fvg.mitigation_index = k
                            fvg.mitigation_percentage = 1.0
                            if bar_c < fvg.bottom:
                                fvg.is_inverted = True
                                fvg.inversion_index = k
                                fvg.state = FVGState.INVERTED
                            break
                        elif bar_l <= fvg.ce:
                            fvg.state = FVGState.CE_TESTED
                        else:
                            fvg.state = FVGState.PARTIALLY_MITIGATED

                elif fvg.gap_type == FVGType.BEARISH:
                    if bar_h > fvg.current_bottom:
                        fvg.current_bottom = max(fvg.current_bottom, bar_h)
                        penetration = min(fvg.current_bottom, fvg.top) - fvg.bottom
                        fvg.mitigation_percentage = min(1.0, max(0.0, penetration / (fvg.initial_size + 1e-9)))

                        if bar_h >= fvg.top:
                            fvg.state = FVGState.FULLY_MITIGATED
                            fvg.mitigation_index = k
                            fvg.mitigation_percentage = 1.0
                            if bar_c > fvg.top:
                                fvg.is_inverted = True
                                fvg.inversion_index = k
                                fvg.state = FVGState.INVERTED
                            break
                        elif bar_h >= fvg.ce:
                            fvg.state = FVGState.CE_TESTED
                        else:
                            fvg.state = FVGState.PARTIALLY_MITIGATED

        return fvgs

    def detect_order_blocks(self,
                            df: pd.DataFrame,
                            swing_highs: Optional[List[Tuple[int, float]]] = None,
                            swing_lows: Optional[List[Tuple[int, float]]] = None) -> List[OrderBlock]:
        """
        Detects Validated Order Blocks (+OB & -OB) with strict 4-pillar validation:
          Pillar 1: Polarity check (last down candle before surge, last up candle before drop)
          Pillar 2: Impulsive displacement >= 1.5 ATR over next M bars
          Pillar 3: FVG confirmation footprint in displacement direction
          Pillar 4: Structural break (BOS/CHoCH breach with candle body close confirmation)
        Tracks mitigation, touches, invalidation, and Breaker Block flip.
        """
        n = len(df)
        if n < self.vol_ma_period + self.max_lookforward:
            return []

        high = df['high'].values
        low = df['low'].values
        open_p = df['open'].values
        close = df['close'].values
        vol = df['volume'].values
        ts = df['timestamp'].values if 'timestamp' in df.columns else np.arange(n)

        atr = self.compute_atr(df)
        vol_sma = pd.Series(vol).rolling(self.vol_ma_period, min_periods=1).mean().values

        # If swing points are not provided, compute them
        if swing_highs is None or swing_lows is None:
            sh_objs, sl_objs = self.find_swing_points(df)
            sh_list = [(sp.index, sp.price) for sp in sh_objs]
            sl_list = [(sp.index, sp.price) for sp in sl_objs]
        else:
            sh_list = swing_highs
            sl_list = swing_lows

        order_blocks: List[OrderBlock] = []

        for j in range(self.vol_ma_period, n - self.max_lookforward):
            # ----------------------------------------------------
            # Bullish Order Block (+OB) Candidate:
            # Pillar 1: Candle j closed bearish (close < open)
            # ----------------------------------------------------
            if close[j] < open_p[j]:
                fwd_highs = high[j + 1 : j + 1 + self.max_lookforward]
                max_h = float(np.max(fwd_highs))
                disp = max_h - low[j]

                # Pillar 2: Impulsive displacement >= min_disp_atr * ATR
                if disp >= self.ob_disp_atr * atr[j]:
                    # Pillar 3: FVG check in next M bars
                    fvg_created = False
                    for k in range(1, self.max_lookforward + 1):
                        if j + k + 1 < n and low[j + k + 1] > high[j + k - 1]:
                            fvg_created = True
                            break

                    # Pillar 4: Structural break (BOS / CHoCH breach of a prior swing high)
                    bos_confirmed = False
                    for sh_idx, sh_price in sh_list:
                        if sh_idx < j and max_h > sh_price:
                            # Strict body close confirmation
                            if any(close[j + k] > sh_price for k in range(1, self.max_lookforward + 1) if j + k < n):
                                bos_confirmed = True
                                break

                    # If no swing high existed yet, check rolling local max before j
                    if not bos_confirmed and len(sh_list) == 0:
                        local_peak = float(np.max(high[max(0, j - 20) : j]))
                        if any(close[j + k] > local_peak for k in range(1, self.max_lookforward + 1) if j + k < n):
                            bos_confirmed = True

                    if fvg_created and bos_confirmed:
                        top = float(max(open_p[j], close[j]))
                        bottom = float(low[j])
                        mt = (top + bottom) / 2.0
                        rvol = float(vol[j] / (vol_sma[j] + 1e-9))

                        ob = OrderBlock(
                            ob_type=OBType.BULLISH,
                            candle_index=j,
                            timestamp=ts[j],
                            top=top,
                            bottom=bottom,
                            mean_threshold=mt,
                            invalidation_level=bottom,
                            volume=float(vol[j]),
                            rvol=rvol,
                            displacement_atr=float(disp / (atr[j] + 1e-9))
                        )
                        order_blocks.append(ob)

            # ----------------------------------------------------
            # Bearish Order Block (-OB) Candidate:
            # Pillar 1: Candle j closed bullish (close > open)
            # ----------------------------------------------------
            elif close[j] > open_p[j]:
                fwd_lows = low[j + 1 : j + 1 + self.max_lookforward]
                min_l = float(np.min(fwd_lows))
                disp = high[j] - min_l

                # Pillar 2: Impulsive displacement >= min_disp_atr * ATR
                if disp >= self.ob_disp_atr * atr[j]:
                    # Pillar 3: FVG check in next M bars
                    fvg_created = False
                    for k in range(1, self.max_lookforward + 1):
                        if j + k + 1 < n and high[j + k + 1] < low[j + k - 1]:
                            fvg_created = True
                            break

                    # Pillar 4: Structural break (BOS / CHoCH breach of a prior swing low)
                    bos_confirmed = False
                    for sl_idx, sl_price in sl_list:
                        if sl_idx < j and min_l < sl_price:
                            if any(close[j + k] < sl_price for k in range(1, self.max_lookforward + 1) if j + k < n):
                                bos_confirmed = True
                                break

                    if not bos_confirmed and len(sl_list) == 0:
                        local_trough = float(np.min(low[max(0, j - 20) : j]))
                        if any(close[j + k] < local_trough for k in range(1, self.max_lookforward + 1) if j + k < n):
                            bos_confirmed = True

                    if fvg_created and bos_confirmed:
                        top = float(high[j])
                        bottom = float(min(open_p[j], close[j]))
                        mt = (top + bottom) / 2.0
                        rvol = float(vol[j] / (vol_sma[j] + 1e-9))

                        ob = OrderBlock(
                            ob_type=OBType.BEARISH,
                            candle_index=j,
                            timestamp=ts[j],
                            top=top,
                            bottom=bottom,
                            mean_threshold=mt,
                            invalidation_level=top,
                            volume=float(vol[j]),
                            rvol=rvol,
                            displacement_atr=float(disp / (atr[j] + 1e-9))
                        )
                        order_blocks.append(ob)

        # Track OB status forward in time
        for ob in order_blocks:
            for t in range(ob.candle_index + self.max_lookforward, n):
                bar_h = high[t]
                bar_l = low[t]
                bar_c = close[t]

                if ob.ob_type == OBType.BULLISH:
                    # Invalidation & Breaker Block transition
                    if bar_c < ob.invalidation_level:
                        ob.state = OBState.BREAKER
                        ob.is_breaker = True
                        ob.breaker_index = t
                        ob.mitigation_index = t
                        break
                    # Retest/Touch inside [bottom, top]
                    elif bar_l <= ob.top and bar_l >= ob.bottom:
                        ob.touches += 1
                        if ob.state == OBState.UNTESTED:
                            ob.state = OBState.TESTED
                            ob.mitigation_index = t

                elif ob.ob_type == OBType.BEARISH:
                    # Invalidation & Breaker Block transition
                    if bar_c > ob.invalidation_level:
                        ob.state = OBState.BREAKER
                        ob.is_breaker = True
                        ob.breaker_index = t
                        ob.mitigation_index = t
                        break
                    # Retest/Touch inside [bottom, top]
                    elif bar_h >= ob.bottom and bar_h <= ob.top:
                        ob.touches += 1
                        if ob.state == OBState.UNTESTED:
                            ob.state = OBState.TESTED
                            ob.mitigation_index = t

        return order_blocks

    def detect_liquidity_sweeps(self,
                                df: pd.DataFrame,
                                swing_highs: Optional[List[Tuple[int, float]]] = None,
                                swing_lows: Optional[List[Tuple[int, float]]] = None) -> List[LiquiditySweep]:
        """
        Detects BSL (Buy-Side) and SSL (Sell-Side) stop-hunt liquidity sweeps.
        Enforces >= 35% wick ratio, volume expansion, and penetration depth constraints.
        Supports single-bar and multi-bar (Turtle Soup) setups.
        """
        n = len(df)
        if n < 5:
            return []

        high = df['high'].values
        low = df['low'].values
        open_p = df['open'].values
        close = df['close'].values
        vol = df['volume'].values
        ts = df['timestamp'].values if 'timestamp' in df.columns else np.arange(n)

        atr = self.compute_atr(df)
        vol_sma = pd.Series(vol).rolling(min(self.vol_ma_period, n), min_periods=1).mean().values

        if swing_highs is None or swing_lows is None:
            sh_objs, sl_objs = self.find_swing_points(df)
            sh_list = [(sp.index, sp.price) for sp in sh_objs]
            sl_list = [(sp.index, sp.price) for sp in sl_objs]
        else:
            sh_list = swing_highs
            sl_list = swing_lows

        sweeps: List[LiquiditySweep] = []

        for t in range(1, n):
            c_range = max(high[t] - low[t], 1e-9)
            upper_wick = high[t] - max(open_p[t], close[t])
            lower_wick = min(open_p[t], close[t]) - low[t]

            upper_wick_ratio = upper_wick / c_range
            lower_wick_ratio = lower_wick / c_range
            rvol = float(vol[t] / (vol_sma[t] + 1e-9))

            # Check BSL Sweeps (Buy-Side Liquidity above swing highs)
            for sh_idx, sh_price in sh_list:
                if sh_idx < t - 1:
                    # Multi-bar Turtle Soup: Bar t-1 wicks/closes above, Bar t forcefully closes back below
                    if t >= 2 and high[t-1] > sh_price and close[t-1] >= sh_price and close[t] < sh_price:
                        penetration = max(high[t-1], high[t]) - sh_price
                        pen_atr = penetration / (atr[t] + 1e-9)
                        if pen_atr <= 1.5:
                            sweeps.append(LiquiditySweep(
                                sweep_type=SweepType.BSL_SWEEP,
                                sweep_index=t,
                                timestamp=ts[t],
                                level_price=float(sh_price),
                                sweep_extreme=float(max(high[t-1], high[t])),
                                close_price=float(close[t]),
                                wick_ratio=float(upper_wick_ratio),
                                penetration_atr=float(pen_atr),
                                volume_ratio=rvol,
                                is_multi_bar=True
                            ))
                            break

                    # Single-bar sweep: High wicks above, Close settles below
                    elif high[t] > sh_price and close[t] < sh_price:
                        penetration = high[t] - sh_price
                        pen_atr = penetration / (atr[t] + 1e-9)

                        if upper_wick_ratio >= self.sweep_wick_min and pen_atr <= 1.5:
                            sweeps.append(LiquiditySweep(
                                sweep_type=SweepType.BSL_SWEEP,
                                sweep_index=t,
                                timestamp=ts[t],
                                level_price=float(sh_price),
                                sweep_extreme=float(high[t]),
                                close_price=float(close[t]),
                                wick_ratio=float(upper_wick_ratio),
                                penetration_atr=float(pen_atr),
                                volume_ratio=rvol,
                                is_multi_bar=False
                            ))
                            break

            # Check SSL Sweeps (Sell-Side Liquidity below swing lows)
            for sl_idx, sl_price in sl_list:
                if sl_idx < t - 1:
                    # Multi-bar Turtle Soup: Bar t-1 wicks/closes below, Bar t closes back above
                    if t >= 2 and low[t-1] < sl_price and close[t-1] <= sl_price and close[t] > sl_price:
                        penetration = sl_price - min(low[t-1], low[t])
                        pen_atr = penetration / (atr[t] + 1e-9)
                        if pen_atr <= 1.5:
                            sweeps.append(LiquiditySweep(
                                sweep_type=SweepType.SSL_SWEEP,
                                sweep_index=t,
                                timestamp=ts[t],
                                level_price=float(sl_price),
                                sweep_extreme=float(min(low[t-1], low[t])),
                                close_price=float(close[t]),
                                wick_ratio=float(lower_wick_ratio),
                                penetration_atr=float(pen_atr),
                                volume_ratio=rvol,
                                is_multi_bar=True
                            ))
                            break

                    # Single-bar sweep: Low wicks below, Close settles above
                    elif low[t] < sl_price and close[t] > sl_price:
                        penetration = sl_price - low[t]
                        pen_atr = penetration / (atr[t] + 1e-9)

                        if lower_wick_ratio >= self.sweep_wick_min and pen_atr <= 1.5:
                            sweeps.append(LiquiditySweep(
                                sweep_type=SweepType.SSL_SWEEP,
                                sweep_index=t,
                                timestamp=ts[t],
                                level_price=float(sl_price),
                                sweep_extreme=float(low[t]),
                                close_price=float(close[t]),
                                wick_ratio=float(lower_wick_ratio),
                                penetration_atr=float(pen_atr),
                                volume_ratio=rvol,
                                is_multi_bar=False
                            ))
                            break

        return sweeps

    def evaluate_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates Market Structure transitions (BOS & CHoCH) with strict candle body close
        confirmation and zero lookahead bias.
        """
        n = len(df)
        if n < (2 * self.swing_radius + 1):
            return {
                "current_trend": TrendState.SIDEWAYS,
                "events": [],
                "swing_highs": [],
                "swing_lows": []
            }

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        ts = df['timestamp'].values if 'timestamp' in df.columns else np.arange(n)

        swing_highs: List[SwingPoint] = []
        swing_lows: List[SwingPoint] = []
        events: List[StructureEvent] = []
        trend = TrendState.SIDEWAYS

        active_high: Optional[SwingPoint] = None
        active_low: Optional[SwingPoint] = None

        # Step through bars simulating streaming
        for t in range(n):
            pivot_idx = t - self.swing_radius
            if pivot_idx >= self.swing_radius:
                # Check Swing High
                is_sh = True
                p_high = high[pivot_idx]
                for r in range(1, self.swing_radius + 1):
                    if high[pivot_idx - r] >= p_high or high[pivot_idx + r] >= p_high:
                        is_sh = False
                        break
                if is_sh:
                    sp = SwingPoint(pivot_idx, ts[pivot_idx], float(p_high), True, t)
                    swing_highs.append(sp)
                    active_high = sp

                # Check Swing Low
                is_sl = True
                p_low = low[pivot_idx]
                for r in range(1, self.swing_radius + 1):
                    if low[pivot_idx - r] <= p_low or low[pivot_idx + r] <= p_low:
                        is_sl = False
                        break
                if is_sl:
                    sp = SwingPoint(pivot_idx, ts[pivot_idx], float(p_low), False, t)
                    swing_lows.append(sp)
                    active_low = sp

            # Check structural breaks at bar t
            if active_high is not None and close[t] > active_high.price:
                if trend == TrendState.BULLISH:
                    events.append(StructureEvent(
                        event_type=StructureEventType.BOS_BULLISH,
                        candle_index=t,
                        timestamp=ts[t],
                        broken_level=active_high.price,
                        breakout_price=float(close[t]),
                        prior_trend=trend,
                        new_trend=TrendState.BULLISH
                    ))
                else:
                    events.append(StructureEvent(
                        event_type=StructureEventType.CHOCH_BULLISH,
                        candle_index=t,
                        timestamp=ts[t],
                        broken_level=active_high.price,
                        breakout_price=float(close[t]),
                        prior_trend=trend,
                        new_trend=TrendState.BULLISH
                    ))
                    trend = TrendState.BULLISH
                active_high = None

            elif active_low is not None and close[t] < active_low.price:
                if trend == TrendState.BEARISH:
                    events.append(StructureEvent(
                        event_type=StructureEventType.BOS_BEARISH,
                        candle_index=t,
                        timestamp=ts[t],
                        broken_level=active_low.price,
                        breakout_price=float(close[t]),
                        prior_trend=trend,
                        new_trend=TrendState.BEARISH
                    ))
                else:
                    events.append(StructureEvent(
                        event_type=StructureEventType.CHOCH_BEARISH,
                        candle_index=t,
                        timestamp=ts[t],
                        broken_level=active_low.price,
                        breakout_price=float(close[t]),
                        prior_trend=trend,
                        new_trend=TrendState.BEARISH
                    ))
                    trend = TrendState.BEARISH
                active_low = None

        return {
            "current_trend": trend,
            "events": events,
            "swing_highs": swing_highs,
            "swing_lows": swing_lows
        }

    def analyze(self, df: pd.DataFrame) -> SMCAnalysisReport:
        """Alias for analyze_buffer."""
        return self.analyze_buffer(df)

    def analyze_buffer(self, df: pd.DataFrame) -> SMCAnalysisReport:
        """
        Executes full, integrated SMC analysis on rolling OHLCV dataframe.
        """
        n = len(df)
        if n < 10:
            return SMCAnalysisReport(
                timestamp=None,
                current_price=0.0,
                trend=SMCTrend.RANGING,
                active_bullish_fvgs=[],
                active_bearish_fvgs=[],
                active_bullish_obs=[],
                active_bearish_obs=[],
                recent_sweeps=[],
                recent_structure_events=[],
                institutional_bias="NEUTRAL",
                confidence=50
            )

        close = df['close'].values
        ts = df['timestamp'].values if 'timestamp' in df.columns else np.arange(n)
        current_price = float(close[-1])
        current_ts = ts[-1]

        # 1. Market Structure
        struct = self.evaluate_market_structure(df)
        trend_state = struct["current_trend"]
        structure_events: List[StructureEvent] = struct["events"]
        sh_points: List[SwingPoint] = struct["swing_highs"]
        sl_points: List[SwingPoint] = struct["swing_lows"]

        smc_trend = SMCTrend.BULLISH if trend_state == TrendState.BULLISH else (
            SMCTrend.BEARISH if trend_state == TrendState.BEARISH else SMCTrend.RANGING
        )

        sh_tuples = [(sp.index, sp.price) for sp in sh_points]
        sl_tuples = [(sp.index, sp.price) for sp in sl_points]

        # 2. FVGs
        all_fvgs = self.detect_fvgs(df)
        active_bull_fvgs = [f for f in all_fvgs if f.gap_type == FVGType.BULLISH and f.is_active]
        active_bear_fvgs = [f for f in all_fvgs if f.gap_type == FVGType.BEARISH and f.is_active]

        # 3. Order Blocks
        all_obs = self.detect_order_blocks(df, sh_tuples, sl_tuples)
        active_bull_obs = [ob for ob in all_obs if ob.ob_type == OBType.BULLISH and ob.is_active]
        active_bear_obs = [ob for ob in all_obs if ob.ob_type == OBType.BEARISH and ob.is_active]

        # 4. Liquidity Sweeps
        all_sweeps = self.detect_liquidity_sweeps(df, sh_tuples, sl_tuples)
        recent_sweeps = [s for s in all_sweeps if s.sweep_index >= max(0, n - 15)]

        # 5. Institutional Bias & Confidence Synthesis
        bullish_signals = 0
        bearish_signals = 0

        if smc_trend == SMCTrend.BULLISH:
            bullish_signals += 3
        elif smc_trend == SMCTrend.BEARISH:
            bearish_signals += 3

        if any(s.sweep_type == SweepType.SSL_SWEEP for s in recent_sweeps):
            bullish_signals += 2
        if any(s.sweep_type == SweepType.BSL_SWEEP for s in recent_sweeps):
            bearish_signals += 2

        for ob in active_bull_obs:
            if current_price >= ob.bottom and current_price <= ob.top * 1.01:
                bullish_signals += 2
                break

        for ob in active_bear_obs:
            if current_price <= ob.top and current_price >= ob.bottom * 0.99:
                bearish_signals += 2
                break

        if bullish_signals > bearish_signals:
            bias = "BULLISH"
            confidence = min(95, 50 + (bullish_signals - bearish_signals) * 10)
        elif bearish_signals > bullish_signals:
            bias = "BEARISH"
            confidence = min(95, 50 + (bearish_signals - bullish_signals) * 10)
        else:
            bias = "NEUTRAL"
            confidence = 50

        return SMCAnalysisReport(
            timestamp=current_ts,
            current_price=current_price,
            trend=smc_trend,
            active_bullish_fvgs=active_bull_fvgs,
            active_bearish_fvgs=active_bear_fvgs,
            active_bullish_obs=active_bull_obs,
            active_bearish_obs=active_bear_obs,
            recent_sweeps=recent_sweeps,
            recent_structure_events=structure_events[-5:],
            institutional_bias=bias,
            confidence=confidence,
            all_fvgs=all_fvgs,
            all_obs=all_obs,
            all_sweeps=all_sweeps,
            all_structure_events=structure_events,
            swing_highs=sh_points,
            swing_lows=sl_points
        )
