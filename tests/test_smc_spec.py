"""
Test verification script for SMC algorithms specified in docs/smc_spec.md
"""
import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

# ============================================================================
# Core Definitions from docs/smc_spec.md
# ============================================================================

class FVGType(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"

class FVGState(Enum):
    UNMITIGATED = "UNMITIGATED"
    PARTIALLY_MITIGATED = "PARTIALLY_MITIGATED"
    CE_TESTED = "CE_TESTED"
    FULLY_MITIGATED = "FULLY_MITIGATED"
    INVERTED = "INVERTED"

@dataclass
class FairValueGap:
    gap_type: FVGType
    creation_index: int
    creation_timestamp: Any
    top: float
    bottom: float
    ce: float
    initial_size: float
    current_top: float
    current_bottom: float
    state: FVGState = FVGState.UNMITIGATED
    mitigation_index: Optional[int] = None
    mitigation_percentage: float = 0.0
    is_inverted: bool = False

    @property
    def is_active(self) -> bool:
        return self.state in [FVGState.UNMITIGATED, FVGState.PARTIALLY_MITIGATED, FVGState.CE_TESTED]

class FVGDetector:
    def __init__(self, min_atr_mult: float = 0.20, atr_period: int = 14):
        self.min_atr_mult = min_atr_mult
        self.atr_period = atr_period
        self.fvgs: List[FairValueGap] = []

    def detect_and_update(self, df: pd.DataFrame) -> List[FairValueGap]:
        if len(df) < 3:
            return []

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = len(df)

        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                   np.abs(low[1:] - close[:-1])))
        tr = np.insert(tr, 0, high[0] - low[0])
        atr = pd.Series(tr).rolling(window=self.atr_period, min_periods=1).mean().values

        self.fvgs.clear()

        for i in range(2, n):
            atr_thresh = self.min_atr_mult * atr[i]
            
            # Bullish FVG
            if low[i] > high[i-2]:
                gap_size = low[i] - high[i-2]
                if gap_size >= atr_thresh:
                    top = float(low[i])
                    bottom = float(high[i-2])
                    ce = (top + bottom) / 2.0
                    fvg = FairValueGap(
                        gap_type=FVGType.BULLISH,
                        creation_index=i,
                        creation_timestamp=df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                        top=top,
                        bottom=bottom,
                        ce=ce,
                        initial_size=gap_size,
                        current_top=top,
                        current_bottom=bottom
                    )
                    self.fvgs.append(fvg)

            # Bearish FVG
            elif high[i] < low[i-2]:
                gap_size = low[i-2] - high[i]
                if gap_size >= atr_thresh:
                    top = float(low[i-2])
                    bottom = float(high[i])
                    ce = (top + bottom) / 2.0
                    fvg = FairValueGap(
                        gap_type=FVGType.BEARISH,
                        creation_index=i,
                        creation_timestamp=df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                        top=top,
                        bottom=bottom,
                        ce=ce,
                        initial_size=gap_size,
                        current_top=top,
                        current_bottom=bottom
                    )
                    self.fvgs.append(fvg)

        for fvg in self.fvgs:
            start_k = fvg.creation_index + 1
            for k in range(start_k, n):
                bar_h = high[k]
                bar_l = low[k]
                bar_c = close[k]

                if fvg.gap_type == FVGType.BULLISH:
                    if bar_l < fvg.current_top:
                        fvg.current_top = min(fvg.current_top, bar_l)
                        penetration = (fvg.top - max(fvg.current_top, fvg.bottom))
                        fvg.mitigation_percentage = min(1.0, max(0.0, penetration / fvg.initial_size))

                        if bar_l <= fvg.bottom:
                            fvg.state = FVGState.FULLY_MITIGATED
                            fvg.mitigation_index = k
                            fvg.mitigation_percentage = 1.0
                            if bar_c < fvg.bottom:
                                fvg.is_inverted = True
                                fvg.state = FVGState.INVERTED
                            break
                        elif bar_l <= fvg.ce:
                            fvg.state = FVGState.CE_TESTED
                        else:
                            fvg.state = FVGState.PARTIALLY_MITIGATED

                elif fvg.gap_type == FVGType.BEARISH:
                    if bar_h > fvg.current_bottom:
                        fvg.current_bottom = max(fvg.current_bottom, bar_h)
                        penetration = (min(fvg.current_bottom, fvg.top) - fvg.bottom)
                        fvg.mitigation_percentage = min(1.0, max(0.0, penetration / fvg.initial_size))

                        if bar_h >= fvg.top:
                            fvg.state = FVGState.FULLY_MITIGATED
                            fvg.mitigation_index = k
                            fvg.mitigation_percentage = 1.0
                            if bar_c > fvg.top:
                                fvg.is_inverted = True
                                fvg.state = FVGState.INVERTED
                            break
                        elif bar_h >= fvg.ce:
                            fvg.state = FVGState.CE_TESTED
                        else:
                            fvg.state = FVGState.PARTIALLY_MITIGATED

        return self.fvgs

# ============================================================================
# Sweep Types & Detector
# ============================================================================

class SweepType(Enum):
    BSL_SWEEP = "BSL_SWEEP"
    SSL_SWEEP = "SSL_SWEEP"

@dataclass
class LiquiditySweep:
    sweep_type: SweepType
    sweep_index: int
    timestamp: Any
    level_price: float
    sweep_extreme: float
    close_price: float
    wick_ratio: float
    penetration_atr: float
    volume_ratio: float
    is_multi_bar: bool = False

class LiquiditySweepDetector:
    def __init__(self, min_wick_ratio: float = 0.35, max_penetration_atr: float = 1.5, atr_period: int = 14):
        self.min_wick_ratio = min_wick_ratio
        self.max_penetration_atr = max_penetration_atr
        self.atr_period = atr_period

    def detect_sweeps(self, df: pd.DataFrame, swing_highs: List[Tuple[int, float]], swing_lows: List[Tuple[int, float]]) -> List[LiquiditySweep]:
        n = len(df)
        if n < 5:
            return []

        high = df['high'].values
        low = df['low'].values
        open_p = df['open'].values
        close = df['close'].values
        vol = df['volume'].values

        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                   np.abs(low[1:] - close[:-1])))
        tr = np.insert(tr, 0, high[0] - low[0])
        atr = pd.Series(tr).rolling(self.atr_period, min_periods=1).mean().values
        vol_sma = pd.Series(vol).rolling(min(20, n), min_periods=1).mean().values

        sweeps: List[LiquiditySweep] = []

        for t in range(1, n):
            c_range = max(high[t] - low[t], 1e-9)
            upper_wick = high[t] - max(open_p[t], close[t])
            lower_wick = min(open_p[t], close[t]) - low[t]
            
            upper_wick_ratio = upper_wick / c_range
            lower_wick_ratio = lower_wick / c_range
            rvol = vol[t] / (vol_sma[t] + 1e-9)

            for sh_idx, sh_price in swing_highs:
                if sh_idx < t - 1:
                    if high[t] > sh_price and close[t] < sh_price:
                        penetration = high[t] - sh_price
                        pen_atr = penetration / (atr[t] + 1e-9)
                        if upper_wick_ratio >= self.min_wick_ratio and pen_atr <= self.max_penetration_atr:
                            sweeps.append(LiquiditySweep(
                                sweep_type=SweepType.BSL_SWEEP,
                                sweep_index=t,
                                timestamp=df['timestamp'].iloc[t] if 'timestamp' in df.columns else t,
                                level_price=sh_price,
                                sweep_extreme=high[t],
                                close_price=close[t],
                                wick_ratio=upper_wick_ratio,
                                penetration_atr=pen_atr,
                                volume_ratio=rvol,
                                is_multi_bar=False
                            ))
                            break

            for sl_idx, sl_price in swing_lows:
                if sl_idx < t - 1:
                    if low[t] < sl_price and close[t] > sl_price:
                        penetration = sl_price - low[t]
                        pen_atr = penetration / (atr[t] + 1e-9)
                        if lower_wick_ratio >= self.min_wick_ratio and pen_atr <= self.max_penetration_atr:
                            sweeps.append(LiquiditySweep(
                                sweep_type=SweepType.SSL_SWEEP,
                                sweep_index=t,
                                timestamp=df['timestamp'].iloc[t] if 'timestamp' in df.columns else t,
                                level_price=sl_price,
                                sweep_extreme=low[t],
                                close_price=close[t],
                                wick_ratio=lower_wick_ratio,
                                penetration_atr=pen_atr,
                                volume_ratio=rvol,
                                is_multi_bar=False
                            ))
                            break

        return sweeps

# ============================================================================
# Market Structure & BOS/CHoCH
# ============================================================================

class TrendState(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"

class StructureEventType(Enum):
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"

@dataclass
class SwingPoint:
    index: int
    timestamp: Any
    price: float
    is_high: bool
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

class MarketStructureEngine:
    def __init__(self, swing_radius: int = 2):
        self.radius = swing_radius
        self.trend: TrendState = TrendState.SIDEWAYS
        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.events: List[StructureEvent] = []

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if n < (2 * self.radius + 1):
            return {"trend": self.trend, "events": [], "swing_highs": [], "swing_lows": []}

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(n)

        self.swing_highs.clear()
        self.swing_lows.clear()
        self.events.clear()
        self.trend = TrendState.SIDEWAYS

        active_high: Optional[SwingPoint] = None
        active_low: Optional[SwingPoint] = None

        for t in range(n):
            pivot_idx = t - self.radius
            if pivot_idx >= self.radius:
                is_sh = True
                p_high = high[pivot_idx]
                for r in range(1, self.radius + 1):
                    if high[pivot_idx - r] >= p_high or high[pivot_idx + r] >= p_high:
                        is_sh = False
                        break
                if is_sh:
                    sp = SwingPoint(pivot_idx, timestamps[pivot_idx], p_high, True, t)
                    self.swing_highs.append(sp)
                    active_high = sp

                is_sl = True
                p_low = low[pivot_idx]
                for r in range(1, self.radius + 1):
                    if low[pivot_idx - r] <= p_low or low[pivot_idx + r] <= p_low:
                        is_sl = False
                        break
                if is_sl:
                    sp = SwingPoint(pivot_idx, timestamps[pivot_idx], p_low, False, t)
                    self.swing_lows.append(sp)
                    active_low = sp

            if active_high is not None and close[t] > active_high.price:
                if self.trend == TrendState.BULLISH:
                    event = StructureEvent(
                        event_type=StructureEventType.BOS_BULLISH,
                        candle_index=t,
                        timestamp=timestamps[t],
                        broken_level=active_high.price,
                        breakout_price=close[t],
                        prior_trend=self.trend,
                        new_trend=TrendState.BULLISH
                    )
                    self.events.append(event)
                else:
                    event = StructureEvent(
                        event_type=StructureEventType.CHOCH_BULLISH,
                        candle_index=t,
                        timestamp=timestamps[t],
                        broken_level=active_high.price,
                        breakout_price=close[t],
                        prior_trend=self.trend,
                        new_trend=TrendState.BULLISH
                    )
                    self.trend = TrendState.BULLISH
                    self.events.append(event)
                active_high = None

            elif active_low is not None and close[t] < active_low.price:
                if self.trend == TrendState.BEARISH:
                    event = StructureEvent(
                        event_type=StructureEventType.BOS_BEARISH,
                        candle_index=t,
                        timestamp=timestamps[t],
                        broken_level=active_low.price,
                        breakout_price=close[t],
                        prior_trend=self.trend,
                        new_trend=TrendState.BEARISH
                    )
                    self.events.append(event)
                else:
                    event = StructureEvent(
                        event_type=StructureEventType.CHOCH_BEARISH,
                        candle_index=t,
                        timestamp=timestamps[t],
                        broken_level=active_low.price,
                        breakout_price=close[t],
                        prior_trend=self.trend,
                        new_trend=TrendState.BEARISH
                    )
                    self.trend = TrendState.BEARISH
                    self.events.append(event)
                active_low = None

        return {
            "current_trend": self.trend,
            "events": self.events,
            "swing_highs": self.swing_highs,
            "swing_lows": self.swing_lows
        }

# ============================================================================
# Unit Tests
# ============================================================================

def test_bullish_fvg_detection_and_mitigation():
    candles = [
        {"open": 96, "high": 100, "low": 95, "close": 99, "volume": 1000},
        {"open": 99, "high": 104, "low": 99, "close": 103, "volume": 1200},
        {"open": 103, "high": 110, "low": 105, "close": 109, "volume": 2000},
        {"open": 109, "high": 108, "low": 103, "close": 105, "volume": 1100},
        {"open": 105, "high": 106, "low": 101, "close": 104, "volume": 900},
        {"open": 104, "high": 102, "low": 99, "close": 98, "volume": 1500},
    ]
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="1h")

    detector = FVGDetector(min_atr_mult=0.1)
    fvgs = detector.detect_and_update(df)

    assert len(fvgs) >= 1
    fvg = fvgs[0]
    assert fvg.gap_type == FVGType.BULLISH
    assert fvg.bottom == 100.0
    assert fvg.top == 105.0
    assert fvg.ce == 102.5
    assert fvg.state in [FVGState.FULLY_MITIGATED, FVGState.INVERTED]
    assert fvg.mitigation_percentage == 1.0

def test_bearish_fvg_detection():
    candles = [
        {"open": 108, "high": 110, "low": 105, "close": 106, "volume": 1000},
        {"open": 106, "high": 106, "low": 100, "close": 101, "volume": 1200},
        {"open": 100, "high": 98, "low": 94, "close": 95, "volume": 2000},
    ]
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="1h")

    detector = FVGDetector(min_atr_mult=0.1)
    fvgs = detector.detect_and_update(df)

    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg.gap_type == FVGType.BEARISH
    assert fvg.bottom == 98.0
    assert fvg.top == 105.0
    assert fvg.ce == 101.5
    assert fvg.state == FVGState.UNMITIGATED

def test_bsl_liquidity_sweep():
    # Swing high at candle 1: H=120
    # Candle 4: sweeps above 120 (H=122), but closes back down at 118 with long upper wick
    candles = [
        {"open": 110, "high": 114, "low": 109, "close": 113, "volume": 1000},
        {"open": 113, "high": 120, "low": 112, "close": 116, "volume": 1100}, # Swing high
        {"open": 116, "high": 117, "low": 114, "close": 115, "volume": 900},
        {"open": 115, "high": 116, "low": 113, "close": 114, "volume": 950},
        {"open": 115, "high": 122, "low": 114, "close": 117, "volume": 2200}, # Sweep bar: high 122 > 120, close 117 < 120
    ]
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="1h")

    detector = LiquiditySweepDetector(min_wick_ratio=0.30)
    sweeps = detector.detect_sweeps(df, swing_highs=[(1, 120.0)], swing_lows=[])

    assert len(sweeps) == 1
    assert sweeps[0].sweep_type == SweepType.BSL_SWEEP
    assert sweeps[0].level_price == 120.0
    assert sweeps[0].sweep_extreme == 122.0

def test_ssl_liquidity_sweep():
    # Swing low at candle 1: L=80
    # Candle 4: sweeps below 80 (L=77), but closes back up at 83 with long lower wick
    candles = [
        {"open": 90, "high": 92, "low": 86, "close": 87, "volume": 1000},
        {"open": 87, "high": 88, "low": 80, "close": 84, "volume": 1100}, # Swing low
        {"open": 84, "high": 86, "low": 83, "close": 85, "volume": 900},
        {"open": 85, "high": 87, "low": 84, "close": 86, "volume": 950},
        {"open": 85, "high": 87, "low": 77, "close": 83, "volume": 2200}, # Sweep bar: low 77 < 80, close 83 > 80
    ]
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="1h")

    detector = LiquiditySweepDetector(min_wick_ratio=0.30)
    sweeps = detector.detect_sweeps(df, swing_highs=[], swing_lows=[(1, 80.0)])

    assert len(sweeps) == 1
    assert sweeps[0].sweep_type == SweepType.SSL_SWEEP
    assert sweeps[0].level_price == 80.0
    assert sweeps[0].sweep_extreme == 77.0

def test_market_structure_bos_and_choch():
    # Generate series that forms swing high, breaks above (CHOCH_BULLISH), forms next high, breaks above (BOS_BULLISH)
    # radius = 1 for compact test
    candles = [
        {"open": 100, "high": 102, "low": 98, "close": 100, "volume": 1000},
        {"open": 100, "high": 110, "low": 99, "close": 105, "volume": 1000}, # Swing high at idx 1 (price 110)
        {"open": 105, "high": 104, "low": 96, "close": 97, "volume": 1000},  # Low at idx 2
        {"open": 97, "high": 115, "low": 97, "close": 114, "volume": 1500},  # Closes above 110 -> CHOCH_BULLISH
        {"open": 114, "high": 125, "low": 113, "close": 122, "volume": 1200}, # Swing high at idx 4 (price 125)
        {"open": 122, "high": 120, "low": 115, "close": 116, "volume": 1000},
        {"open": 116, "high": 130, "low": 116, "close": 128, "volume": 2000}, # Closes above 125 -> BOS_BULLISH
    ]
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="1h")

    engine = MarketStructureEngine(swing_radius=1)
    res = engine.evaluate(df)

    event_types = [e.event_type for e in res["events"]]
    assert StructureEventType.CHOCH_BULLISH in event_types
    assert StructureEventType.BOS_BULLISH in event_types
    assert res["current_trend"] == TrendState.BULLISH
