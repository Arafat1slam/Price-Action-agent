"""
indicators_engine.py
====================
Production-grade Analytical & Quantitative Indicator Engine.
Implements:
1. VolumeProfileEngine: Equidistant price binning, candle overlap allocation, POC, Value Area (70%), HVN/LVN.
2. VWAPEngine: Typical price, cumulative/rolling VWAP, +-1/2/3 sigma bands, exhaustion signals.
3. VSAEngine: RVOL, volume z-score, spread ratio, close location, Wyckoff absorption/climaxes/tests/sweeps.
4. KDESupportResistanceEngine: Continuous Gaussian KDE, adaptive ATR bandwidth, 65% density zones, polarity flips.
5. FibonacciEngine: Dynamic fractal swing anchoring, retracements, Golden Pocket, extensions (-0.272, -0.618, 1.618).
6. IndicatorConfluenceEngine: Multi-factor confluence scoring & zone synthesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


# ============================================================================
# Helper Functions: Data Normalization & Validation
# ============================================================================

def _normalize_ohlcv(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts open, high, low, close, volume numpy arrays from DataFrame,
    handling lowercase or uppercase column names.
    """
    col_map = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in col_map:
            raise ValueError(f"Missing required OHLCV column '{col}' in DataFrame.")

    opens = df[col_map["open"]].to_numpy(dtype=np.float64)
    highs = df[col_map["high"]].to_numpy(dtype=np.float64)
    lows = df[col_map["low"]].to_numpy(dtype=np.float64)
    closes = df[col_map["close"]].to_numpy(dtype=np.float64)
    volumes = df[col_map["volume"]].to_numpy(dtype=np.float64)

    return opens, highs, lows, closes, volumes


# ============================================================================
# 1. Volume Profile & Auction Market Theory Engine
# ============================================================================

@dataclass(frozen=True)
class PriceBin:
    price_low: float
    price_high: float
    price_center: float
    volume: float
    pct_of_total: float


@dataclass
class VolumeProfileResult:
    poc_price: float
    vah_price: float
    val_price: float
    total_volume: float
    bins: List[PriceBin]
    hvn_levels: List[float]
    lvn_levels: List[float]
    vpoc_untested: bool = True


@dataclass
class VolumeProfileConfig:
    num_bins: int = 60
    value_area_pct: float = 0.70
    hvn_prominence_factor: float = 0.25


class VolumeProfileEngine:
    """
    Quantitative Volume Profile Engine implementing Steidlmayer Auction Market Theory.
    Distributes candle volume uniformly across overlapping price bins.
    Calculates Point of Control (POC), Value Area (VAH / VAL), and HVN/LVN nodes.
    """

    def __init__(
        self,
        config: Optional[VolumeProfileConfig] = None,
        num_bins: int = 60,
        value_area_pct: float = 0.70,
        hvn_prominence_factor: float = 0.25,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = VolumeProfileConfig(
                num_bins=num_bins,
                value_area_pct=value_area_pct,
                hvn_prominence_factor=hvn_prominence_factor,
            )

    def compute(
        self, df: pd.DataFrame, config: Optional[VolumeProfileConfig] = None
    ) -> VolumeProfileResult:
        cfg = config or self.config
        if len(df) == 0:
            return VolumeProfileResult(
                poc_price=0.0,
                vah_price=0.0,
                val_price=0.0,
                total_volume=0.0,
                bins=[],
                hvn_levels=[],
                lvn_levels=[],
                vpoc_untested=False,
            )

        _, highs, lows, closes, volumes = _normalize_ohlcv(df)

        min_p = float(np.min(lows))
        max_p = float(np.max(highs))
        if max_p <= min_p:
            max_p = min_p + 1e-4

        num_bins = max(5, cfg.num_bins)
        bin_edges = np.linspace(min_p, max_p, num_bins + 1)
        bin_volumes = np.zeros(num_bins, dtype=np.float64)

        # Distribute volume uniformly across overlapping candle range
        for h, l, v in zip(highs, lows, volumes):
            c_range = h - l
            if c_range <= 1e-9:
                idx = int(np.clip(np.searchsorted(bin_edges, l, side="right") - 1, 0, num_bins - 1))
                bin_volumes[idx] += v
                continue

            b_start = int(np.clip(np.searchsorted(bin_edges, l, side="right") - 1, 0, num_bins - 1))
            b_end = int(np.clip(np.searchsorted(bin_edges, h, side="left"), 0, num_bins - 1))

            for b_idx in range(b_start, b_end + 1):
                b_low = bin_edges[b_idx]
                b_high = bin_edges[b_idx + 1]
                overlap = max(0.0, min(h, b_high) - max(l, b_low))
                if overlap > 0.0:
                    bin_volumes[b_idx] += v * (overlap / c_range)

        total_vol = float(np.sum(bin_volumes))
        if total_vol <= 0.0:
            total_vol = 1e-9

        # Point of Control (POC) with centroid tie-breaking
        max_v = np.max(bin_volumes)
        max_indices = np.where(bin_volumes == max_v)[0]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        if len(max_indices) > 1:
            centroid = float(np.sum(bin_centers * bin_volumes) / total_vol)
            poc_idx = int(max_indices[np.argmin(np.abs(bin_centers[max_indices] - centroid))])
        else:
            poc_idx = int(max_indices[0])

        poc_price = float(bin_centers[poc_idx])

        # Value Area (VA) 70% Steidlmayer Dual-Bin Expansion
        target_va_vol = cfg.value_area_pct * total_vol
        accum_vol = bin_volumes[poc_idx]
        up_idx = poc_idx
        down_idx = poc_idx

        while accum_vol < target_va_vol and (up_idx < num_bins - 1 or down_idx > 0):
            next_up = bin_volumes[up_idx + 1] if up_idx + 1 < num_bins else 0.0
            next_down = bin_volumes[down_idx - 1] if down_idx - 1 >= 0 else 0.0

            if next_up > next_down:
                up_idx += 1
                accum_vol += next_up
            elif next_down > next_up:
                down_idx -= 1
                accum_vol += next_down
            else:
                if up_idx + 1 < num_bins:
                    up_idx += 1
                    accum_vol += next_up
                if down_idx - 1 >= 0 and accum_vol < target_va_vol:
                    down_idx -= 1
                    accum_vol += next_down

        val_price = float(bin_edges[down_idx])
        vah_price = float(bin_edges[up_idx + 1])

        # HVN & LVN Detection via Peak/Trough Finding
        vol_mean = float(np.mean(bin_volumes))
        vol_std = float(np.std(bin_volumes))
        prom = cfg.hvn_prominence_factor * vol_std if vol_std > 0 else 0.05 * vol_mean

        peaks, _ = find_peaks(bin_volumes, distance=2, prominence=prom if prom > 0 else None)
        valleys, _ = find_peaks(-bin_volumes, distance=2, prominence=prom if prom > 0 else None)

        hvn_levels = [float(bin_centers[p]) for p in peaks if bin_volumes[p] >= vol_mean]
        lvn_levels = [float(bin_centers[v]) for v in valleys if bin_volumes[v] <= vol_mean]

        # Check if POC has been touched by the latest candle
        latest_low = float(lows[-1])
        latest_high = float(highs[-1])
        vpoc_untested = not (latest_low <= poc_price <= latest_high)

        bins_list = [
            PriceBin(
                price_low=float(bin_edges[i]),
                price_high=float(bin_edges[i + 1]),
                price_center=float(bin_centers[i]),
                volume=float(bin_volumes[i]),
                pct_of_total=float(bin_volumes[i] / total_vol),
            )
            for i in range(num_bins)
        ]

        return VolumeProfileResult(
            poc_price=round(poc_price, 4),
            vah_price=round(vah_price, 4),
            val_price=round(val_price, 4),
            total_volume=round(total_vol, 4),
            bins=bins_list,
            hvn_levels=[round(h, 4) for h in hvn_levels],
            lvn_levels=[round(l, 4) for l in lvn_levels],
            vpoc_untested=vpoc_untested,
        )

    def calculate(self, df: pd.DataFrame, config: Optional[VolumeProfileConfig] = None) -> VolumeProfileResult:
        return self.compute(df, config=config)


def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 60,
    value_area_pct: float = 0.70,
    config: Optional[VolumeProfileConfig] = None,
) -> VolumeProfileResult:
    """Convenience standalone wrapper for VolumeProfileEngine."""
    if config is None:
        config = VolumeProfileConfig(num_bins=num_bins, value_area_pct=value_area_pct)
    engine = VolumeProfileEngine(config=config)
    return engine.compute(df)


# ============================================================================
# 2. VWAP & Standard Deviation Bands Engine
# ============================================================================

class VWAPAnchorType(Enum):
    SESSION = "SESSION"
    ROLLING = "ROLLING"
    ANCHORED = "ANCHORED"


@dataclass(frozen=True)
class VWAPExhaustionSignal:
    bar_index: int
    price: float
    vwap: float
    z_score: float
    sigma_level: float
    signal_type: str  # "OVERBOUGHT_EXHAUSTION", "OVERSOLD_EXHAUSTION", etc.
    bias: str  # "BEARISH", "BULLISH"
    description: str


@dataclass
class VWAPResult:
    vwap: pd.Series
    upper_1sigma: pd.Series
    lower_1sigma: pd.Series
    upper_2sigma: pd.Series
    lower_2sigma: pd.Series
    upper_3sigma: pd.Series
    lower_3sigma: pd.Series
    std_dev: pd.Series
    z_score: pd.Series
    exhaustion_signals: List[VWAPExhaustionSignal] = field(default_factory=list)


class VWAPEngine:
    """
    Vectorized Volume-Weighted Average Price (VWAP) Engine.
    Computes Typical Price, running variance, and standard deviation bands (1, 2, 3 sigma).
    Detects statistical exhaustion when price exceeds +-2 sigma and +-3 sigma.
    """

    def __init__(
        self,
        anchor_type: VWAPAnchorType = VWAPAnchorType.ANCHORED,
        rolling_window: Optional[int] = None,
        anchor_col: Optional[str] = None,
        exhaustion_threshold: float = 2.0,
    ):
        self.anchor_type = anchor_type
        self.rolling_window = rolling_window
        self.anchor_col = anchor_col
        self.exhaustion_threshold = exhaustion_threshold

    def compute(
        self,
        df: pd.DataFrame,
        anchor_col: Optional[str] = None,
        rolling_window: Optional[int] = None,
    ) -> VWAPResult:
        if len(df) == 0:
            empty_s = pd.Series(dtype=np.float64)
            return VWAPResult(
                vwap=empty_s,
                upper_1sigma=empty_s,
                lower_1sigma=empty_s,
                upper_2sigma=empty_s,
                lower_2sigma=empty_s,
                upper_3sigma=empty_s,
                lower_3sigma=empty_s,
                std_dev=empty_s,
                z_score=empty_s,
                exhaustion_signals=[],
            )

        col_map = {c.lower(): c for c in df.columns}
        h = df[col_map["high"]].astype(np.float64)
        l = df[col_map["low"]].astype(np.float64)
        c = df[col_map["close"]].astype(np.float64)
        v = df[col_map["volume"]].astype(np.float64)

        tp = (h + l + c) / 3.0
        tp_vol = tp * v
        tp2_vol = (tp ** 2) * v

        anc_col = anchor_col or self.anchor_col
        roll_win = rolling_window if rolling_window is not None else self.rolling_window

        if anc_col is not None and anc_col in df.columns:
            groups = df.groupby(df[anc_col])
            cum_tp_vol = groups.apply(lambda g: tp_vol.loc[g.index].cumsum()).reset_index(level=0, drop=True)
            cum_tp2_vol = groups.apply(lambda g: tp2_vol.loc[g.index].cumsum()).reset_index(level=0, drop=True)
            cum_vol = groups.apply(lambda g: v.loc[g.index].cumsum()).reset_index(level=0, drop=True)
        elif roll_win is not None and roll_win > 0:
            cum_tp_vol = tp_vol.rolling(window=roll_win, min_periods=1).sum()
            cum_tp2_vol = tp2_vol.rolling(window=roll_win, min_periods=1).sum()
            cum_vol = v.rolling(window=roll_win, min_periods=1).sum()
        else:
            cum_tp_vol = tp_vol.cumsum()
            cum_tp2_vol = tp2_vol.cumsum()
            cum_vol = v.cumsum()

        safe_vol = cum_vol.replace(0, 1e-9)
        vwap = cum_tp_vol / safe_vol

        # Vectorized variance: E[X^2] - (E[X])^2
        variance = (cum_tp2_vol / safe_vol) - (vwap ** 2)
        variance = np.maximum(variance, 0.0)
        std_dev = np.sqrt(variance)
        safe_std = std_dev.replace(0, 1e-9)

        z_score = (c - vwap) / safe_std

        upper_1sigma = vwap + 1.0 * std_dev
        lower_1sigma = vwap - 1.0 * std_dev
        upper_2sigma = vwap + 2.0 * std_dev
        lower_2sigma = vwap - 2.0 * std_dev
        upper_3sigma = vwap + 3.0 * std_dev
        lower_3sigma = vwap - 3.0 * std_dev

        # Exhaustion Signals Extraction
        exhaustion_signals: List[VWAPExhaustionSignal] = []
        z_arr = z_score.to_numpy()
        c_arr = c.to_numpy()
        vwap_arr = vwap.to_numpy()

        for idx, (z, price, cur_vwap) in enumerate(zip(z_arr, c_arr, vwap_arr)):
            if z >= 3.0:
                exhaustion_signals.append(
                    VWAPExhaustionSignal(
                        bar_index=idx,
                        price=float(price),
                        vwap=float(cur_vwap),
                        z_score=round(float(z), 2),
                        sigma_level=3.0,
                        signal_type="EXTREME_OVERBOUGHT",
                        bias="BEARISH",
                        description=f"Extreme +3σ Overbought Outlier (z={z:.2f}): Imminent liquidation snapback.",
                    )
                )
            elif z >= self.exhaustion_threshold:
                exhaustion_signals.append(
                    VWAPExhaustionSignal(
                        bar_index=idx,
                        price=float(price),
                        vwap=float(cur_vwap),
                        z_score=round(float(z), 2),
                        sigma_level=2.0,
                        signal_type="OVERBOUGHT_EXHAUSTION",
                        bias="BEARISH",
                        description=f"Statistical +2σ Exhaustion (z={z:.2f}): Mean-reversion to VWAP favored.",
                    )
                )
            elif z <= -3.0:
                exhaustion_signals.append(
                    VWAPExhaustionSignal(
                        bar_index=idx,
                        price=float(price),
                        vwap=float(cur_vwap),
                        z_score=round(float(z), 2),
                        sigma_level=-3.0,
                        signal_type="EXTREME_OVERSOLD",
                        bias="BULLISH",
                        description=f"Extreme -3σ Oversold Outlier (z={z:.2f}): Strong mean-reversion snapback.",
                    )
                )
            elif z <= -self.exhaustion_threshold:
                exhaustion_signals.append(
                    VWAPExhaustionSignal(
                        bar_index=idx,
                        price=float(price),
                        vwap=float(cur_vwap),
                        z_score=round(float(z), 2),
                        sigma_level=-2.0,
                        signal_type="OVERSOLD_EXHAUSTION",
                        bias="BULLISH",
                        description=f"Statistical -2σ Exhaustion (z={z:.2f}): Mean-reversion to VWAP favored.",
                    )
                )

        return VWAPResult(
            vwap=vwap,
            upper_1sigma=upper_1sigma,
            lower_1sigma=lower_1sigma,
            upper_2sigma=upper_2sigma,
            lower_2sigma=lower_2sigma,
            upper_3sigma=upper_3sigma,
            lower_3sigma=lower_3sigma,
            std_dev=std_dev,
            z_score=z_score,
            exhaustion_signals=exhaustion_signals,
        )

    def calculate(
        self,
        df: pd.DataFrame,
        anchor_col: Optional[str] = None,
        rolling_window: Optional[int] = None,
    ) -> VWAPResult:
        return self.compute(df, anchor_col=anchor_col, rolling_window=rolling_window)


def calculate_vwap(
    df: pd.DataFrame,
    anchor_col: Optional[str] = None,
    rolling_window: Optional[int] = None,
) -> VWAPResult:
    """Convenience standalone wrapper for VWAPEngine."""
    engine = VWAPEngine(anchor_col=anchor_col, rolling_window=rolling_window)
    return engine.compute(df)


# ============================================================================
# 3. Wyckoff Volume Spread Analysis (VSA) Engine
# ============================================================================

class VSASignalType(Enum):
    STOPPING_VOLUME = "STOPPING_VOLUME"
    ABSORPTION = "ABSORPTION"
    NO_SUPPLY = "NO_SUPPLY"
    NO_DEMAND = "NO_DEMAND"
    BUYING_CLIMAX = "BUYING_CLIMAX"
    SELLING_CLIMAX = "SELLING_CLIMAX"
    UPTHRUST = "UPTHRUST"
    SPRING = "SPRING"
    EFFORT_NO_RESULT = "EFFORT_NO_RESULT"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class VSASignal:
    bar_index: int
    signal_type: VSASignalType
    bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    rvol: float
    volume_zscore: float
    spread_ratio: float
    close_location: float
    confidence: float
    description: str


class VSAEngine:
    """
    Wyckoff Volume Spread Analysis (VSA) Engine.
    Normalized metrics: RVOL vs SMA20, volume z-score, spread ratio, close location (CL).
    Detects Stopping Volume/Absorption, No Supply, No Demand, Climaxes, and Sweeps (Upthrust/Spring).
    """

    def __init__(
        self,
        lookback: int = 20,
        rvol_high: float = 2.0,
        rvol_climax: float = 2.5,
        rvol_low: float = 0.75,
    ):
        self.lookback = lookback
        self.rvol_high = rvol_high
        self.rvol_climax = rvol_climax
        self.rvol_low = rvol_low

    def analyze(self, df: pd.DataFrame, lookback: Optional[int] = None) -> List[VSASignal]:
        lb = lookback or self.lookback
        signals: List[VSASignal] = []
        n = len(df)
        if n < lb + 2:
            return signals

        opens, highs, lows, closes, volumes = _normalize_ohlcv(df)
        spreads = highs - lows

        vol_series = pd.Series(volumes)
        spread_series = pd.Series(spreads)
        close_series = pd.Series(closes)

        vol_sma = vol_series.rolling(window=lb, min_periods=5).mean().to_numpy()
        vol_std = vol_series.rolling(window=lb, min_periods=5).std().to_numpy()
        spread_sma = spread_series.rolling(window=lb, min_periods=5).mean().to_numpy()
        ema20 = close_series.ewm(span=20, adjust=False).mean().to_numpy()

        for i in range(lb, n):
            c_spread = spreads[i]
            c_range = max(c_spread, 1e-9)
            c_vol = volumes[i]
            c_close = closes[i]
            c_open = opens[i]
            p_close = closes[i - 1]

            rvol = float(c_vol / (vol_sma[i] + 1e-9))
            z_vol = float((c_vol - vol_sma[i]) / (vol_std[i] + 1e-9)) if vol_std[i] > 0 else 0.0
            spread_ratio = float(c_spread / (spread_sma[i] + 1e-9))
            close_loc = float((c_close - lows[i]) / c_range)

            upper_wick = float((highs[i] - max(c_close, c_open)) / c_range)
            lower_wick = float((min(c_close, c_open) - lows[i]) / c_range)

            # Trend context via EMA20 and 5-bar momentum
            is_downtrend = (c_close < ema20[i]) or (p_close < closes[max(0, i - 5)])
            is_uptrend = (c_close > ema20[i]) or (p_close > closes[max(0, i - 5)])

            # Recent local extrema over last 5 bars for sweep detection
            recent_high = np.max(highs[max(0, i - 5) : i])
            recent_low = np.min(lows[max(0, i - 5) : i])

            # 1. Stopping Volume / Absorption
            if is_downtrend and (rvol >= self.rvol_high or z_vol >= 2.0) and spread_ratio <= 1.15 and close_loc >= 0.45:
                sig_type = VSASignalType.ABSORPTION if close_loc >= 0.60 else VSASignalType.STOPPING_VOLUME
                conf = min(0.95, 0.65 + (rvol / 10.0))
                signals.append(
                    VSASignal(
                        bar_index=i,
                        signal_type=sig_type,
                        bias="BULLISH",
                        rvol=round(rvol, 2),
                        volume_zscore=round(z_vol, 2),
                        spread_ratio=round(spread_ratio, 2),
                        close_location=round(close_loc, 2),
                        confidence=round(conf, 2),
                        description=f"{sig_type.value}: Smart money absorption in downtrend (RVOL={rvol:.2f}, CL={close_loc:.2f}).",
                    )
                )

            # 2. No Supply: Low volume test of support
            elif (
                c_close <= p_close
                and rvol <= self.rvol_low
                and c_vol < volumes[i - 1]
                and c_vol < volumes[i - 2]
                and spread_ratio <= 0.90
                and close_loc >= 0.30
            ):
                signals.append(
                    VSASignal(
                        bar_index=i,
                        signal_type=VSASignalType.NO_SUPPLY,
                        bias="BULLISH",
                        rvol=round(rvol, 2),
                        volume_zscore=round(z_vol, 2),
                        spread_ratio=round(spread_ratio, 2),
                        close_location=round(close_loc, 2),
                        confidence=0.82,
                        description=f"No Supply: Volume dried up on down bar (RVOL={rvol:.2f}). Lack of selling pressure.",
                    )
                )

            # 3. No Demand: Low volume test of resistance
            elif (
                c_close >= p_close
                and rvol <= self.rvol_low
                and c_vol < volumes[i - 1]
                and c_vol < volumes[i - 2]
                and spread_ratio <= 0.90
                and close_loc <= 0.70
            ):
                signals.append(
                    VSASignal(
                        bar_index=i,
                        signal_type=VSASignalType.NO_DEMAND,
                        bias="BEARISH",
                        rvol=round(rvol, 2),
                        volume_zscore=round(z_vol, 2),
                        spread_ratio=round(spread_ratio, 2),
                        close_location=round(close_loc, 2),
                        confidence=0.82,
                        description=f"No Demand: Volume dried up on rally bar (RVOL={rvol:.2f}). Lack of buyers.",
                    )
                )

            # 4. Buying Climax: Exhaustion at top
            elif is_uptrend and (rvol >= self.rvol_climax or z_vol >= 2.5) and spread_ratio >= 1.5 and (close_loc <= 0.60 or upper_wick >= 0.35):
                signals.append(
                    VSASignal(
                        bar_index=i,
                        signal_type=VSASignalType.BUYING_CLIMAX,
                        bias="BEARISH",
                        rvol=round(rvol, 2),
                        volume_zscore=round(z_vol, 2),
                        spread_ratio=round(spread_ratio, 2),
                        close_location=round(close_loc, 2),
                        confidence=0.90,
                        description=f"Buying Climax: Heavy distribution into retail FOMO (RVOL={rvol:.2f}, UpperWick={upper_wick:.2f}).",
                    )
                )

            # 5. Selling Climax: Panic capitulation at bottom
            elif is_downtrend and (rvol >= self.rvol_climax or z_vol >= 2.5) and spread_ratio >= 1.5 and (close_loc >= 0.40 or lower_wick >= 0.35):
                signals.append(
                    VSASignal(
                        bar_index=i,
                        signal_type=VSASignalType.SELLING_CLIMAX,
                        bias="BULLISH",
                        rvol=round(rvol, 2),
                        volume_zscore=round(z_vol, 2),
                        spread_ratio=round(spread_ratio, 2),
                        close_location=round(close_loc, 2),
                        confidence=0.90,
                        description=f"Selling Climax: Panic selling absorbed by institutional smart money (RVOL={rvol:.2f}).",
                    )
                )

            # 6. Upthrust (UTAD) / Sweep of Highs
            elif highs[i] > recent_high and c_close <= recent_high and (rvol >= 1.2 or z_vol >= 0.8) and upper_wick >= 0.38 and close_loc <= 0.35:
                signals.append(
                    VSASignal(
                        bar_index=i,
                        signal_type=VSASignalType.UPTHRUST,
                        bias="BEARISH",
                        rvol=round(rvol, 2),
                        volume_zscore=round(z_vol, 2),
                        spread_ratio=round(spread_ratio, 2),
                        close_location=round(close_loc, 2),
                        confidence=0.88,
                        description=f"Upthrust / Sweep of Highs: Failed breakout with heavy upper rejection (RVOL={rvol:.2f}).",
                    )
                )

            # 7. Wyckoff Spring / Sweep of Lows
            elif lows[i] < recent_low and c_close >= recent_low and (rvol >= 1.2 or z_vol >= 0.8) and lower_wick >= 0.38 and close_loc >= 0.65:
                signals.append(
                    VSASignal(
                        bar_index=i,
                        signal_type=VSASignalType.SPRING,
                        bias="BULLISH",
                        rvol=round(rvol, 2),
                        volume_zscore=round(z_vol, 2),
                        spread_ratio=round(spread_ratio, 2),
                        close_location=round(close_loc, 2),
                        confidence=0.88,
                        description=f"Wyckoff Spring / Sweep of Lows: Stop run followed by aggressive snapback (RVOL={rvol:.2f}).",
                    )
                )

            # 8. Effort with No Result
            elif rvol >= 1.8 and spread_ratio <= 0.80:
                bias = "BULLISH" if close_loc >= 0.50 else "BEARISH"
                signals.append(
                    VSASignal(
                        bar_index=i,
                        signal_type=VSASignalType.EFFORT_NO_RESULT,
                        bias=bias,
                        rvol=round(rvol, 2),
                        volume_zscore=round(z_vol, 2),
                        spread_ratio=round(spread_ratio, 2),
                        close_location=round(close_loc, 2),
                        confidence=0.78,
                        description=f"Effort vs Result: High effort (RVOL={rvol:.2f}) with narrow spread ({spread_ratio:.2f}). Hidden absorption.",
                    )
                )

        return signals


def analyze_vsa(df: pd.DataFrame, lookback: int = 20) -> List[VSASignal]:
    """Convenience standalone wrapper for VSAEngine."""
    engine = VSAEngine(lookback=lookback)
    return engine.analyze(df)


# ============================================================================
# 4. Kernel Density Estimation (KDE) Support & Resistance Engine
# ============================================================================

@dataclass(frozen=True)
class KDESRZone:
    peak_price: float
    zone_low: float
    zone_high: float
    density: float
    prominence: float
    level_type: str  # "SUPPORT" or "RESISTANCE"
    touch_count: int
    score: float
    is_flipped: bool = False
    flip_type: Optional[str] = None  # "FLIPPED_SUPPORT", "FLIPPED_RESISTANCE"


@dataclass
class KDESRResult:
    zones: List[KDESRZone]
    nearest_support: Optional[KDESRZone]
    nearest_resistance: Optional[KDESRZone]
    eval_grid: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None


class KDESupportResistanceEngine:
    """
    Non-parametric Support & Resistance Engine using SciPy Gaussian Kernel Density Estimation.
    Extracts multi-factor weighted pivot touch points (structural, recency, volume).
    Uses Adaptive ATR bandwidth to prevent over-smoothing.
    Finds density modes with prominence filtering, 65% density span zone boundaries,
    and classifies polarity flips (broken resistance becoming support and vice versa).
    """

    def __init__(
        self,
        bandwidth_scale: float = 0.50,
        num_eval_points: int = 500,
        prominence_factor: float = 0.15,
    ):
        self.bandwidth_scale = bandwidth_scale
        self.num_eval_points = num_eval_points
        self.prominence_factor = prominence_factor

    def compute(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None,
        atr_period: int = 14,
    ) -> KDESRResult:
        n = len(df)
        if n < 10:
            return KDESRResult(zones=[], nearest_support=None, nearest_resistance=None)

        _, highs, lows, closes, volumes = _normalize_ohlcv(df)

        if current_price is None:
            current_price = float(closes[-1])

        # Compute ATR for adaptive bandwidth calculation
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        atr_val = float(np.mean(tr[-atr_period:])) if len(tr) >= atr_period else float(np.mean(tr))
        if atr_val <= 0:
            atr_val = max(1e-4, 0.01 * current_price)

        med_vol = float(np.median(volumes)) + 1e-9

        points: List[float] = []
        weights: List[float] = []

        # 1. Structural Pivots (2-bar lookback / lookforward fractals)
        for i in range(2, n - 2):
            recency = math.exp(0.015 * (i - n))
            vol_w = (volumes[i] / med_vol) ** 0.5

            # Swing High
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                points.append(float(highs[i]))
                weights.append(2.0 * recency * vol_w)

            # Swing Low
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                points.append(float(lows[i]))
                weights.append(2.0 * recency * vol_w)

        # 2. Secondary Touch Points (Candle Closes at alternating bars)
        for i in range(0, n, 2):
            points.append(float(closes[i]))
            weights.append(0.5 * math.exp(0.010 * (i - n)))

        if len(points) < 5:
            return KDESRResult(zones=[], nearest_support=None, nearest_resistance=None)

        pts_arr = np.array(points, dtype=np.float64)
        w_arr = np.array(weights, dtype=np.float64)
        sum_w = np.sum(w_arr)
        if sum_w <= 0:
            w_arr = np.ones_like(w_arr) / len(w_arr)
        else:
            w_arr /= sum_w

        # 3. Gaussian KDE Fitting with Adaptive ATR Bandwidth
        try:
            kde = gaussian_kde(pts_arr, weights=w_arr)
        except Exception:
            return KDESRResult(zones=[], nearest_support=None, nearest_resistance=None)

        std_p = float(np.std(pts_arr))
        if std_p > 1e-6:
            silverman_h = kde.factor * std_p * self.bandwidth_scale
            # Volatility floor: bound smoothing by 0.35 * ATR
            target_h = max(silverman_h, 0.35 * atr_val)
            kde.set_bandwidth(bw_method=target_h / std_p)
        else:
            kde.set_bandwidth(bw_method=kde.factor * self.bandwidth_scale)

        p_min = float(np.min(pts_arr) * 0.98)
        p_max = float(np.max(pts_arr) * 1.02)
        grid = np.linspace(p_min, p_max, self.num_eval_points)
        density = kde.evaluate(grid)

        max_dens = float(np.max(density))
        if max_dens <= 1e-9:
            return KDESRResult(zones=[], nearest_support=None, nearest_resistance=None)

        peak_indices, props = find_peaks(
            density,
            prominence=self.prominence_factor * max_dens,
            distance=max(5, int(self.num_eval_points / 35)),
        )

        zones: List[KDESRZone] = []
        prominences = props.get("prominences", np.zeros(len(peak_indices)))

        # Historical mid-price reference for polarity flips
        mid_idx = n // 2
        past_closes = closes[:mid_idx] if mid_idx > 0 else closes

        for idx_pos, idx in enumerate(peak_indices):
            peak_p = float(grid[idx])
            peak_d = float(density[idx])
            prom = float(prominences[idx_pos])

            # Zone boundaries at 65% of peak density
            threshold = peak_d * 0.65
            left_idx = idx
            while left_idx > 0 and density[left_idx] > threshold:
                left_idx -= 1
            right_idx = idx
            while right_idx < self.num_eval_points - 1 and density[right_idx] > threshold:
                right_idx += 1

            z_low = float(grid[left_idx])
            z_high = float(grid[right_idx])

            # Count actual price touches in zone
            touch_cnt = int(np.sum((pts_arr >= z_low) & (pts_arr <= z_high)))
            lvl_type = "RESISTANCE" if peak_p > current_price else "SUPPORT"

            # Polarity Flip Detection:
            # If peak was above past prices (was resistance), but price broke above and current price is above -> Flipped Support
            is_flipped = False
            flip_type = None
            if len(past_closes) > 0:
                past_median = float(np.median(past_closes))
                if past_median < peak_p and current_price > peak_p * 1.002:
                    is_flipped = True
                    flip_type = "FLIPPED_SUPPORT"
                elif past_median > peak_p and current_price < peak_p * 0.998:
                    is_flipped = True
                    flip_type = "FLIPPED_RESISTANCE"

            flip_multiplier = 1.25 if is_flipped else 1.0
            score = float((prom / max_dens) * math.sqrt(max(1, touch_cnt)) * flip_multiplier)

            zones.append(
                KDESRZone(
                    peak_price=round(peak_p, 2),
                    zone_low=round(z_low, 2),
                    zone_high=round(z_high, 2),
                    density=round(peak_d, 6),
                    prominence=round(prom, 6),
                    level_type=lvl_type,
                    touch_count=touch_cnt,
                    score=round(score, 3),
                    is_flipped=is_flipped,
                    flip_type=flip_type,
                )
            )

        zones.sort(key=lambda z: z.score, reverse=True)

        supports = [z for z in zones if z.level_type == "SUPPORT"]
        resistances = [z for z in zones if z.level_type == "RESISTANCE"]

        nearest_sup = max(supports, key=lambda z: z.peak_price) if supports else None
        nearest_res = min(resistances, key=lambda z: z.peak_price) if resistances else None

        return KDESRResult(
            zones=zones,
            nearest_support=nearest_sup,
            nearest_resistance=nearest_res,
            eval_grid=grid,
            density=density,
        )


def calculate_kde_sr(
    df: pd.DataFrame,
    current_price: Optional[float] = None,
    bandwidth_factor: float = 0.50,
    num_eval_points: int = 500,
) -> Any:
    """Convenience standalone wrapper for KDESupportResistanceEngine."""
    if current_price is None and len(df) > 0:
        current_price = float(df["close"].iloc[-1])
    engine = KDESupportResistanceEngine(
        bandwidth_scale=bandwidth_factor, num_eval_points=num_eval_points
    )
    res = engine.compute(df, current_price=current_price)
    return res.zones


# ============================================================================
# 5. Dynamic Fibonacci Retracement & Extensions Engine
# ============================================================================

class SwingTrend(Enum):
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"


@dataclass(frozen=True)
class FibonacciLevel:
    ratio: float
    price: float
    level_type: str  # "RETRACEMENT" or "EXTENSION"
    description: str


@dataclass
class FibonacciResult:
    trend: SwingTrend
    swing_a_price: float
    swing_b_price: float
    swing_a_idx: int
    swing_b_idx: int
    levels: List[FibonacciLevel]
    golden_pocket_low: float
    golden_pocket_high: float
    retracements: Dict[float, float] = field(default_factory=dict)
    extensions: Dict[float, float] = field(default_factory=dict)


FibonacciAnalysisResult = FibonacciResult  # Backward-compatible alias


class FibonacciEngine:
    """
    Automated Multi-Period Fractal Swing Anchoring and Fibonacci Engine.
    Computes Retracements: 0.236, 0.382, 0.500, 0.618 (Golden Pocket 0.618 - 0.650), 0.786.
    Computes Extensions: -0.272, -0.618, 1.272, 1.618, 2.000, 2.618.
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def auto_detect_swings(
        self, df: pd.DataFrame, lookback: Optional[int] = None
    ) -> Tuple[int, float, int, float, SwingTrend]:
        _, highs, lows, _, _ = _normalize_ohlcv(df)
        n = len(df)
        lb = min(n, lookback or self.lookback)

        sub_highs = highs[n - lb : n]
        sub_lows = lows[n - lb : n]

        rel_high_idx = int(np.argmax(sub_highs))
        rel_low_idx = int(np.argmin(sub_lows))

        swing_high_idx = n - lb + rel_high_idx
        swing_low_idx = n - lb + rel_low_idx

        max_h = float(highs[swing_high_idx])
        min_l = float(lows[swing_low_idx])

        # If swing high occurred after swing low -> UPTREND impulse (Low A -> High B)
        if swing_high_idx > swing_low_idx:
            return swing_low_idx, min_l, swing_high_idx, max_h, SwingTrend.UPTREND
        else:
            return swing_high_idx, max_h, swing_low_idx, min_l, SwingTrend.DOWNTREND

    def compute(
        self,
        df_or_swing_a: Union[pd.DataFrame, float],
        swing_b: Optional[float] = None,
        trend: Optional[Union[str, SwingTrend]] = None,
        override_swings: Optional[Tuple[float, float, Any]] = None,
    ) -> FibonacciResult:
        """
        Computes Fibonacci retracements and extensions.
        Supports passing a DataFrame (auto-detects swings) or explicit swing values (swing_a, swing_b, trend).
        """
        if isinstance(df_or_swing_a, pd.DataFrame):
            df = df_or_swing_a
            if override_swings is not None:
                a_p, b_p, tr_val = override_swings
                idx_a, idx_b = 0, len(df) - 1
                tr = SwingTrend[tr_val.upper()] if isinstance(tr_val, str) else tr_val
            else:
                idx_a, a_p, idx_b, b_p, tr = self.auto_detect_swings(df)
        else:
            a_p = float(df_or_swing_a)
            b_p = float(swing_b if swing_b is not None else a_p)
            idx_a, idx_b = 0, 1
            if trend is None:
                tr = SwingTrend.UPTREND if b_p >= a_p else SwingTrend.DOWNTREND
            elif isinstance(trend, str):
                tr = SwingTrend.UPTREND if trend.upper() == "UPTREND" else SwingTrend.DOWNTREND
            else:
                tr = trend

        retrace_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
        levels: List[FibonacciLevel] = []
        retracements_dict: Dict[float, float] = {}
        extensions_dict: Dict[float, float] = {}

        if tr == SwingTrend.UPTREND:
            price_range = max(1e-9, b_p - a_p)

            # Retracements: measured down from swing high B
            for r in retrace_ratios:
                p = round(b_p - r * price_range, 4)
                levels.append(FibonacciLevel(r, p, "RETRACEMENT", f"Fib {r*100:.1f}% Retracement"))
                retracements_dict[r] = p

            # Extensions: expansion targets above swing high B
            ext_definitions = [
                (-0.272, round(b_p + 0.272 * price_range, 4), "Fib -27.2% Expansion Target"),
                (1.272, round(a_p + 1.272 * price_range, 4), "Fib 127.2% Extension Target"),
                (-0.618, round(b_p + 0.618 * price_range, 4), "Fib -61.8% Golden Expansion Target"),
                (1.618, round(a_p + 1.618 * price_range, 4), "Fib 161.8% Golden Extension Target"),
                (2.000, round(a_p + 2.000 * price_range, 4), "Fib 200.0% Extension Target"),
                (2.618, round(a_p + 2.618 * price_range, 4), "Fib 261.8% Extension Target"),
            ]
            for ratio, p, desc in ext_definitions:
                levels.append(FibonacciLevel(ratio, p, "EXTENSION", desc))
                extensions_dict[ratio] = p

            gp_l = round(b_p - 0.650 * price_range, 4)
            gp_h = round(b_p - 0.618 * price_range, 4)

        else:  # DOWNTREND
            price_range = max(1e-9, a_p - b_p)

            # Retracements: measured up from swing low B
            for r in retrace_ratios:
                p = round(b_p + r * price_range, 4)
                levels.append(FibonacciLevel(r, p, "RETRACEMENT", f"Fib {r*100:.1f}% Retracement"))
                retracements_dict[r] = p

            # Extensions: expansion targets below swing low B
            ext_definitions = [
                (-0.272, round(b_p - 0.272 * price_range, 4), "Fib -27.2% Expansion Target"),
                (1.272, round(a_p - 1.272 * price_range, 4), "Fib 127.2% Extension Target"),
                (-0.618, round(b_p - 0.618 * price_range, 4), "Fib -61.8% Golden Expansion Target"),
                (1.618, round(a_p - 1.618 * price_range, 4), "Fib 161.8% Golden Extension Target"),
                (2.000, round(a_p - 2.000 * price_range, 4), "Fib 200.0% Extension Target"),
                (2.618, round(a_p - 2.618 * price_range, 4), "Fib 261.8% Extension Target"),
            ]
            for ratio, p, desc in ext_definitions:
                levels.append(FibonacciLevel(ratio, p, "EXTENSION", desc))
                extensions_dict[ratio] = p

            gp_l = round(b_p + 0.618 * price_range, 4)
            gp_h = round(b_p + 0.650 * price_range, 4)

        return FibonacciResult(
            trend=tr,
            swing_a_price=round(a_p, 4),
            swing_b_price=round(b_p, 4),
            swing_a_idx=idx_a,
            swing_b_idx=idx_b,
            levels=levels,
            golden_pocket_low=min(gp_l, gp_h),
            golden_pocket_high=max(gp_l, gp_h),
            retracements=retracements_dict,
            extensions=extensions_dict,
        )

    def calculate(
        self,
        df_or_swing_a: Union[pd.DataFrame, float],
        swing_b: Optional[float] = None,
        trend: Optional[Union[str, SwingTrend]] = None,
    ) -> FibonacciResult:
        return self.compute(df_or_swing_a, swing_b=swing_b, trend=trend)


def calculate_fibonacci(
    swing_a: Union[pd.DataFrame, float],
    swing_b: Optional[float] = None,
    trend: str = "UPTREND",
) -> FibonacciResult:
    """Convenience standalone wrapper for FibonacciEngine."""
    engine = FibonacciEngine()
    return engine.compute(swing_a, swing_b=swing_b, trend=trend)


# ============================================================================
# 6. Unified Institutional Confluence Engine
# ============================================================================

@dataclass
class ConfluenceZone:
    price_center: float
    price_low: float
    price_high: float
    confluence_score: int  # 0 - 100%
    bias: str  # "BULLISH", "BEARISH"
    alignments: List[str]
    vsa_confirmation: Optional[str]


class IndicatorConfluenceEngine:
    """
    Synthesizes independent mathematical signals from Volume Profile, VWAP,
    KDE Support/Resistance, Fibonacci, and Wyckoff VSA into unified institutional zones.
    """

    def __init__(self, tolerance_pct: float = 0.004):
        self.tolerance_pct = tolerance_pct

    def evaluate(
        self,
        current_price: float,
        vp: VolumeProfileResult,
        vwap_res: VWAPResult,
        kde_res: KDESRResult,
        fib_res: FibonacciResult,
        vsa_signals: List[VSASignal],
        tolerance_pct: Optional[float] = None,
    ) -> List[ConfluenceZone]:
        tol = tolerance_pct if tolerance_pct is not None else self.tolerance_pct
        confluence_zones: List[ConfluenceZone] = []

        latest_vwap = float(vwap_res.vwap.iloc[-1]) if len(vwap_res.vwap) > 0 else current_price
        u1 = float(vwap_res.upper_1sigma.iloc[-1]) if len(vwap_res.upper_1sigma) > 0 else latest_vwap
        l1 = float(vwap_res.lower_1sigma.iloc[-1]) if len(vwap_res.lower_1sigma) > 0 else latest_vwap
        u2 = float(vwap_res.upper_2sigma.iloc[-1]) if len(vwap_res.upper_2sigma) > 0 else latest_vwap
        l2 = float(vwap_res.lower_2sigma.iloc[-1]) if len(vwap_res.lower_2sigma) > 0 else latest_vwap

        recent_vsa = vsa_signals[-1] if vsa_signals else None

        for z in kde_res.zones:
            score = 25  # Base score for valid KDE peak
            alignments = [f"KDE {z.level_type} ({z.touch_count} touches)"]
            p = z.peak_price

            # 1. Volume Profile Alignment
            if abs(p - vp.poc_price) / max(1e-9, p) <= tol:
                score += 25
                alignments.append(f"Volume Profile POC ({vp.poc_price:.2f})")
            elif abs(p - vp.vah_price) / max(1e-9, p) <= tol:
                score += 20
                alignments.append(f"Volume Profile VAH ({vp.vah_price:.2f})")
            elif abs(p - vp.val_price) / max(1e-9, p) <= tol:
                score += 20
                alignments.append(f"Volume Profile VAL ({vp.val_price:.2f})")

            # 2. Fibonacci Alignment
            if fib_res.golden_pocket_low <= p <= fib_res.golden_pocket_high:
                score += 20
                alignments.append(f"Fib Golden Pocket [{fib_res.golden_pocket_low:.2f} - {fib_res.golden_pocket_high:.2f}]")
            else:
                for fl in fib_res.levels:
                    if abs(p - fl.price) / max(1e-9, p) <= tol:
                        score += 15
                        alignments.append(f"{fl.description} ({fl.price:.2f})")
                        break

            # 3. VWAP Alignment
            if abs(p - latest_vwap) / max(1e-9, p) <= tol:
                score += 15
                alignments.append(f"VWAP Core ({latest_vwap:.2f})")
            elif abs(p - l1) / max(1e-9, p) <= tol or abs(p - u1) / max(1e-9, p) <= tol:
                score += 12
                alignments.append("VWAP +-1σ Value Band")
            elif abs(p - l2) / max(1e-9, p) <= tol or abs(p - u2) / max(1e-9, p) <= tol:
                score += 15
                alignments.append("VWAP +-2σ Exhaustion Band")

            # 4. VSA Confirmation
            vsa_text = None
            if recent_vsa and abs(current_price - p) / max(1e-9, p) <= tol * 1.5:
                score += 15
                vsa_text = f"{recent_vsa.signal_type.value} ({recent_vsa.bias})"
                alignments.append(f"VSA Confirmation: {vsa_text}")

            bias = "BULLISH" if z.level_type == "SUPPORT" else "BEARISH"
            confluence_zones.append(
                ConfluenceZone(
                    price_center=p,
                    price_low=z.zone_low,
                    price_high=z.zone_high,
                    confluence_score=min(100, score),
                    bias=bias,
                    alignments=alignments,
                    vsa_confirmation=vsa_text,
                )
            )

        confluence_zones.sort(key=lambda x: x.confluence_score, reverse=True)
        return confluence_zones


def evaluate_confluence(
    current_price: float,
    vp: VolumeProfileResult,
    vwap_res: VWAPResult,
    kde_res: KDESRResult,
    fib_res: FibonacciResult,
    vsa_signals: List[VSASignal],
    tolerance_pct: float = 0.004,
) -> List[ConfluenceZone]:
    """Convenience standalone wrapper for IndicatorConfluenceEngine."""
    engine = IndicatorConfluenceEngine(tolerance_pct=tolerance_pct)
    return engine.evaluate(
        current_price=current_price,
        vp=vp,
        vwap_res=vwap_res,
        kde_res=kde_res,
        fib_res=fib_res,
        vsa_signals=vsa_signals,
        tolerance_pct=tolerance_pct,
    )


# ============================================================================
# 7. TTM Squeeze & Volatility Compression Engine
# ============================================================================

@dataclass(frozen=True)
class TTMSqueezeResult:
    is_squeeze_on: bool
    momentum: float
    momentum_direction: str       # "BULLISH", "BEARISH", "NEUTRAL"
    bb_upper: float
    bb_lower: float
    kc_upper: float
    kc_lower: float
    squeeze_bars: int


class TTMSqueezeEngine:
    """
    Detects institutional volatility compression prior to large breakout expansions
    by evaluating whether Bollinger Bands (20, 2.0) are compressed inside Keltner Channels (20, 1.5).
    """
    def __init__(self, bb_length: int = 20, bb_mult: float = 2.0, kc_length: int = 20, kc_mult: float = 1.5):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult

    def compute(self, df: pd.DataFrame) -> TTMSqueezeResult:
        opens, highs, lows, closes, volumes = _normalize_ohlcv(df)
        n = len(closes)
        if n < max(self.bb_length, self.kc_length):
            return TTMSqueezeResult(
                is_squeeze_on=False,
                momentum=0.0,
                momentum_direction="NEUTRAL",
                bb_upper=float(closes[-1]) if n > 0 else 0.0,
                bb_lower=float(closes[-1]) if n > 0 else 0.0,
                kc_upper=float(closes[-1]) if n > 0 else 0.0,
                kc_lower=float(closes[-1]) if n > 0 else 0.0,
                squeeze_bars=0,
            )

        close_series = pd.Series(closes)
        high_series = pd.Series(highs)
        low_series = pd.Series(lows)

        # 1. Bollinger Bands
        sma = close_series.rolling(self.bb_length).mean()
        std = close_series.rolling(self.bb_length).std()
        bb_upper = sma + (self.bb_mult * std)
        bb_lower = sma - (self.bb_mult * std)

        # 2. Keltner Channels (EMA + ATR)
        ema = close_series.ewm(span=self.kc_length, adjust=False).mean()
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        tr = np.insert(tr, 0, highs[0] - lows[0])
        atr = pd.Series(tr).rolling(self.kc_length).mean()
        kc_upper = ema + (self.kc_mult * atr)
        kc_lower = ema - (self.kc_mult * atr)

        # Squeeze condition: BB inside KC
        squeeze_series = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        is_squeeze_on = bool(squeeze_series.iloc[-1])

        # Count consecutive squeeze bars
        squeeze_bars = 0
        for val in reversed(squeeze_series.values):
            if val:
                squeeze_bars += 1
            else:
                break

        # Momentum Histogram: Linear regression of price delta from mean of donchian midline and SMA
        highest_high = high_series.rolling(self.kc_length).max()
        lowest_low = low_series.rolling(self.kc_length).min()
        donchian_mid = (highest_high + lowest_low) / 2.0
        delta = closes[-1] - ((donchian_mid.iloc[-1] + sma.iloc[-1]) / 2.0)
        momentum = float(delta)

        if momentum > 0:
            mom_dir = "BULLISH"
        elif momentum < 0:
            mom_dir = "BEARISH"
        else:
            mom_dir = "NEUTRAL"

        return TTMSqueezeResult(
            is_squeeze_on=is_squeeze_on,
            momentum=momentum,
            momentum_direction=mom_dir,
            bb_upper=float(bb_upper.iloc[-1]),
            bb_lower=float(bb_lower.iloc[-1]),
            kc_upper=float(kc_upper.iloc[-1]),
            kc_lower=float(kc_lower.iloc[-1]),
            squeeze_bars=squeeze_bars,
        )


# ============================================================================
# 8. Choppiness Index (CHOP) Engine
# ============================================================================

@dataclass(frozen=True)
class ChoppinessResult:
    chop_index: float
    is_choppy: bool               # True if CHOP >= 61.8 (consolidating / chop gate)
    is_trending: bool             # True if CHOP <= 38.2 (strong directional impulse)
    market_state: str             # "CHOPPY", "TRENDING", "TRANSITIONAL"


class ChoppinessIndexEngine:
    """
    Calculates the Choppiness Index (CHOP) to quantify trendiness vs congestion.
    CHOP > 61.8: Market is sideways/choppy (filter out breakout trades).
    CHOP < 38.2: Strong institutional trend expansion.
    """
    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, df: pd.DataFrame) -> ChoppinessResult:
        opens, highs, lows, closes, volumes = _normalize_ohlcv(df)
        n = len(closes)
        if n <= self.period:
            return ChoppinessResult(
                chop_index=50.0,
                is_choppy=False,
                is_trending=False,
                market_state="TRANSITIONAL",
            )

        # True Range
        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        tr = np.insert(tr, 0, highs[0] - lows[0])
        tr_series = pd.Series(tr)
        high_series = pd.Series(highs)
        low_series = pd.Series(lows)

        sum_tr = tr_series.rolling(self.period).sum().iloc[-1]
        max_high = high_series.rolling(self.period).max().iloc[-1]
        min_low = low_series.rolling(self.period).min().iloc[-1]
        range_hl = max_high - min_low

        if range_hl <= 0 or sum_tr <= 0:
            chop = 50.0
        else:
            ratio = sum_tr / range_hl
            chop = 100.0 * (math.log10(ratio) / math.log10(self.period))

        chop = max(0.0, min(100.0, float(chop)))
        if chop >= 61.8:
            state = "CHOPPY"
            is_c = True
            is_t = False
        elif chop <= 38.2:
            state = "TRENDING"
            is_c = False
            is_t = True
        else:
            state = "TRANSITIONAL"
            is_c = False
            is_t = False

        return ChoppinessResult(
            chop_index=chop,
            is_choppy=is_c,
            is_trending=is_t,
            market_state=state,
        )

