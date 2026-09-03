import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

# ==========================================
# 1. Volume Profile
# ==========================================
@dataclass
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

def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 0.70
) -> VolumeProfileResult:
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values

    min_p = float(np.min(lows))
    max_p = float(np.max(highs))
    if max_p == min_p:
        max_p += 1e-6

    bin_edges = np.linspace(min_p, max_p, num_bins + 1)
    bin_volumes = np.zeros(num_bins)

    for h, l, v in zip(highs, lows, volumes):
        c_range = h - l
        if c_range <= 1e-9:
            idx = np.clip(np.searchsorted(bin_edges, l, side="right") - 1, 0, num_bins - 1)
            bin_volumes[idx] += v
            continue

        for b_idx in range(num_bins):
            b_low = bin_edges[b_idx]
            b_high = bin_edges[b_idx + 1]
            overlap = max(0.0, min(h, b_high) - max(l, b_low))
            if overlap > 0:
                bin_volumes[b_idx] += v * (overlap / c_range)

    total_vol = float(np.sum(bin_volumes))
    if total_vol <= 0:
        total_vol = 1e-9

    poc_idx = int(np.argmax(bin_volumes))
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0

    target_va_vol = value_area_pct * total_vol
    accum_vol = bin_volumes[poc_idx]
    up_idx = poc_idx
    down_idx = poc_idx

    while accum_vol < target_va_vol and (up_idx < num_bins - 1 or down_idx > 0):
        next_up_vol = bin_volumes[up_idx + 1] if up_idx + 1 < num_bins else 0.0
        next_down_vol = bin_volumes[down_idx - 1] if down_idx - 1 >= 0 else 0.0

        if next_up_vol > next_down_vol:
            up_idx += 1
            accum_vol += next_up_vol
        elif next_down_vol > next_up_vol:
            down_idx -= 1
            accum_vol += next_down_vol
        else:
            if up_idx + 1 < num_bins:
                up_idx += 1
                accum_vol += next_up_vol
            if down_idx - 1 >= 0 and accum_vol < target_va_vol:
                down_idx -= 1
                accum_vol += next_down_vol

    val_price = bin_edges[down_idx]
    vah_price = bin_edges[up_idx + 1]

    # HVN & LVN Detection
    vol_mean = np.mean(bin_volumes)
    vol_std = np.std(bin_volumes)
    peaks, _ = find_peaks(bin_volumes, distance=3, prominence=0.2 * vol_std if vol_std > 0 else None)
    valleys, _ = find_peaks(-bin_volumes, distance=3, prominence=0.2 * vol_std if vol_std > 0 else None)

    hvn_levels = [float((bin_edges[p] + bin_edges[p + 1]) / 2.0) for p in peaks if bin_volumes[p] > vol_mean]
    lvn_levels = [float((bin_edges[v] + bin_edges[v + 1]) / 2.0) for v in valleys if bin_volumes[v] < vol_mean]

    price_bins = [
        PriceBin(
            price_low=float(bin_edges[i]),
            price_high=float(bin_edges[i + 1]),
            price_center=float((bin_edges[i] + bin_edges[i + 1]) / 2.0),
            volume=float(bin_volumes[i]),
            pct_of_total=float(bin_volumes[i] / total_vol)
        )
        for i in range(num_bins)
    ]

    return VolumeProfileResult(
        poc_price=poc_price,
        vah_price=vah_price,
        val_price=val_price,
        total_volume=total_vol,
        bins=price_bins,
        hvn_levels=hvn_levels,
        lvn_levels=lvn_levels
    )

# ==========================================
# 2. VWAP & Standard Deviation Bands
# ==========================================
@dataclass
class VWAPResult:
    vwap: pd.Series
    upper_1sigma: pd.Series
    lower_1sigma: pd.Series
    upper_2sigma: pd.Series
    lower_2sigma: pd.Series
    std_dev: pd.Series

def calculate_vwap(
    df: pd.DataFrame,
    anchor_col: Optional[str] = None
) -> VWAPResult:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]

    if anchor_col is not None and anchor_col in df.columns:
        groups = df.groupby(df[anchor_col])
        cum_tp_vol = groups.apply(lambda g: (tp.loc[g.index] * vol.loc[g.index]).cumsum()).reset_index(level=0, drop=True)
        cum_vol = groups.apply(lambda g: vol.loc[g.index].cumsum()).reset_index(level=0, drop=True)
    else:
        cum_tp_vol = (tp * vol).cumsum()
        cum_vol = vol.cumsum().replace(0, 1e-9)

    vwap = cum_tp_vol / cum_vol

    if anchor_col is not None and anchor_col in df.columns:
        groups = df.groupby(df[anchor_col])
        cum_tp2_vol = groups.apply(lambda g: (tp.loc[g.index]**2 * vol.loc[g.index]).cumsum()).reset_index(level=0, drop=True)
    else:
        cum_tp2_vol = (tp**2 * vol).cumsum()

    variance = (cum_tp2_vol / cum_vol) - (vwap**2)
    variance = np.maximum(variance, 0.0)
    std_dev = np.sqrt(variance)

    return VWAPResult(
        vwap=vwap,
        upper_1sigma=vwap + 1.0 * std_dev,
        lower_1sigma=vwap - 1.0 * std_dev,
        upper_2sigma=vwap + 2.0 * std_dev,
        lower_2sigma=vwap - 2.0 * std_dev,
        std_dev=std_dev
    )

# ==========================================
# 3. Wyckoff Volume Spread Analysis (VSA)
# ==========================================
class VSASignalType(Enum):
    ABSORPTION = "ABSORPTION"
    STOPPING_VOLUME = "STOPPING_VOLUME"
    NO_SUPPLY = "NO_SUPPLY"
    NO_DEMAND = "NO_DEMAND"
    BUYING_CLIMAX = "BUYING_CLIMAX"
    SELLING_CLIMAX = "SELLING_CLIMAX"
    UPTHRUST = "UPTHRUST"
    SPRING = "SPRING"
    NEUTRAL = "NEUTRAL"

@dataclass
class VSASignal:
    bar_index: int
    signal_type: VSASignalType
    bias: str
    rvol: float
    spread_ratio: float
    close_location: float
    description: str

def analyze_vsa(df: pd.DataFrame, lookback: int = 20) -> List[VSASignal]:
    signals: List[VSASignal] = []
    if len(df) < lookback + 2:
        return signals

    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values
    volumes = df["volume"].values

    spreads = highs - lows
    vol_series = pd.Series(volumes)
    spread_series = pd.Series(spreads)

    vol_sma = vol_series.rolling(window=lookback, min_periods=5).mean().values
    spread_sma = spread_series.rolling(window=lookback, min_periods=5).mean().values

    for i in range(lookback, len(df)):
        c_spread = spreads[i]
        c_range = max(c_spread, 1e-9)
        c_vol = volumes[i]
        c_close = closes[i]
        p_close = closes[i - 1]

        rvol = c_vol / (vol_sma[i] + 1e-9)
        spread_ratio = c_spread / (spread_sma[i] + 1e-9)
        close_loc = (c_close - lows[i]) / c_range

        is_downtrend = closes[i - 1] < closes[i - 5]
        is_uptrend = closes[i - 1] > closes[i - 5]

        if is_downtrend and rvol >= 2.0 and spread_ratio <= 1.1 and close_loc >= 0.45:
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.STOPPING_VOLUME,
                bias="BULLISH",
                rvol=rvol,
                spread_ratio=spread_ratio,
                close_location=close_loc,
                description=f"Stopping volume: Smart money absorption (RVOL={rvol:.2f})"
            ))
        elif c_close <= p_close and rvol <= 0.75 and c_vol < volumes[i - 1] and c_vol < volumes[i - 2] and spread_ratio <= 0.9:
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.NO_SUPPLY,
                bias="BULLISH",
                rvol=rvol,
                spread_ratio=spread_ratio,
                close_location=close_loc,
                description=f"No Supply: Volume dried up on down bar (RVOL={rvol:.2f})"
            ))
        elif c_close >= p_close and rvol <= 0.75 and c_vol < volumes[i - 1] and c_vol < volumes[i - 2] and spread_ratio <= 0.9:
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.NO_DEMAND,
                bias="BEARISH",
                rvol=rvol,
                spread_ratio=spread_ratio,
                close_location=close_loc,
                description=f"No Demand: Lack of buyers on up bar (RVOL={rvol:.2f})"
            ))
        elif is_uptrend and rvol >= 2.5 and spread_ratio >= 1.6 and close_loc <= 0.60:
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.BUYING_CLIMAX,
                bias="BEARISH",
                rvol=rvol,
                spread_ratio=spread_ratio,
                close_location=close_loc,
                description=f"Buying Climax: Distribution on ultra-high volume (RVOL={rvol:.2f})"
            ))
        elif is_downtrend and rvol >= 2.5 and spread_ratio >= 1.6 and close_loc >= 0.40:
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.SELLING_CLIMAX,
                bias="BULLISH",
                rvol=rvol,
                spread_ratio=spread_ratio,
                close_location=close_loc,
                description=f"Selling Climax: Panic selling absorbed (RVOL={rvol:.2f})"
            ))

    return signals

# ==========================================
# 4. KDE Support & Resistance
# ==========================================
@dataclass
class KDESRZone:
    peak_price: float
    zone_low: float
    zone_high: float
    density: float
    prominence: float
    level_type: str
    touch_count: int

def calculate_kde_sr(
    df: pd.DataFrame,
    current_price: float,
    bandwidth_factor: float = 0.5,
    num_eval_points: int = 500
) -> List[KDESRZone]:
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    pivot_points = []
    pivot_weights = []

    for i in range(2, n - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            pivot_points.append(highs[i])
            recency = np.exp(0.02 * (i - n))
            pivot_weights.append(1.5 * recency)
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            pivot_points.append(lows[i])
            recency = np.exp(0.02 * (i - n))
            pivot_weights.append(1.5 * recency)

    for i in range(0, n, 2):
        pivot_points.append(closes[i])
        pivot_weights.append(0.5 * np.exp(0.01 * (i - n)))

    if len(pivot_points) < 5:
        return []

    pts = np.array(pivot_points)
    weights = np.array(pivot_weights)
    weights = weights / np.sum(weights)

    kde = gaussian_kde(pts, weights=weights)
    kde.set_bandwidth(bw_method=kde.factor * bandwidth_factor)

    p_min = float(np.min(pts) * 0.98)
    p_max = float(np.max(pts) * 1.02)
    eval_grid = np.linspace(p_min, p_max, num_eval_points)
    density = kde.evaluate(eval_grid)

    peak_indices, properties = find_peaks(
        density,
        prominence=0.15 * np.max(density),
        distance=int(num_eval_points / 30)
    )

    zones: List[KDESRZone] = []
    for idx in peak_indices:
        peak_p = float(eval_grid[idx])
        peak_d = float(density[idx])
        prom_arr = properties["prominences"]
        prom = float(prom_arr[np.where(peak_indices == idx)[0][0]])

        half_height = peak_d * 0.65
        left_idx = idx
        while left_idx > 0 and density[left_idx] > half_height:
            left_idx -= 1
        right_idx = idx
        while right_idx < num_eval_points - 1 and density[right_idx] > half_height:
            right_idx += 1

        zone_low = float(eval_grid[left_idx])
        zone_high = float(eval_grid[right_idx])

        touches = int(np.sum((pts >= zone_low) & (pts <= zone_high)))
        lvl_type = "RESISTANCE" if peak_p > current_price else "SUPPORT"

        zones.append(KDESRZone(
            peak_price=peak_p,
            zone_low=zone_low,
            zone_high=zone_high,
            density=peak_d,
            prominence=prom,
            level_type=lvl_type,
            touch_count=touches
        ))

    zones.sort(key=lambda z: z.prominence * z.touch_count, reverse=True)
    return zones

# ==========================================
# 5. Fibonacci Retracement & Extensions
# ==========================================
@dataclass
class FibonacciLevel:
    ratio: float
    price: float
    level_type: str
    description: str

@dataclass
class FibonacciAnalysisResult:
    trend: str
    swing_a_price: float
    swing_b_price: float
    levels: List[FibonacciLevel]
    golden_pocket_low: float
    golden_pocket_high: float

def calculate_fibonacci(
    swing_a: float,
    swing_b: float,
    trend: str = "UPTREND"
) -> FibonacciAnalysisResult:
    retrace_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    extension_ratios = [1.272, 1.618]
    levels: List[FibonacciLevel] = []

    if trend.upper() == "UPTREND":
        price_range = swing_b - swing_a
        for r in retrace_ratios:
            lvl_price = swing_b - r * price_range
            levels.append(FibonacciLevel(
                ratio=r,
                price=lvl_price,
                level_type="RETRACEMENT",
                description=f"Fib {r*100:.1f}% Retracement"
            ))
        for e in extension_ratios:
            lvl_price = swing_a + e * price_range
            levels.append(FibonacciLevel(
                ratio=e,
                price=lvl_price,
                level_type="EXTENSION",
                description=f"Fib {e*100:.1f}% Extension Target"
            ))
        gp_low = swing_b - 0.650 * price_range
        gp_high = swing_b - 0.618 * price_range
    else:
        price_range = swing_a - swing_b
        for r in retrace_ratios:
            lvl_price = swing_b + r * price_range
            levels.append(FibonacciLevel(
                ratio=r,
                price=lvl_price,
                level_type="RETRACEMENT",
                description=f"Fib {r*100:.1f}% Retracement"
            ))
        for e in extension_ratios:
            lvl_price = swing_a - e * price_range
            levels.append(FibonacciLevel(
                ratio=e,
                price=lvl_price,
                level_type="EXTENSION",
                description=f"Fib {e*100:.1f}% Extension Target"
            ))
        gp_low = swing_b + 0.618 * price_range
        gp_high = swing_b + 0.650 * price_range

    return FibonacciAnalysisResult(
        trend=trend.upper(),
        swing_a_price=swing_a,
        swing_b_price=swing_b,
        levels=levels,
        golden_pocket_low=min(gp_low, gp_high),
        golden_pocket_high=max(gp_low, gp_high)
    )

def create_sample_df():
    np.random.seed(42)
    n = 100
    prices = 60000.0 + np.cumsum(np.random.randn(n) * 100)
    highs = prices + np.abs(np.random.randn(n) * 80)
    lows = prices - np.abs(np.random.randn(n) * 80)
    opens = (highs + lows) / 2.0 + np.random.randn(n) * 20
    closes = prices
    volumes = np.random.uniform(50, 300, n)
    volumes[40] = 950.0  # stopping volume
    volumes[85] = 1100.0 # climax volume

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })

def test_volume_profile():
    df = create_sample_df()
    vp = calculate_volume_profile(df, num_bins=40)
    assert vp.poc_price > 0
    assert vp.val_price <= vp.poc_price <= vp.vah_price
    assert vp.total_volume > 0
    assert len(vp.bins) == 40
    # Value area must contain approx 70% of volume
    va_bins = [b for b in vp.bins if vp.val_price <= b.price_center <= vp.vah_price]
    va_volume = sum(b.volume for b in va_bins)
    assert va_volume >= 0.65 * vp.total_volume

def test_vwap():
    df = create_sample_df()
    vw = calculate_vwap(df)
    assert len(vw.vwap) == len(df)
    # Check sigma ordering
    assert (vw.lower_2sigma <= vw.lower_1sigma).all()
    assert (vw.lower_1sigma <= vw.vwap).all()
    assert (vw.vwap <= vw.upper_1sigma).all()
    assert (vw.upper_1sigma <= vw.upper_2sigma).all()

def test_vsa():
    df = create_sample_df()
    signals = analyze_vsa(df)
    assert len(signals) > 0
    types = [s.signal_type for s in signals]
    assert VSASignalType.STOPPING_VOLUME in types or VSASignalType.NO_SUPPLY in types or VSASignalType.NO_DEMAND in types

def test_kde_sr():
    df = create_sample_df()
    curr_p = df["close"].iloc[-1]
    zones = calculate_kde_sr(df, curr_p)
    assert len(zones) > 0
    for z in zones:
        assert z.zone_low <= z.peak_price <= z.zone_high
        assert z.touch_count >= 0
        assert z.level_type in ["SUPPORT", "RESISTANCE"]

def test_fibonacci():
    df = create_sample_df()
    low_p = float(df["low"].min())
    high_p = float(df["high"].max())
    fib = calculate_fibonacci(low_p, high_p, "UPTREND")
    assert fib.trend == "UPTREND"
    assert len(fib.levels) == 7
    assert fib.golden_pocket_low < fib.golden_pocket_high
    assert low_p <= fib.golden_pocket_low <= high_p

if __name__ == "__main__":
    test_volume_profile()
    test_vwap()
    test_vsa()
    test_kde_sr()
    test_fibonacci()
    print("ALL 5 INDICATOR TESTS PASSED SUCCESSFULLY!")

