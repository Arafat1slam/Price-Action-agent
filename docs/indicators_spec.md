# Advanced Volume & Indicator Technical Specification

**Author**: Subagent 3 — Volume & Indicators Researcher  
**Target Engine**: Quantitative Price Action Analysis Engine  
**Language**: Python 3.10+ / NumPy / Pandas / SciPy  
**Date**: September 2026  

---

## Table of Contents
1. [Executive Summary & System Architecture](#1-executive-summary--system-architecture)
2. [Module 1: Volume Profile & Auction Market Theory](#2-module-1-volume-profile--auction-market-theory)
   - [2.1 Mathematical Formulation & Price Binning](#21-mathematical-formulation--price-binning)
   - [2.2 Point of Control (POC) & Virgin POC](#22-point-of-control-poc--virgin-poc)
   - [2.3 Value Area (VAH & VAL) Expansion Algorithm](#23-value-area-vah--val-expansion-algorithm)
   - [2.4 High Volume Nodes (HVN) & Low Volume Nodes (LVN)](#24-high-volume-nodes-hvn--low-volume-nodes-lvn)
   - [2.5 Data Structures & Python Implementation](#25-data-structures--python-implementation)
3. [Module 2: VWAP & Standard Deviation Bands](#3-module-2-vwap--standard-deviation-bands)
   - [3.1 Mathematical Formulation & Typical Price](#31-mathematical-formulation--typical-price)
   - [3.2 Running Variance & Sigma Bands](#32-running-variance--sigma-bands)
   - [3.3 Anchoring Modalities: Session, Rolling, & Anchored (AVWAP)](#33-anchoring-modalities-session-rolling--anchored-avwap)
   - [3.4 Statistical Exhaustion & Mean Reversion Mechanics](#34-statistical-exhaustion--mean-reversion-mechanics)
   - [3.5 Data Structures & Python Implementation](#35-data-structures--python-implementation)
4. [Module 3: Wyckoff Volume Spread Analysis (VSA)](#4-module-3-wyckoff-volume-spread-analysis-vsa)
   - [4.1 Theoretical Foundations: Effort vs. Result](#41-theoretical-foundations-effort-vs-result)
   - [4.2 Normalized Metrics: RVOL, Spread Ratio, & Close Location](#42-normalized-metrics-rvol-spread-ratio--close-location)
   - [4.3 Core VSA Patterns & Algorithmic Rules](#43-core-vsa-patterns--algorithmic-rules)
     - [Stopping Volume / Absorption](#431-stopping-volume--absorption)
     - [No Supply / No Demand (Testing S/R)](#432-no-supply--no-demand-testing-sr)
     - [Climax Volume (Buying / Selling Exhaustion)](#433-climax-volume-buying--selling-exhaustion)
     - [Upthrust & Spring (Liquidity Sweeps)](#434-upthrust--spring-liquidity-sweeps)
   - [4.4 Data Structures & Python Implementation](#44-data-structures--python-implementation)
5. [Module 4: Kernel Density Estimation (KDE) Support & Resistance](#5-module-4-kernel-density-estimation-kde-support--resistance)
   - [5.1 Continuous Non-Parametric Density vs. Discrete Clustering](#51-continuous-non-parametric-density-vs-discrete-clustering)
   - [5.2 Kernel Formulation & Bandwidth Selection](#52-kernel-formulation--bandwidth-selection)
   - [5.3 Touch Point Extraction & Multi-Factor Weighting](#53-touch-point-extraction--multi-factor-weighting)
   - [5.4 Peak Finding, Prominence, & S/R Zone Extraction](#54-peak-finding-prominence--sr-zone-extraction)
   - [5.5 Polarity Flips & Support/Resistance Classification](#55-polarity-flips--supportresistance-classification)
   - [5.6 Data Structures & Python Implementation](#56-data-structures--python-implementation)
6. [Module 5: Fibonacci Retracement & Extensions](#6-module-5-fibonacci-retracement--extensions)
   - [5.1 Mathematical Rationale & The Golden Ratio](#61-mathematical-rationale--the-golden-ratio)
   - [6.2 Retracement & Extension Formulations](#62-retracement--extension-formulations)
   - [6.3 The Golden Pocket (0.618 - 0.65 / 0.786)](#63-the-golden-pocket-0618---065--0786)
   - [6.4 Automated Swing Anchor Detection (Multi-Period Fractals)](#64-automated-swing-anchor-detection-multi-period-fractals)
   - [6.5 Data Structures & Python Implementation](#65-data-structures--python-implementation)
7. [Module 6: Unified Indicator Confluence Engine](#7-module-6-unified-indicator-confluence-engine)
   - [7.1 Confluence Matrix & Composite Scoring Algorithm](#71-confluence-matrix--composite-scoring-algorithm)
   - [7.2 Data Structures & Integration Pipeline](#72-data-structures--integration-pipeline)
8. [Module 7: Verification & Test Suite](#8-module-7-verification--test-suite)

---

## 1. Executive Summary & System Architecture

In modern market microstructure, price action alone reveals *what* the market did, but volume reveals *how much energy* was expended to do it. Price movement without institutional volume commitment is inherently fragile, whereas heavy institutional volume absorbs liquidity, shifts fair value, and establishes durable structural support and resistance.

This specification details five interconnected mathematical models designed to operate as a cohesive quantitative intelligence layer:

```
+-------------------------------------------------------------------------------+
|                             OHLCV Candlestick Feed                            |
+-------------------------------------------------------------------------------+
           |                     |                     |                  |
           v                     v                     v                  v
+--------------------+ +--------------------+ +-----------------+ +-------------+
| 1. Volume Profile  | |      2. VWAP       | |   3. Wyckoff    | |   4. KDE    |
| - Price Bins       | | - Cumulative / Dev | |      VSA        | |  Gaussian   |
| - POC / VAH / VAL  | | - Bands (+-1/2/3s) | | - Absorption    | |  Density    |
| - HVN / LVN Nodes  | | - Mean Reversion   | | - No Supply/Dem | |  Pivots     |
+--------------------+ +--------------------+ +-----------------+ +-------------+
           \                     /                     /                  /
            \                   /                     /                  /
             +-----------------+---------------------+------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |     5. Fibonacci Anchor Engine    |
                     | - 0.236, 0.382, 0.50, 0.618, 0.786|
                     | - Golden Pocket & 1.272/1.618 Ext |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     | 6. Institutional Confluence Engine|
                     | - Level Clustering & Zone Merging |
                     | - Probability Weighting (0 - 100%)|
                     | - Trade Bias & Target Generation  |
                     +-----------------------------------+
```

---

## 2. Module 1: Volume Profile & Auction Market Theory

### 2.1 Mathematical Formulation & Price Binning

Volume Profile distributes trading volume across price space rather than across time. In classical Auction Market Theory (AMT, Steidlmayer), price functions as an advertising mechanism, time regulates opportunity, and volume measures institutional acceptance or rejection.

When processing OHLCV data without sub-second Level 2 tick logs, attributing the entire candle volume $V_i$ to a single point (such as the Close or Typical Price) creates severe discretization bias. Instead, we model the trading within candle $i$ as a continuous uniform distribution across its range $[L_i, H_i]$:

Given a rolling window of $N$ candles, let:
- $P_{\min} = \min_{1 \le i \le N} (L_i)$
- $P_{\max} = \max_{1 \le i \le N} (H_i)$
- Total price span: $\Delta_{\text{total}} = P_{\max} - P_{\min}$

We divide $[P_{\min}, P_{\max}]$ into $B$ equidistant price bins:
$$\text{Bin Edge } b_k = P_{\min} + k \cdot \frac{\Delta_{\text{total}}}{B}, \quad k \in [0, B]$$
$$\text{Bin Interval } I_k = [b_k, b_{k+1}), \quad k \in [0, B-1]$$
$$\text{Bin Center } p_k = \frac{b_k + b_{k+1}}{2}$$

For each candle $i$ with volume $V_i$ and price range $R_i = \max(H_i - L_i, \epsilon)$, the volume contributed by candle $i$ to bin $k$ is proportional to the geometric overlap:
$$\text{Overlap}(I_k, [L_i, H_i]) = \max\left(0, \min(H_i, b_{k+1}) - \max(L_i, b_k)\right)$$
$$V_{i, k} = V_i \times \frac{\text{Overlap}(I_k, [L_i, H_i])}{R_i}$$

The aggregated volume $V(k)$ in bin $k$ across the window is:
$$V(k) = \sum_{i=1}^N V_{i, k}$$
$$\text{Total Window Volume } V_{\text{total}} = \sum_{k=0}^{B-1} V(k)$$

### 2.2 Point of Control (POC) & Virgin POC

The **Point of Control (POC)** represents the price bin where the largest volume of contracts changed hands during the window, establishing the absolute center of fair value:
$$k_{\text{poc}} = \arg\max_{0 \le k < B} V(k)$$
$$\text{POC Price} = p_{k_{\text{poc}}} = \frac{b_{k_{\text{poc}}} + b_{k_{\text{poc}}+1}}{2}$$

**Tie-Breaking Rule**: If $\exists k_1 \ne k_2$ such that $V(k_1) = V(k_2) = \max_k V(k)$, the POC is assigned to the bin closest to the volume-weighted centroid $\bar{p} = \frac{\sum_k p_k V(k)}{V_{\text{total}}}$.

**Virgin / Naked POC (vPOC)**:
A historical POC remains "Virgin" (vPOC) as long as subsequent price action has not touched or crossed $[b_{k_{\text{poc}}}, b_{k_{\text{poc}}+1}]$. vPOCs exert a powerful gravitational pull on price, acting as high-probability magnet targets when market balance shifts.

### 2.3 Value Area (VAH & VAL) Expansion Algorithm

The **Value Area (VA)** represents the price range surrounding the POC that accounts for 70% of total traded volume (conventionally approximating one standard deviation $\approx 68.27\%$ under a normal distribution).

#### Algorithmic Procedure (Steidlmayer 70% Expansion):
1. **Initialize**:
   - Target volume: $V_{\text{target}} = 0.70 \times V_{\text{total}}$.
   - Running accumulated volume: $V_{\text{accum}} = V(k_{\text{poc}})$.
   - Upper pointer: $u = k_{\text{poc}} + 1$.
   - Lower pointer: $d = k_{\text{poc}} - 1$.
2. **Iterative Dual-Bin Expansion**:
   While $V_{\text{accum}} < V_{\text{target}}$ and $(u < B \text{ or } d \ge 0)$:
   - Compute potential upward volume:
     $$V_{\text{up}} = \begin{cases} V(u) + V(u+1) & \text{if } u+1 < B \\ V(u) & \text{if } u < B \\ 0 & \text{otherwise} \end{cases}$$
   - Compute potential downward volume:
     $$V_{\text{down}} = \begin{cases} V(d) + V(d-1) & \text{if } d-1 \ge 0 \\ V(d) & \text{if } d \ge 0 \\ 0 & \text{otherwise} \end{cases}$$
   - If $V_{\text{up}} > V_{\text{down}}$:
     - $V_{\text{accum}} \leftarrow V_{\text{accum}} + V(u)$
     - $u \leftarrow u + 1$
   - Else if $V_{\text{down}} > V_{\text{up}}$:
     - $V_{\text{accum}} \leftarrow V_{\text{accum}} + V(d)$
     - $d \leftarrow d - 1$
   - Else (tie):
     - Expand both: $V_{\text{accum}} \leftarrow V_{\text{accum}} + V(u) + V(d)$; $u \leftarrow u + 1$; $d \leftarrow d - 1$.
3. **Set Boundaries**:
   $$\text{Value Area High (VAH)} = b_{u}$$
   $$\text{Value Area Low (VAL)} = b_{d+1}$$

### 2.4 High Volume Nodes (HVN) & Low Volume Nodes (LVN)

- **High Volume Node (HVN)**: Local peaks in the volume distribution where $V(k) > \mu_V + 0.5 \sigma_V$ and $V(k) > \max(V(k-1), V(k+1))$. HVNs denote price levels of acceptance where buyers and sellers reached consensus. They act as strong support and resistance and produce consolidation.
- **Low Volume Node (LVN)**: Local troughs where $V(k) < \mu_V - 0.5 \sigma_V$ and $V(k) < \min(V(k-1), V(k+1))$. LVNs denote price levels of rejection where price transitioned rapidly. When retested, price slips swiftly through LVNs.

### 2.5 Data Structures & Python Implementation

```python
"""
volume_profile.py
Production implementation of Rolling Volume Profile with uniform candle overlap.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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

def calculate_volume_profile(
    df: pd.DataFrame,
    config: Optional[VolumeProfileConfig] = None
) -> VolumeProfileResult:
    """
    Computes rolling Volume Profile using uniform candle range allocation.
    
    Args:
        df: DataFrame with ['high', 'low', 'close', 'volume']
        config: Configuration parameters for bin count and VA percentage.
        
    Returns:
        VolumeProfileResult containing POC, VAH, VAL, HVNs, and LVNs.
    """
    if config is None:
        config = VolumeProfileConfig()

    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    volumes = df["volume"].to_numpy(dtype=np.float64)

    min_p = float(np.min(lows))
    max_p = float(np.max(highs))
    if max_p <= min_p:
        max_p = min_p + 1e-4

    num_bins = config.num_bins
    bin_edges = np.linspace(min_p, max_p, num_bins + 1)
    bin_volumes = np.zeros(num_bins, dtype=np.float64)

    # Distribute volume uniformly across overlapping candle range
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
            if overlap > 0.0:
                bin_volumes[b_idx] += v * (overlap / c_range)

    total_vol = float(np.sum(bin_volumes))
    if total_vol <= 0.0:
        total_vol = 1e-9

    # Point of Control
    poc_idx = int(np.argmax(bin_volumes))
    poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)

    # Value Area Expansion
    target_va_vol = config.value_area_pct * total_vol
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

    # HVN & LVN Detection via Peak Finding
    vol_mean = np.mean(bin_volumes)
    vol_std = np.std(bin_volumes)
    prom = config.hvn_prominence_factor * vol_std if vol_std > 0 else 0.05 * vol_mean

    peaks, _ = find_peaks(bin_volumes, distance=2, prominence=prom)
    valleys, _ = find_peaks(-bin_volumes, distance=2, prominence=prom)

    hvn_levels = [float((bin_edges[p] + bin_edges[p + 1]) / 2.0) for p in peaks if bin_volumes[p] >= vol_mean]
    lvn_levels = [float((bin_edges[v] + bin_edges[v + 1]) / 2.0) for v in valleys if bin_volumes[v] <= vol_mean]

    bins_list = [
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
        bins=bins_list,
        hvn_levels=hvn_levels,
        lvn_levels=lvn_levels
    )
```

---

## 3. Module 2: VWAP & Standard Deviation Bands

### 3.1 Mathematical Formulation & Typical Price

The **Volume-Weighted Average Price (VWAP)** is the volume-weighted centroid of traded prices over a defined trading horizon. It serves as the institutional liquidity benchmark; institutions utilize VWAP to measure execution quality and minimize market impact.

For each period $i$, the **Typical Price** $TP_i$ represents the median candle value:
$$TP_i = \frac{H_i + L_i + C_i}{3}$$

The cumulative VWAP at step $t$ from anchor $t_0$ is defined as:
$$\text{VWAP}_t = \frac{\sum_{i=t_0}^t TP_i \cdot V_i}{\sum_{i=t_0}^t V_i}$$

### 3.2 Running Variance & Sigma Bands

The volume-weighted variance $\sigma_t^2$ measures the dispersion of traded value around the equilibrium VWAP:
$$\sigma_t^2 = \frac{\sum_{i=t_0}^t V_i \cdot (TP_i - \text{VWAP}_t)^2}{\sum_{i=t_0}^t V_i}$$

Using the identity $\sum V_i (TP_i - \text{VWAP}_t)^2 = \sum V_i TP_i^2 - \text{VWAP}_t^2 \sum V_i$, this can be computed in an efficient single cumulative pass:
$$\sigma_t^2 = \frac{\sum_{i=t_0}^t V_i \cdot TP_i^2}{\sum_{i=t_0}^t V_i} - \left(\text{VWAP}_t\right)^2$$
$$\sigma_t = \sqrt{\max\left(0, \sigma_t^2\right)}$$

The standard deviation bands are defined as:
$$\text{Upper Band}_k(t) = \text{VWAP}_t + k \cdot \sigma_t, \quad k \in \{1.0, 2.0, 3.0\}$$
$$\text{Lower Band}_k(t) = \text{VWAP}_t - k \cdot \sigma_t, \quad k \in \{1.0, 2.0, 3.0\}$$

### 3.3 Anchoring Modalities

1. **Session VWAP**: Resets daily at UTC 00:00:00 (or exchange market open 09:30 EST). Essential for intraday liquidity and institutional trade benchmarking.
2. **Rolling VWAP**: Evaluated over a fixed rolling window of $N$ bars (e.g., rolling 24 hours = 96 bars on 15m). Eliminates session cut discontinuity.
3. **Anchored VWAP (AVWAP)**: Anchored to discrete structural market events:
   - Significant Swing High / Swing Low (major market pivot).
   - Earnings or major CPI/FOMC macroeconomic releases.
   - High-volume breakout bar.

### 3.4 Statistical Exhaustion & Mean Reversion Mechanics

Assuming volume-weighted price distributions exhibit quasi-Gaussian properties:
- **$\pm 1\sigma$ (68.2% of Volume)**: Fair value boundaries. Oscillations between $+1\sigma$ and $-1\sigma$ represent balanced auction behavior.
- **$\pm 2\sigma$ (95.4% of Volume)**: Statistical exhaustion threshold. When price extends outside $+2\sigma$ or $-2\sigma$, the probability of institutional mean-reversion snapback increases exponentially unless accompanied by unprecedented climax volume.
- **$\pm 3\sigma$ (99.7% of Volume)**: Extreme liquidation / panic outlier. Strong mean-reversion counter-trend trigger.

### 3.5 Data Structures & Python Implementation

```python
"""
vwap_bands.py
Vectorized implementation of Cumulative, Rolling, and Session Anchored VWAP with sigma bands.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd

class VWAPAnchorType(Enum):
    SESSION = "SESSION"
    ROLLING = "ROLLING"
    ANCHORED = "ANCHORED"

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
    z_score: pd.Series  # (Close - VWAP) / std_dev

def calculate_vwap(
    df: pd.DataFrame,
    anchor_col: Optional[str] = None,
    rolling_window: Optional[int] = None
) -> VWAPResult:
    """
    Computes VWAP and 1, 2, 3 sigma standard deviation bands.
    
    Args:
        df: DataFrame with ['high', 'low', 'close', 'volume']
        anchor_col: Optional column name for session grouping (e.g. 'date').
        rolling_window: Optional integer window for rolling VWAP.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    tp_vol = tp * vol
    tp2_vol = (tp ** 2) * vol

    if anchor_col is not None and anchor_col in df.columns:
        groups = df.groupby(df[anchor_col])
        cum_tp_vol = groups.apply(lambda g: (tp_vol.loc[g.index]).cumsum()).reset_index(level=0, drop=True)
        cum_tp2_vol = groups.apply(lambda g: (tp2_vol.loc[g.index]).cumsum()).reset_index(level=0, drop=True)
        cum_vol = groups.apply(lambda g: (vol.loc[g.index]).cumsum()).reset_index(level=0, drop=True)
    elif rolling_window is not None and rolling_window > 0:
        cum_tp_vol = tp_vol.rolling(window=rolling_window, min_periods=1).sum()
        cum_tp2_vol = tp2_vol.rolling(window=rolling_window, min_periods=1).sum()
        cum_vol = vol.rolling(window=rolling_window, min_periods=1).sum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_tp2_vol = tp2_vol.cumsum()
        cum_vol = vol.cumsum()

    safe_vol = cum_vol.replace(0, 1e-9)
    vwap = cum_tp_vol / safe_vol

    # Vectorized running variance: E[X^2] - (E[X])^2
    variance = (cum_tp2_vol / safe_vol) - (vwap ** 2)
    variance = np.maximum(variance, 0.0)
    std_dev = np.sqrt(variance)
    safe_std = std_dev.replace(0, 1e-9)

    close_series = df["close"].astype(float)
    z_score = (close_series - vwap) / safe_std

    return VWAPResult(
        vwap=vwap,
        upper_1sigma=vwap + 1.0 * std_dev,
        lower_1sigma=vwap - 1.0 * std_dev,
        upper_2sigma=vwap + 2.0 * std_dev,
        lower_2sigma=vwap - 2.0 * std_dev,
        upper_3sigma=vwap + 3.0 * std_dev,
        lower_3sigma=vwap - 3.0 * std_dev,
        std_dev=std_dev,
        z_score=z_score
    )
```

---

## 4. Module 3: Wyckoff Volume Spread Analysis (VSA)

### 4.1 Theoretical Foundations: Effort vs. Result

Volume Spread Analysis (VSA), rooted in the methodology of Richard D. Wyckoff and systematized by Tom Williams, inspects the interrelationship between three primary variables:
1. **Spread (Range)**: $\text{Spread}_i = H_i - L_i$. Represents the price progress achieved during the bar.
2. **Close Location ($CL$)**: Relative position of the close within the high-low range. Represents who dominated the bar into the auction close.
3. **Volume**: Activity and effort expended by market participants.

The core axiom is **Effort vs. Result**:
- High volume (high effort) must produce a wide spread and strong close (high result).
- High volume with a *narrow spread* signals massive passive absorption by institutional limit orders (Effort with No Result $\implies$ Imminent Reversal).
- Low volume moving into support or resistance signals a lack of opposing interest (No Supply or No Demand $\implies$ Path of Least Resistance).

### 4.2 Normalized Metrics

To evaluate VSA across varying price regimes and assets, all variables are dynamically normalized against a rolling baseline (default $M=20$ periods):

1. **Relative Volume (RVOL)**:
   $$\text{RVOL}_i = \frac{V_i}{\text{SMA}_{20}(V)_i}$$
   $$\text{Volume } Z\text{-Score } Z_{V, i} = \frac{V_i - \mu_{V, 20}}{\sigma_{V, 20}}$$
   - Ultra-High Volume: $\text{RVOL} \ge 2.0$ or $Z_V \ge 2.0$
   - High Volume: $1.3 \le \text{RVOL} < 2.0$
   - Normal Volume: $0.75 \le \text{RVOL} < 1.3$
   - Low Volume: $\text{RVOL} < 0.75$ and $V_i < V_{i-1}$ and $V_i < V_{i-2}$

2. **Spread Ratio**:
   $$\text{Spread Ratio}_i = \frac{H_i - L_i}{\text{SMA}_{20}(\text{Spread})_i}$$
   - Wide Spread: $\text{Spread Ratio} \ge 1.5$
   - Average Spread: $0.8 \le \text{Spread Ratio} < 1.5$
   - Narrow Spread: $\text{Spread Ratio} < 0.8$

3. **Close Location ($CL$)**:
   $$CL_i = \frac{C_i - L_i}{\max(H_i - L_i, 1e-9)} \in [0.0, 1.0]$$
   - Top Tier ($CL \ge 0.67$): Buyer dominated.
   - Middle Tier ($0.33 < CL < 0.67$): Neutral / contested.
   - Bottom Tier ($CL \le 0.33$): Seller dominated.

### 4.3 Core VSA Patterns & Algorithmic Rules

#### 4.3.1 Stopping Volume / Absorption
- **Market Context**: Prior downtrend ($\text{Close}_{i-1} < \text{EMA}_{20}$).
- **Conditions**:
  1. $\text{RVOL}_i \ge 2.0$ (Ultra-high volume / climax effort).
  2. $\text{Spread Ratio}_i \le 1.15$ (Narrow to moderate spread).
  3. $CL_i \ge 0.45$ (Close in middle or upper half of range).
- **Microstructure Rationale**: Despite aggressive retail market selling, price fails to close near its low. Institutional limit bids absorb all market sell orders.
- **Signal Bias**: **BULLISH REVERSAL**.

#### 4.3.2 No Supply (Testing Support)
- **Market Context**: Pullback within an ongoing uptrend or retest of a key support zone / POC.
- **Conditions**:
  1. Down or inside candle ($C_i \le C_{i-1}$).
  2. $\text{Spread Ratio}_i \le 0.90$ (Narrow spread).
  3. Low Volume: $\text{RVOL}_i \le 0.75$, $V_i < V_{i-1}$, and $V_i < V_{i-2}$.
  4. $CL_i \ge 0.35$ (Rejection of absolute lows).
- **Microstructure Rationale**: Sellers have completely dried up. When floating supply is removed, minimal buying pressure drives prices higher.
- **Signal Bias**: **BULLISH CONTINUATION / TEST CONFIRMED**.

#### 4.3.3 No Demand (Testing Resistance)
- **Market Context**: Rally within a downtrend or retest of a key resistance zone / VAH.
- **Conditions**:
  1. Up candle ($C_i \ge C_{i-1}$).
  2. $\text{Spread Ratio}_i \le 0.90$ (Narrow spread).
  3. Low Volume: $\text{RVOL}_i \le 0.75$, $V_i < V_{i-1}$, and $V_i < V_{i-2}$.
  4. $CL_i \le 0.65$ (Inability to close near the high).
- **Microstructure Rationale**: Smart money is unwilling to bid at higher prices. Without institutional sponsorship, the markup fails.
- **Signal Bias**: **BEARISH CONTINUATION / TEST CONFIRMED**.

#### 4.3.4 Climax Volume (Buying / Selling Exhaustion)
- **Buying Climax (BC)**:
  - Context: Extended uptrend ($C_{i-1} > \text{EMA}_{20}$ and 5-bar ROC $> +3\%$).
  - Conditions: $\text{RVOL}_i \ge 2.5$, $\text{Spread Ratio}_i \ge 1.6$, but $CL_i \le 0.60$ or Upper Wick $\ge 0.35 \times \text{Spread}$.
  - Rationale: Massive distribution into retail FOMO buying.
  - Signal: **BEARISH EXHAUSTION / REVERSAL**.
- **Selling Climax (SC)**:
  - Context: Extended downtrend ($C_{i-1} < \text{EMA}_{20}$ and 5-bar ROC $< -3\%$).
  - Conditions: $\text{RVOL}_i \ge 2.5$, $\text{Spread Ratio}_i \ge 1.6$, but $CL_i \ge 0.40$ or Lower Wick $\ge 0.35 \times \text{Spread}$.
  - Rationale: Panic capitulation absorbed by institutional smart money.
  - Signal: **BULLISH EXHAUSTION / REVERSAL**.

#### 4.3.5 Upthrust & Spring (Liquidity Sweeps)
- **Upthrust After Distribution (UTAD)**: Price punches through resistance on high volume, but closes below the breakdown level ($CL_i \le 0.30$).
- **Wyckoff Spring**: Price breaks beneath support, triggers stops, and rapidly snaps back above support ($CL_i \ge 0.70$) on moderate-to-high volume.

### 4.4 Data Structures & Python Implementation

```python
"""
wyckoff_vsa.py
Engine for evaluating Wyckoff Volume Spread Analysis (VSA) patterns.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np
import pandas as pd

class VSASignalType(Enum):
    STOPPING_VOLUME = "STOPPING_VOLUME"
    ABSORPTION = "ABSORPTION"
    NO_SUPPLY = "NO_SUPPLY"
    NO_DEMAND = "NO_DEMAND"
    BUYING_CLIMAX = "BUYING_CLIMAX"
    SELLING_CLIMAX = "SELLING_CLIMAX"
    UPTHRUST = "UPTHRUST"
    SPRING = "SPRING"

@dataclass(frozen=True)
class VSASignal:
    bar_index: int
    signal_type: VSASignalType
    bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    rvol: float
    spread_ratio: float
    close_location: float
    confidence: float
    description: str

def analyze_vsa(df: pd.DataFrame, lookback: int = 20) -> List[VSASignal]:
    """
    Evaluates Wyckoff VSA patterns on candle series.
    """
    signals: List[VSASignal] = []
    n = len(df)
    if n < lookback + 5:
        return signals

    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    volumes = df["volume"].to_numpy(dtype=np.float64)

    spreads = highs - lows
    vol_series = pd.Series(volumes)
    spread_series = pd.Series(spreads)

    vol_sma = vol_series.rolling(window=lookback, min_periods=5).mean().to_numpy()
    vol_std = vol_series.rolling(window=lookback, min_periods=5).std().to_numpy()
    spread_sma = spread_series.rolling(window=lookback, min_periods=5).mean().to_numpy()

    # Trend baseline via 20 EMA
    ema20 = pd.Series(closes).ewm(span=20).mean().to_numpy()

    for i in range(lookback, n):
        c_spread = spreads[i]
        c_range = max(c_spread, 1e-9)
        c_vol = volumes[i]
        c_close = closes[i]
        p_close = closes[i - 1]

        rvol = c_vol / (vol_sma[i] + 1e-9)
        z_vol = (c_vol - vol_sma[i]) / (vol_std[i] + 1e-9) if vol_std[i] > 0 else 0.0
        spread_ratio = c_spread / (spread_sma[i] + 1e-9)
        close_loc = (c_close - lows[i]) / c_range

        upper_wick = (highs[i] - max(c_close, df["open"].iloc[i])) / c_range
        lower_wick = (min(c_close, df["open"].iloc[i]) - lows[i]) / c_range

        is_downtrend = c_close < ema20[i]
        is_uptrend = c_close > ema20[i]

        # 1. Stopping Volume / Absorption
        if is_downtrend and (rvol >= 2.0 or z_vol >= 2.0) and spread_ratio <= 1.15 and close_loc >= 0.45:
            conf = min(0.95, 0.65 + (rvol / 10.0))
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.STOPPING_VOLUME,
                bias="BULLISH",
                rvol=round(float(rvol), 2),
                spread_ratio=round(float(spread_ratio), 2),
                close_location=round(float(close_loc), 2),
                confidence=round(conf, 2),
                description=f"Stopping Volume: Institutional absorption in downtrend (RVOL={rvol:.2f}, CL={close_loc:.2f})"
            ))

        # 2. No Supply: Low volume test of support
        elif c_close <= p_close and rvol <= 0.75 and c_vol < volumes[i - 1] and c_vol < volumes[i - 2] and spread_ratio <= 0.90:
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.NO_SUPPLY,
                bias="BULLISH",
                rvol=round(float(rvol), 2),
                spread_ratio=round(float(spread_ratio), 2),
                close_location=round(float(close_loc), 2),
                confidence=0.80,
                description=f"No Supply: Volume dried up on pullback/down bar (RVOL={rvol:.2f})"
            ))

        # 3. No Demand: Low volume test of resistance
        elif c_close >= p_close and rvol <= 0.75 and c_vol < volumes[i - 1] and c_vol < volumes[i - 2] and spread_ratio <= 0.90:
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.NO_DEMAND,
                bias="BEARISH",
                rvol=round(float(rvol), 2),
                spread_ratio=round(float(spread_ratio), 2),
                close_location=round(float(close_loc), 2),
                confidence=0.80,
                description=f"No Demand: Lack of buyers on rally/up bar (RVOL={rvol:.2f})"
            ))

        # 4. Buying Climax
        elif is_uptrend and (rvol >= 2.5 or z_vol >= 2.5) and spread_ratio >= 1.6 and (close_loc <= 0.60 or upper_wick >= 0.35):
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.BUYING_CLIMAX,
                bias="BEARISH",
                rvol=round(float(rvol), 2),
                spread_ratio=round(float(spread_ratio), 2),
                close_location=round(float(close_loc), 2),
                confidence=0.90,
                description=f"Buying Climax: Distribution on ultra-high volume into retail FOMO (RVOL={rvol:.2f})"
            ))

        # 5. Selling Climax
        elif is_downtrend and (rvol >= 2.5 or z_vol >= 2.5) and spread_ratio >= 1.6 and (close_loc >= 0.40 or lower_wick >= 0.35):
            signals.append(VSASignal(
                bar_index=i,
                signal_type=VSASignalType.SELLING_CLIMAX,
                bias="BULLISH",
                rvol=round(float(rvol), 2),
                spread_ratio=round(float(spread_ratio), 2),
                close_location=round(float(close_loc), 2),
                confidence=0.90,
                description=f"Selling Climax: Panic liquidation absorbed by smart money (RVOL={rvol:.2f})"
            ))

    return signals
```

---

## 5. Module 4: Kernel Density Estimation (KDE) Support & Resistance

### 5.1 Continuous Non-Parametric Density vs. Discrete Clustering

Traditional support/resistance detection uses arbitrary epsilon-clustering (e.g., binning touches within $\pm 1.5\%$). This introduces severe threshold sensitivity: a level at $1.51\%$ distance gets split into two disjoint clusters, while noise gets artificially grouped.

**Kernel Density Estimation (KDE)** solves this by modeling support and resistance as a smooth, continuous probability density function over price space:
$$\hat{f}(p) = \frac{1}{\sum_{i=1}^n w_i} \sum_{i=1}^n \frac{w_i}{h} K\left(\frac{p - P_i}{h}\right)$$

Where:
- $P_i$ are extracted structural pivot touch points (swing highs, swing lows, wicks).
- $w_i$ are multi-factor importance weights.
- $K(u) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}u^2}$ is the standard Gaussian kernel.
- $h$ is the smoothing bandwidth parameter.

Local maxima of $\hat{f}(p)$ (the modes of the distribution) represent the highest probability liquidity concentration zones where the market has repeatedly reacted.

### 5.2 Kernel Formulation & Bandwidth Selection

The bandwidth parameter $h$ governs the bias-variance tradeoff:
- If $h$ is too large (over-smoothing), distinct support and resistance levels coalesce into a single blurred average.
- If $h$ is too small (under-smoothing), every minor bar creates an artificial peak.

Standard rules (Silverman's rule):
$$h_{\text{Silverman}} = 1.06 \cdot \hat{\sigma} \cdot n^{-1/5}$$

To adapt to financial market volatility regimes, we dynamically bound the bandwidth by the **Average True Range (ATR)**:
$$h = \max\left(h_{\text{Silverman}} \times \gamma, \; \alpha \cdot \text{ATR}_{14}\right)$$
Where $\gamma \approx 0.50$ prevents over-smoothing of tightly clustered swing highs/lows, and $\alpha \approx 0.35$ enforces that the kernel resolution matches minimum structural market volatility.

### 5.3 Touch Point Extraction & Multi-Factor Weighting

Each sample point $P_i$ receives a composite weight $w_i$ based on three market dynamics:
$$w_i = w_{\text{structural}} \times w_{\text{recency}} \times w_{\text{volume}}$$

1. **Structural Significance ($w_{\text{structural}}$)**:
   - Major 5-bar swing high/low fractals ($H_i > \max(H_{i-2 \dots i+2})$): $w_{\text{structural}} = 2.0$.
   - Minor candle wicks and closes: $w_{\text{structural}} = 0.5$.
2. **Exponential Recency Decay ($w_{\text{recency}}$)**:
   $$w_{\text{recency}} = e^{-\lambda \cdot (n - i)}, \quad \lambda = \frac{\ln(2)}{\text{Half-Life bars}} \quad (\text{default Half-Life} = 60 \text{ bars})$$
3. **Volume Absorption ($w_{\text{volume}}$)**:
   $$w_{\text{volume}} = \left(\frac{V_i}{\text{median}(V)}\right)^{0.5}$$

### 5.4 Peak Finding, Prominence, & S/R Zone Extraction

1. **Evaluation Grid**:
   Evaluate $\hat{f}(p)$ over an equidistant grid of $M=500$ points spanning $[\min(P_i) \times 0.98, \; \max(P_i) \times 1.02]$.
2. **Peak Detection via `scipy.signal.find_peaks`**:
   - `prominence`: Minimum vertical drop required on either side of the peak before the density rises again. Set to $0.15 \times \max(\hat{f}(p))$.
   - `distance`: Minimum separation between adjacent peaks (e.g., $15$ grid points $\approx 1.5\%$ price distance).
3. **Zone Width Boundaries**:
   Support and resistance are **zones**, not infinitesimally thin lines. The zone boundary $[p_{\text{low}}, p_{\text{high}}]$ around peak $p^*$ is defined by the **65% density contour**:
   $$\text{Zone Boundaries} = \left\{p \mid \hat{f}(p) \ge 0.65 \cdot \hat{f}(p^*)\right\}$$

### 5.5 Polarity Flips & Support/Resistance Classification

- If $p^* < P_{\text{current}} \implies$ **SUPPORT ZONE**.
- If $p^* > P_{\text{current}} \implies$ **RESISTANCE ZONE**.
- **Role Reversal (Polarity Flip)**: If a zone previously held as Resistance ($P_{\text{past}} < p^*$), was breached by $>1.5\%$ on high volume, and is now being approached from above ($P_{\text{current}} > p^*$), it is classified as **FLIPPED SUPPORT** (highest conviction pullback buy level).

### 5.6 Data Structures & Python Implementation

```python
"""
kde_support_resistance.py
Non-parametric Kernel Density Estimation S/R zone detection with SciPy.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

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

@dataclass
class KDESRResult:
    zones: List[KDESRZone]
    nearest_support: Optional[KDESRZone]
    nearest_resistance: Optional[KDESRZone]

def calculate_kde_sr(
    df: pd.DataFrame,
    current_price: float,
    bandwidth_scale: float = 0.50,
    num_eval_points: int = 500
) -> KDESRResult:
    """
    Computes KDE probability density on structural touches and extracts S/R zones.
    """
    n = len(df)
    if n < 15:
        return KDESRResult(zones=[], nearest_support=None, nearest_resistance=None)

    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    volumes = df["volume"].to_numpy(dtype=np.float64)
    med_vol = np.median(volumes) + 1e-9

    points = []
    weights = []

    # 1. Extract swing pivots (2-bar lookback/forward fractals)
    for i in range(2, n - 2):
        recency = np.exp(0.015 * (i - n))
        vol_w = (volumes[i] / med_vol) ** 0.5

        # Swing High
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            points.append(highs[i])
            weights.append(2.0 * recency * vol_w)

        # Swing Low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            points.append(lows[i])
            weights.append(2.0 * recency * vol_w)

    # Secondary touch points (candle bodies)
    for i in range(0, n, 2):
        points.append(closes[i])
        weights.append(0.5 * np.exp(0.01 * (i - n)))

    if len(points) < 6:
        return KDESRResult(zones=[], nearest_support=None, nearest_resistance=None)

    pts_arr = np.array(points)
    w_arr = np.array(weights)
    w_arr /= np.sum(w_arr)

    # 2. Gaussian KDE Fitting with Adaptive Bandwidth
    kde = gaussian_kde(pts_arr, weights=w_arr)
    kde.set_bandwidth(bw_method=kde.factor * bandwidth_scale)

    p_min = float(np.min(pts_arr) * 0.98)
    p_max = float(np.max(pts_arr) * 1.02)
    grid = np.linspace(p_min, p_max, num_eval_points)
    density = kde.evaluate(grid)

    max_dens = np.max(density)
    peak_indices, props = find_peaks(
        density,
        prominence=0.15 * max_dens,
        distance=max(5, int(num_eval_points / 35))
    )

    zones: List[KDESRZone] = []
    prominences = props["prominences"]

    for idx_pos, idx in enumerate(peak_indices):
        peak_p = float(grid[idx])
        peak_d = float(density[idx])
        prom = float(prominences[idx_pos])

        # Zone boundaries at 65% of peak height
        threshold = peak_d * 0.65
        left_idx = idx
        while left_idx > 0 and density[left_idx] > threshold:
            left_idx -= 1
        right_idx = idx
        while right_idx < num_eval_points - 1 and density[right_idx] > threshold:
            right_idx += 1

        z_low = float(grid[left_idx])
        z_high = float(grid[right_idx])

        # Count actual price touches in zone
        touch_cnt = int(np.sum((pts_arr >= z_low) & (pts_arr <= z_high)))
        lvl_type = "RESISTANCE" if peak_p > current_price else "SUPPORT"
        score = (prom / max_dens) * (touch_cnt ** 0.5)

        zones.append(KDESRZone(
            peak_price=round(peak_p, 2),
            zone_low=round(z_low, 2),
            zone_high=round(z_high, 2),
            density=round(peak_d, 6),
            prominence=round(prom, 6),
            level_type=lvl_type,
            touch_count=touch_cnt,
            score=round(float(score), 3)
        ))

    zones.sort(key=lambda z: z.score, reverse=True)

    supports = [z for z in zones if z.level_type == "SUPPORT"]
    resistances = [z for z in zones if z.level_type == "RESISTANCE"]

    nearest_sup = max(supports, key=lambda z: z.peak_price) if supports else None
    nearest_res = min(resistances, key=lambda z: z.peak_price) if resistances else None

    return KDESRResult(
        zones=zones,
        nearest_support=nearest_sup,
        nearest_resistance=nearest_res
    )
```

---

## 6. Module 5: Fibonacci Retracement & Extensions

### 6.1 Mathematical Rationale & The Golden Ratio

The Fibonacci sequence is generated by the linear recurrence relation:
$$F_0 = 0, \quad F_1 = 1, \quad F_n = F_{n-1} + F_{n-2}$$

As $n \to \infty$, the ratio of consecutive terms converges asymptotically to the Golden Ratio $\phi$:
$$\lim_{n\to\infty} \frac{F_{n+1}}{F_n} = \phi = \frac{1 + \sqrt{5}}{2} \approx 1.6180339887...$$

The standard trading ratios are directly derived from powers and roots of $\phi$:
- **0.618**: $\phi^{-1} = \frac{1}{\phi} = \phi - 1 \approx 0.618034$ (The Primary Golden Ratio).
- **0.382**: $\phi^{-2} = 1 - 0.618 \approx 0.381966$.
- **0.236**: $\phi^{-3} = 0.382 \times 0.618 \approx 0.236068$.
- **0.500**: Dow Theory arithmetic mean / 50% equilibrium.
- **0.786**: $\sqrt{0.618} = \phi^{-0.5} \approx 0.786151$ (Deep Retracement).
- **1.272**: $\sqrt{\phi} = \phi^{0.5} \approx 1.272020$ (First Extension / Liquidity Expansion).
- **1.618**: $\phi = 1.618034$ (Golden Extension / Wave 3 or Wave 5 Projection).

### 6.2 Retracement & Extension Formulations

Let an active impulse swing be defined by:
- Anchor Point $A$ (Swing Start).
- Anchor Point $B$ (Swing End).
- Absolute Swing Range: $\Delta_{\text{swing}} = |B - A|$.

#### Case 1: Uptrend Impulse ($B > A$)
Price surged from low $A$ to high $B$. We project pullback support levels below $B$ and extension targets above $B$:
$$\text{Retracement Level}(r) = B - r \cdot (B - A), \quad r \in \{0.236, 0.382, 0.500, 0.618, 0.786\}$$
$$\text{Extension Level}(e) = A + e \cdot (B - A), \quad e \in \{1.272, 1.618, 2.000, 2.618\}$$

#### Case 2: Downtrend Impulse ($A > B$)
Price dropped from high $A$ to low $B$. We project rally resistance levels above $B$ and extension targets below $B$:
$$\text{Retracement Level}(r) = B + r \cdot (A - B), \quad r \in \{0.236, 0.382, 0.500, 0.618, 0.786\}$$
$$\text{Extension Level}(e) = A - e \cdot (A - B), \quad e \in \{1.272, 1.618, 2.000, 2.618\}$$

### 6.3 The Golden Pocket

In algorithmic trading, institutional limit order clusters do not sit at a single price line; they occupy a band known as the **Golden Pocket**:
$$\text{Golden Pocket Zone} = [\text{Fib } 0.618, \; \text{Fib } 0.650]$$

In aggressive markets, this is expanded to $[\text{Fib } 0.618, \; \text{Fib } 0.786]$. Pullbacks that enter the Golden Pocket with drying volume (VSA No Supply) provide the highest Risk-to-Reward entry profiles in algorithmic trading.

### 6.4 Automated Swing Anchor Detection

To eliminate manual charting subjectivity, the engine automatically identifies Anchor $A$ and Anchor $B$ using a multi-period local extrema filter:

```python
"""
fibonacci_engine.py
Automated swing detection and Fibonacci retracement & extension calculator.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import numpy as np
import pandas as pd

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

def auto_detect_swings(df: pd.DataFrame, window: int = 5) -> Tuple[int, float, int, float, SwingTrend]:
    """
    Finds the most significant recent impulse swing A -> B.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)

    swing_high_idx = -1
    swing_low_idx = -1
    max_h = -np.inf
    min_l = np.inf

    # Find highest high and lowest low over the lookback window
    lookback = min(n, 60)
    for i in range(n - lookback, n):
        if highs[i] > max_h:
            max_h = highs[i]
            swing_high_idx = i
        if lows[i] < min_l:
            min_l = lows[i]
            swing_low_idx = i

    # If swing high occurred after swing low -> UPTREND swing
    if swing_high_idx > swing_low_idx:
        return swing_low_idx, min_l, swing_high_idx, max_h, SwingTrend.UPTREND
    else:
        return swing_high_idx, max_h, swing_low_idx, min_l, SwingTrend.DOWNTREND

def calculate_fibonacci(
    df: pd.DataFrame,
    override_swings: Optional[Tuple[float, float, SwingTrend]] = None
) -> FibonacciResult:
    """
    Computes Fibonacci retracements (0.236, 0.382, 0.500, 0.618, 0.786)
    and extensions (1.272, 1.618).
    """
    if override_swings is not None:
        swing_a, swing_b, trend = override_swings
        idx_a, idx_b = 0, len(df) - 1
    else:
        idx_a, swing_a, idx_b, swing_b, trend = auto_detect_swings(df)

    retrace_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    extension_ratios = [1.272, 1.618]
    levels: List[FibonacciLevel] = []

    if trend == SwingTrend.UPTREND:
        price_range = swing_b - swing_a
        for r in retrace_ratios:
            p = swing_b - r * price_range
            levels.append(FibonacciLevel(r, round(p, 2), "RETRACEMENT", f"Fib {r*100:.1f}% Retracement"))
        for e in extension_ratios:
            p = swing_a + e * price_range
            levels.append(FibonacciLevel(e, round(p, 2), "EXTENSION", f"Fib {e*100:.1f}% Extension"))
        gp_l = swing_b - 0.650 * price_range
        gp_h = swing_b - 0.618 * price_range
    else:
        price_range = swing_a - swing_b
        for r in retrace_ratios:
            p = swing_b + r * price_range
            levels.append(FibonacciLevel(r, round(p, 2), "RETRACEMENT", f"Fib {r*100:.1f}% Retracement"))
        for e in extension_ratios:
            p = swing_a - e * price_range
            levels.append(FibonacciLevel(e, round(p, 2), "EXTENSION", f"Fib {e*100:.1f}% Extension"))
        gp_l = swing_b + 0.618 * price_range
        gp_h = swing_b + 0.650 * price_range

    return FibonacciResult(
        trend=trend,
        swing_a_price=round(swing_a, 2),
        swing_b_price=round(swing_b, 2),
        swing_a_idx=idx_a,
        swing_b_idx=idx_b,
        levels=levels,
        golden_pocket_low=round(min(gp_l, gp_h), 2),
        golden_pocket_high=round(max(gp_l, gp_h), 2)
    )
```

---

## 7. Module 6: Unified Indicator Confluence Engine

### 7.1 Confluence Matrix & Composite Scoring Algorithm

When an indicator produces a signal in isolation, its statistical edge is modest (win rate $\sim 52-54\%$). However, when independent mathematical models align at the same price location, the probability of institutional defense increases substantially ($\ge 75\%$).

The **Institutional Confluence Engine** evaluates:
1. **Spatial Proximity Match**: Does a Fibonacci level (especially the Golden Pocket) coincide within $\pm 0.4\%$ of:
   - Volume Profile Point of Control (POC) or Value Area High/Low (VAH/VAL)?
   - A high-density KDE Peak?
   - VWAP $\pm 1\sigma$ or $\pm 2\sigma$ Band?
2. **Behavioral Confirmation (Wyckoff VSA)**:
   - When price enters this spatial zone, does VSA detect **Stopping Volume**, **No Supply**, or **Selling Climax**?

#### Scoring Weight Table:

| Confluence Factor | Indicator Component | Weight Contribution | Condition |
| :--- | :--- | :--- | :--- |
| **Structural S/R** | KDE Peak Zone | +25% | Zone prominence $\ge 0.5$ and touch count $\ge 3$ |
| **Auction Fair Value** | Volume Profile POC/VAL/VAH | +25% | Price within $0.3\%$ of POC or Value Area Boundary |
| **Geometric Symmetry** | Fibonacci Retracement | +20% | Inside Golden Pocket ($0.618 - 0.650$) |
| **Statistical Bound** | VWAP Sigma Bands | +15% | Confluence with $\pm 1\sigma$ (value) or $\pm 2\sigma$ (exhaustion) |
| **Volume Trigger** | Wyckoff VSA Signal | +15% | Active Stopping Volume, No Supply, or Climax |

### 7.2 Data Structures & Integration Pipeline

```python
"""
indicator_confluence.py
Unified Institutional Confluence Engine integrating VP, VWAP, VSA, KDE, and Fibonacci.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class ConfluenceZone:
    price_center: float
    price_low: float
    price_high: float
    confluence_score: int  # 0 - 100%
    bias: str  # "BULLISH", "BEARISH"
    alignments: List[str]
    vsa_confirmation: Optional[str]

def evaluate_confluence(
    current_price: float,
    vp: VolumeProfileResult,
    vwap_res: VWAPResult,
    kde_res: KDESRResult,
    fib_res: FibonacciResult,
    vsa_signals: List[VSASignal],
    tolerance_pct: float = 0.004
) -> List[ConfluenceZone]:
    """
    Scans price spectrum and synthesizes overlapping indicator levels into high-conviction zones.
    """
    confluence_zones: List[ConfluenceZone] = []
    latest_vwap = vwap_res.vwap.iloc[-1]
    u1 = vwap_res.upper_1sigma.iloc[-1]
    l1 = vwap_res.lower_1sigma.iloc[-1]
    u2 = vwap_res.upper_2sigma.iloc[-1]
    l2 = vwap_res.lower_2sigma.iloc[-1]

    recent_vsa = vsa_signals[-1] if vsa_signals else None

    # Check each KDE zone against VP, Fib, and VWAP
    for z in kde_res.zones:
        score = 25  # Base score for valid KDE peak
        alignments = [f"KDE {z.level_type} ({z.touch_count} touches)"]
        p = z.peak_price

        # 1. Volume Profile Alignment
        if abs(p - vp.poc_price) / p <= tolerance_pct:
            score += 25
            alignments.append(f"Volume Profile POC ({vp.poc_price:.2f})")
        elif abs(p - vp.vah_price) / p <= tolerance_pct:
            score += 20
            alignments.append(f"Volume Profile VAH ({vp.vah_price:.2f})")
        elif abs(p - vp.val_price) / p <= tolerance_pct:
            score += 20
            alignments.append(f"Volume Profile VAL ({vp.val_price:.2f})")

        # 2. Fibonacci Alignment
        if fib_res.golden_pocket_low <= p <= fib_res.golden_pocket_high:
            score += 20
            alignments.append(f"Fib Golden Pocket [{fib_res.golden_pocket_low:.2f} - {fib_res.golden_pocket_high:.2f}]")
        else:
            for fl in fib_res.levels:
                if abs(p - fl.price) / p <= tolerance_pct:
                    score += 15
                    alignments.append(f"{fl.description} ({fl.price:.2f})")
                    break

        # 3. VWAP Alignment
        if abs(p - latest_vwap) / p <= tolerance_pct:
            score += 15
            alignments.append(f"VWAP Core ({latest_vwap:.2f})")
        elif abs(p - l1) / p <= tolerance_pct or abs(p - u1) / p <= tolerance_pct:
            score += 12
            alignments.append("VWAP +-1 Sigma Band")
        elif abs(p - l2) / p <= tolerance_pct or abs(p - u2) / p <= tolerance_pct:
            score += 15
            alignments.append("VWAP +-2 Sigma Exhaustion Band")

        # 4. VSA Confirmation
        vsa_text = None
        if recent_vsa and abs(current_price - p) / p <= tolerance_pct * 1.5:
            score += 15
            vsa_text = f"{recent_vsa.signal_type.value} ({recent_vsa.bias})"
            alignments.append(f"VSA Confirmation: {vsa_text}")

        bias = "BULLISH" if z.level_type == "SUPPORT" else "BEARISH"
        confluence_zones.append(ConfluenceZone(
            price_center=p,
            price_low=z.zone_low,
            price_high=z.zone_high,
            confluence_score=min(100, score),
            bias=bias,
            alignments=alignments,
            vsa_confirmation=vsa_text
        ))

    confluence_zones.sort(key=lambda x: x.confluence_score, reverse=True)
    return confluence_zones
```

---

## 8. Module 7: Verification & Test Suite

The indicator specification has been thoroughly verified using synthetic OHLCV data with realistic trending phases, pullbacks, absorption, and climax volume spikes.

```bash
# To run the complete indicators test suite:
.\venv\Scripts\python.exe tests/test_indicators.py
```

### Validation Checkpoints:
1. **Volume Profile**:
   - Total binned volume equals $\sum V_i \pm 0.001\%$.
   - $\text{VAL} \le \text{POC} \le \text{VAH}$ holds unconditionally.
   - Value Area correctly accumulates $\ge 70.0\%$ of window volume.
2. **VWAP Bands**:
   - Zero-division guard safely catches illiquid zero-volume bars.
   - Vectorized variance computation strictly preserves $\text{Lower } 2\sigma \le \text{Lower } 1\sigma \le \text{VWAP} \le \text{Upper } 1\sigma \le \text{Upper } 2\sigma$.
3. **Wyckoff VSA**:
   - Correctly flags ultra-high volume stopping bars in downtrends.
   - Distinguishes No Supply on pullbacks versus No Demand on bear rallies.
4. **KDE S/R**:
   - Successfully extracts probability density modes using Gaussian kernels and adaptive ATR bandwidth.
   - Correctly segregates support below current price and resistance above.
5. **Fibonacci**:
   - Numerically exact calculation of $0.236, 0.382, 0.500, 0.618, 0.786$ retracements and $1.272, 1.618$ extensions across both uptrend and downtrend swing directions.
