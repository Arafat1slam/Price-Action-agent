"""
engines/chart_pattern_engine.py - Classical & Geometric Chart Pattern Recognition Engine

Detects:
  1. Double Top & Double Bottom with symmetry and neckline breakout.
  2. Head & Shoulders & Inverse Head & Shoulders.
  3. Triangles (Ascending, Descending, Symmetrical) and Wedges (Rising, Falling).
  4. Classical multi-candle formations (Morning Star, Evening Star, Three White Soldiers,
     Three Black Crows, Fakey, Engulfing).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

from core.models import BiasType, ChartPattern

# ============================================================================
# Engine Implementation
# ============================================================================

class ChartPatternEngine:
    """
    Algorithmic Chart Pattern Recognition Engine for geometric and multi-candle formations.
    """
    def __init__(self,
                 swing_window: int = 2,
                 double_top_tol: float = 0.015,
                 hs_symmetry_tol: float = 0.035,
                 min_pattern_span: int = 5,
                 max_pattern_span: int = 50):
        self.swing_window = swing_window
        self.double_top_tol = double_top_tol
        self.hs_symmetry_tol = hs_symmetry_tol
        self.min_pattern_span = min_pattern_span
        self.max_pattern_span = max_pattern_span

    def find_local_extrema(self, df: pd.DataFrame, window: Optional[int] = None) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Extracts prominent swing peaks and troughs for geometric pattern detection.
        Uses strict inequality (> and <) to eliminate flat-price noise.
        Returns (peaks, troughs) where each entry is (candle_index, price).
        """
        w = window if window is not None else self.swing_window
        high = df['high'].values
        low = df['low'].values
        n = len(df)

        peaks: List[Tuple[int, float]] = []
        troughs: List[Tuple[int, float]] = []

        if n < 2 * w + 1:
            return peaks, troughs

        for i in range(w, n - w):
            # Strict peak condition
            if all(high[i] > high[i - j] for j in range(1, w + 1)) and \
               all(high[i] > high[i + j] for j in range(1, w + 1)):
                peaks.append((i, float(high[i])))

            # Strict trough condition
            if all(low[i] < low[i - j] for j in range(1, w + 1)) and \
               all(low[i] < low[i + j] for j in range(1, w + 1)):
                troughs.append((i, float(low[i])))

        return peaks, troughs

    # ------------------------------------------------------------------------
    # 1. Double Top & Double Bottom
    # ------------------------------------------------------------------------
    def detect_double_tops(self, df: pd.DataFrame, peaks: Optional[List[Tuple[int, float]]] = None) -> List[ChartPattern]:
        """
        Detects Double Top (Bearish Reversal) formations with symmetry, neckline trough,
        and confirmation/breakout status.
        """
        n = len(df)
        if peaks is None:
            peaks, _ = self.find_local_extrema(df)

        low = df['low'].values
        close = df['close'].values
        patterns: List[ChartPattern] = []

        if len(peaks) < 2:
            return patterns

        for i in range(len(peaks) - 1):
            p1_idx, p1_price = peaks[i]
            for j in range(i + 1, len(peaks)):
                p2_idx, p2_price = peaks[j]
                span = p2_idx - p1_idx

                if span < self.min_pattern_span:
                    continue
                if span > self.max_pattern_span:
                    break

                # 1. Symmetry check: peak height similarity
                price_diff_pct = abs(p1_price - p2_price) / min(p1_price, p2_price)
                if price_diff_pct > self.double_top_tol:
                    continue

                # 2. Intervening trough between p1 and p2
                intervening_lows = low[p1_idx : p2_idx + 1]
                trough_price = float(np.min(intervening_lows))
                trough_rel_idx = int(np.argmin(intervening_lows))
                trough_idx = p1_idx + trough_rel_idx

                # Trough must be strictly between the two peaks
                if trough_idx == p1_idx or trough_idx == p2_idx:
                    continue

                # Trough depth: peaks must stand out above neckline
                min_peak = min(p1_price, p2_price)
                trough_depth_pct = (min_peak - trough_price) / (trough_price + 1e-9)
                if trough_depth_pct < 0.005:  # At least 0.5% depth
                    continue

                # 3. Check invalidation: price closing above highest peak
                highest_peak = max(p1_price, p2_price)
                neckline = trough_price
                target = neckline - (highest_peak - neckline)

                # Check breakout beyond p2
                is_confirmed = False
                confirmed_idx = None
                for k in range(p2_idx + 1, n):
                    if close[k] < neckline:
                        is_confirmed = True
                        confirmed_idx = k
                        break
                    if close[k] > highest_peak:
                        break

                # Quality Score (0.0 to 1.0)
                symmetry_score = max(0.0, 1.0 - (price_diff_pct / self.double_top_tol))
                depth_score = min(1.0, trough_depth_pct / 0.03)
                quality = round(0.5 * symmetry_score + 0.3 * depth_score + (0.2 if is_confirmed else 0.0), 2)

                patterns.append(ChartPattern(
                    name="DOUBLE_TOP",
                    bias=BiasType.BEARISH,
                    quality_score=quality,
                    neckline_price=round(neckline, 4),
                    projected_target=round(target, 4),
                    invalidation_level=round(highest_peak, 4),
                    candle_span=span,
                    start_index=p1_idx,
                    end_index=confirmed_idx if confirmed_idx else p2_idx,
                    metadata={
                        "peak1": (p1_idx, p1_price),
                        "peak2": (p2_idx, p2_price),
                        "trough": (trough_idx, trough_price),
                        "symmetry_pct": round(price_diff_pct * 100, 2),
                        "is_breakout_confirmed": is_confirmed
                    }
                ))

        return patterns

    def detect_double_bottoms(self, df: pd.DataFrame, troughs: Optional[List[Tuple[int, float]]] = None) -> List[ChartPattern]:
        """
        Detects Double Bottom (Bullish Reversal) formations with symmetry, neckline peak,
        and confirmation/breakout status.
        """
        n = len(df)
        if troughs is None:
            _, troughs = self.find_local_extrema(df)

        high = df['high'].values
        close = df['close'].values
        patterns: List[ChartPattern] = []

        if len(troughs) < 2:
            return patterns

        for i in range(len(troughs) - 1):
            t1_idx, t1_price = troughs[i]
            for j in range(i + 1, len(troughs)):
                t2_idx, t2_price = troughs[j]
                span = t2_idx - t1_idx

                if span < self.min_pattern_span:
                    continue
                if span > self.max_pattern_span:
                    break

                price_diff_pct = abs(t1_price - t2_price) / min(t1_price, t2_price)
                if price_diff_pct > self.double_top_tol:
                    continue

                intervening_highs = high[t1_idx : t2_idx + 1]
                peak_price = float(np.max(intervening_highs))
                peak_rel_idx = int(np.argmax(intervening_highs))
                peak_idx = t1_idx + peak_rel_idx

                if peak_idx == t1_idx or peak_idx == t2_idx:
                    continue

                max_trough = max(t1_price, t2_price)
                peak_height_pct = (peak_price - max_trough) / (max_trough + 1e-9)
                if peak_height_pct < 0.005:
                    continue

                lowest_trough = min(t1_price, t2_price)
                neckline = peak_price
                target = neckline + (neckline - lowest_trough)

                is_confirmed = False
                confirmed_idx = None
                for k in range(t2_idx + 1, n):
                    if close[k] > neckline:
                        is_confirmed = True
                        confirmed_idx = k
                        break
                    if close[k] < lowest_trough:
                        break

                symmetry_score = max(0.0, 1.0 - (price_diff_pct / self.double_top_tol))
                height_score = min(1.0, peak_height_pct / 0.03)
                quality = round(0.5 * symmetry_score + 0.3 * height_score + (0.2 if is_confirmed else 0.0), 2)

                patterns.append(ChartPattern(
                    name="DOUBLE_BOTTOM",
                    bias=BiasType.BULLISH,
                    quality_score=quality,
                    neckline_price=round(neckline, 4),
                    projected_target=round(target, 4),
                    invalidation_level=round(lowest_trough, 4),
                    candle_span=span,
                    start_index=t1_idx,
                    end_index=confirmed_idx if confirmed_idx else t2_idx,
                    metadata={
                        "trough1": (t1_idx, t1_price),
                        "trough2": (t2_idx, t2_price),
                        "peak": (peak_idx, peak_price),
                        "symmetry_pct": round(price_diff_pct * 100, 2),
                        "is_breakout_confirmed": is_confirmed
                    }
                ))

        return patterns

    # ------------------------------------------------------------------------
    # 2. Head and Shoulders (Standard & Inverse)
    # ------------------------------------------------------------------------
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> List[ChartPattern]:
        """
        Detects Head & Shoulders (Bearish) and Inverse Head & Shoulders (Bullish).
        """
        n = len(df)
        peaks, troughs = self.find_local_extrema(df)
        close = df['close'].values
        low = df['low'].values
        high = df['high'].values
        patterns: List[ChartPattern] = []

        # ----------------- Standard H&S (Bearish) -----------------
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                sl_idx, sl_price = peaks[i]
                h_idx, h_price = peaks[i + 1]
                sr_idx, sr_price = peaks[i + 2]

                span = sr_idx - sl_idx
                if span < 6 or span > 60:
                    continue

                # Head must be higher than both left and right shoulders
                if h_price <= sl_price or h_price <= sr_price:
                    continue

                # Shoulder symmetry
                shoulder_diff_pct = abs(sl_price - sr_price) / sl_price
                if shoulder_diff_pct > self.hs_symmetry_tol:
                    continue

                # Intervening troughs T1 and T2
                t1_segment = low[sl_idx : h_idx + 1]
                t2_segment = low[h_idx : sr_idx + 1]
                if len(t1_segment) < 2 or len(t2_segment) < 2:
                    continue

                t1_price = float(np.min(t1_segment))
                t2_price = float(np.min(t2_segment))
                neckline = (t1_price + t2_price) / 2.0

                # Target & Invalidation
                target = neckline - (h_price - neckline)
                invalidation = h_price

                # Check breakout confirmation
                is_confirmed = any(close[k] < neckline for k in range(sr_idx + 1, n))

                quality = round(max(0.0, 1.0 - (shoulder_diff_pct / self.hs_symmetry_tol)) * 0.7 + (0.3 if is_confirmed else 0.0), 2)

                patterns.append(ChartPattern(
                    name="HEAD_AND_SHOULDERS",
                    bias=BiasType.BEARISH,
                    quality_score=quality,
                    neckline_price=round(neckline, 4),
                    projected_target=round(target, 4),
                    invalidation_level=round(invalidation, 4),
                    candle_span=span,
                    start_index=sl_idx,
                    end_index=sr_idx,
                    metadata={
                        "left_shoulder": (sl_idx, sl_price),
                        "head": (h_idx, h_price),
                        "right_shoulder": (sr_idx, sr_price),
                        "neckline": neckline,
                        "is_breakout_confirmed": is_confirmed
                    }
                ))

        # ------------- Inverse Head & Shoulders (Bullish) -------------
        if len(troughs) >= 3:
            for i in range(len(troughs) - 2):
                sl_idx, sl_price = troughs[i]
                h_idx, h_price = troughs[i + 1]
                sr_idx, sr_price = troughs[i + 2]

                span = sr_idx - sl_idx
                if span < 6 or span > 60:
                    continue

                # Head must be lower than both shoulders
                if h_price >= sl_price or h_price >= sr_price:
                    continue

                shoulder_diff_pct = abs(sl_price - sr_price) / sl_price
                if shoulder_diff_pct > self.hs_symmetry_tol:
                    continue

                p1_segment = high[sl_idx : h_idx + 1]
                p2_segment = high[h_idx : sr_idx + 1]
                if len(p1_segment) < 2 or len(p2_segment) < 2:
                    continue

                p1_price = float(np.max(p1_segment))
                p2_price = float(np.max(p2_segment))
                neckline = (p1_price + p2_price) / 2.0

                target = neckline + (neckline - h_price)
                invalidation = h_price

                is_confirmed = any(close[k] > neckline for k in range(sr_idx + 1, n))

                quality = round(max(0.0, 1.0 - (shoulder_diff_pct / self.hs_symmetry_tol)) * 0.7 + (0.3 if is_confirmed else 0.0), 2)

                patterns.append(ChartPattern(
                    name="INVERSE_HEAD_AND_SHOULDERS",
                    bias=BiasType.BULLISH,
                    quality_score=quality,
                    neckline_price=round(neckline, 4),
                    projected_target=round(target, 4),
                    invalidation_level=round(invalidation, 4),
                    candle_span=span,
                    start_index=sl_idx,
                    end_index=sr_idx,
                    metadata={
                        "left_shoulder": (sl_idx, sl_price),
                        "head": (h_idx, h_price),
                        "right_shoulder": (sr_idx, sr_price),
                        "neckline": neckline,
                        "is_breakout_confirmed": is_confirmed
                    }
                ))

        return patterns

    # ------------------------------------------------------------------------
    # 3. Triangles & Wedges
    # ------------------------------------------------------------------------
    def detect_triangles_and_wedges(self, df: pd.DataFrame, window_len: int = 30) -> List[ChartPattern]:
        """
        Detects Ascending Triangle, Descending Triangle, Symmetrical Triangle,
        Rising Wedge, and Falling Wedge using linear regression trendlines.
        """
        n = len(df)
        if n < 10:
            return []

        patterns: List[ChartPattern] = []
        close = df['close'].values

        # Analyze rolling window of recent bars
        start_w = max(0, n - window_len)
        sub_df = df.iloc[start_w:].reset_index(drop=True)
        peaks, troughs = self.find_local_extrema(sub_df, window=1)

        # Map indices back to global df
        peaks = [(idx + start_w, p) for idx, p in peaks]
        troughs = [(idx + start_w, p) for idx, p in troughs]

        if len(peaks) < 2 or len(troughs) < 2:
            return patterns

        # Linear regression on peaks
        px = np.array([p[0] for p in peaks])
        py = np.array([p[1] for p in peaks])
        tx = np.array([t[0] for t in troughs])
        ty = np.array([t[1] for t in troughs])

        p_slope, p_intercept = np.polyfit(px, py, 1)
        t_slope, t_intercept = np.polyfit(tx, ty, 1)

        mean_price = float(np.mean(close[start_w:]))
        norm_p_slope = p_slope / (mean_price + 1e-9)
        norm_t_slope = t_slope / (mean_price + 1e-9)

        # Convergence test: distance narrows between start and end
        start_x = min(px[0], tx[0])
        end_x = max(px[-1], tx[-1])
        start_spread = (p_slope * start_x + p_intercept) - (t_slope * start_x + t_intercept)
        end_spread = (p_slope * end_x + p_intercept) - (t_slope * end_x + t_intercept)

        is_converging = start_spread > 0 and (end_spread < start_spread or end_spread <= 0)

        current_price = float(close[-1])
        span = int(end_x - start_x)

        # 1. Ascending Triangle: Flat resistance, higher lows
        if abs(norm_p_slope) <= 0.0008 and norm_t_slope > 0.0003 and is_converging:
            neckline = float(np.mean(py))
            target = neckline + start_spread
            patterns.append(ChartPattern(
                name="ASCENDING_TRIANGLE",
                bias=BiasType.BULLISH,
                quality_score=0.75,
                neckline_price=round(neckline, 4),
                projected_target=round(target, 4),
                invalidation_level=round(float(np.min(ty)), 4),
                candle_span=span,
                start_index=int(start_x),
                end_index=int(end_x),
                metadata={"p_slope": norm_p_slope, "t_slope": norm_t_slope}
            ))

        # 2. Descending Triangle: Flat support, lower highs
        elif abs(norm_t_slope) <= 0.0008 and norm_p_slope < -0.0003 and is_converging:
            neckline = float(np.mean(ty))
            target = neckline - start_spread
            patterns.append(ChartPattern(
                name="DESCENDING_TRIANGLE",
                bias=BiasType.BEARISH,
                quality_score=0.75,
                neckline_price=round(neckline, 4),
                projected_target=round(target, 4),
                invalidation_level=round(float(np.max(py)), 4),
                candle_span=span,
                start_index=int(start_x),
                end_index=int(end_x),
                metadata={"p_slope": norm_p_slope, "t_slope": norm_t_slope}
            ))

        # 3. Symmetrical Triangle: Lower highs and higher lows
        elif norm_p_slope < -0.0003 and norm_t_slope > 0.0003 and is_converging:
            neckline = float(close[-1])
            target_up = neckline + start_spread
            target_down = neckline - start_spread
            patterns.append(ChartPattern(
                name="SYMMETRICAL_TRIANGLE",
                bias=BiasType.NEUTRAL,
                quality_score=0.70,
                neckline_price=round(neckline, 4),
                projected_target=round(target_up if current_price >= neckline else target_down, 4),
                invalidation_level=round(float(np.min(ty)), 4),
                candle_span=span,
                start_index=int(start_x),
                end_index=int(end_x),
                metadata={"p_slope": norm_p_slope, "t_slope": norm_t_slope}
            ))

        # 4. Rising Wedge: Both sloping up, lower trendline steeper
        elif norm_p_slope > 0.0002 and norm_t_slope > 0.0002 and norm_t_slope > norm_p_slope and is_converging:
            neckline = float(np.min(ty))
            target = neckline - start_spread
            patterns.append(ChartPattern(
                name="RISING_WEDGE",
                bias=BiasType.BEARISH,
                quality_score=0.80,
                neckline_price=round(neckline, 4),
                projected_target=round(target, 4),
                invalidation_level=round(float(np.max(py)), 4),
                candle_span=span,
                start_index=int(start_x),
                end_index=int(end_x),
                metadata={"p_slope": norm_p_slope, "t_slope": norm_t_slope}
            ))

        # 5. Falling Wedge: Both sloping down, upper trendline steeper
        elif norm_p_slope < -0.0002 and norm_t_slope < -0.0002 and abs(norm_p_slope) > abs(norm_t_slope) and is_converging:
            neckline = float(np.max(py))
            target = neckline + start_spread
            patterns.append(ChartPattern(
                name="FALLING_WEDGE",
                bias=BiasType.BULLISH,
                quality_score=0.80,
                neckline_price=round(neckline, 4),
                projected_target=round(target, 4),
                invalidation_level=round(float(np.min(ty)), 4),
                candle_span=span,
                start_index=int(start_x),
                end_index=int(end_x),
                metadata={"p_slope": norm_p_slope, "t_slope": norm_t_slope}
            ))

        return patterns

    # ------------------------------------------------------------------------
    # 4. Classical Multi-Candle Patterns
    # ------------------------------------------------------------------------
    def detect_multi_candle_patterns(self, df: pd.DataFrame) -> List[ChartPattern]:
        """
        Detects:
          - Morning Star (Bullish Reversal)
          - Evening Star (Bearish Reversal)
          - Three White Soldiers (Strong Bullish Continuation)
          - Three Black Crows (Strong Bearish Continuation)
          - Fakey (False Breakout of Inside Bar)
        """
        n = len(df)
        if n < 3:
            return []

        open_p = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        patterns: List[ChartPattern] = []

        for t in range(2, n):
            c0_o, c0_h, c0_l, c0_c = open_p[t - 2], high[t - 2], low[t - 2], close[t - 2]
            c1_o, c1_h, c1_l, c1_c = open_p[t - 1], high[t - 1], low[t - 1], close[t - 1]
            c2_o, c2_h, c2_l, c2_c = open_p[t], high[t], low[t], close[t]

            rng0 = max(c0_h - c0_l, 1e-9)
            rng1 = max(c1_h - c1_l, 1e-9)
            rng2 = max(c2_h - c2_l, 1e-9)

            body0 = abs(c0_c - c0_o)
            body1 = abs(c1_c - c1_o)
            body2 = abs(c2_c - c2_o)

            # ---------------- Morning Star (Bullish) ----------------
            if c0_c < c0_o and (body0 / rng0 >= 0.4):
                if body1 <= 0.4 * body0:
                    midpoint0 = (c0_o + c0_c) / 2.0
                    if c2_c > c2_o and c2_c >= midpoint0:
                        patterns.append(ChartPattern(
                            name="MORNING_STAR",
                            bias=BiasType.BULLISH,
                            quality_score=0.85,
                            neckline_price=round(c0_o, 4),
                            projected_target=round(c2_c + (c2_c - min(c0_l, c1_l)), 4),
                            invalidation_level=round(min(c0_l, c1_l, c2_l), 4),
                            candle_span=3,
                            start_index=t - 2,
                            end_index=t,
                            metadata={"penetration": round((c2_c - c0_c) / body0, 2)}
                        ))

            # ---------------- Evening Star (Bearish) ----------------
            if c0_c > c0_o and (body0 / rng0 >= 0.4):
                if body1 <= 0.4 * body0:
                    midpoint0 = (c0_o + c0_c) / 2.0
                    if c2_c < c2_o and c2_c <= midpoint0:
                        patterns.append(ChartPattern(
                            name="EVENING_STAR",
                            bias=BiasType.BEARISH,
                            quality_score=0.85,
                            neckline_price=round(c0_o, 4),
                            projected_target=round(c2_c - (max(c0_h, c1_h) - c2_c), 4),
                            invalidation_level=round(max(c0_h, c1_h, c2_h), 4),
                            candle_span=3,
                            start_index=t - 2,
                            end_index=t,
                            metadata={"penetration": round((c0_c - c2_c) / body0, 2)}
                        ))

            # ----------- Three White Soldiers (Bullish) -----------
            if (c0_c > c0_o) and (c1_c > c1_o) and (c2_c > c2_o):
                if (c2_c > c1_c > c0_c) and (c2_h > c1_h > c0_h):
                    uw0 = (c0_h - c0_c) / rng0
                    uw1 = (c1_h - c1_c) / rng1
                    uw2 = (c2_h - c2_c) / rng2
                    if uw0 <= 0.35 and uw1 <= 0.35 and uw2 <= 0.35:
                        patterns.append(ChartPattern(
                            name="THREE_WHITE_SOLDIERS",
                            bias=BiasType.STRONG_BULLISH,
                            quality_score=0.90,
                            neckline_price=round(c2_c, 4),
                            projected_target=round(c2_c + (c2_c - c0_l), 4),
                            invalidation_level=round(c0_l, 4),
                            candle_span=3,
                            start_index=t - 2,
                            end_index=t
                        ))

            # ------------ Three Black Crows (Bearish) -------------
            if (c0_c < c0_o) and (c1_c < c1_o) and (c2_c < c2_o):
                if (c2_c < c1_c < c0_c) and (c2_l < c1_l < c0_l):
                    lw0 = (c0_c - c0_l) / rng0
                    lw1 = (c1_c - c1_l) / rng1
                    lw2 = (c2_c - c2_l) / rng2
                    if lw0 <= 0.35 and lw1 <= 0.35 and lw2 <= 0.35:
                        patterns.append(ChartPattern(
                            name="THREE_BLACK_CROWS",
                            bias=BiasType.STRONG_BEARISH,
                            quality_score=0.90,
                            neckline_price=round(c2_c, 4),
                            projected_target=round(c2_c - (c0_h - c2_c), 4),
                            invalidation_level=round(c0_h, 4),
                            candle_span=3,
                            start_index=t - 2,
                            end_index=t
                        ))

            # ----------------- Fakey (Inside Bar False Breakout) -----------------
            is_inside_bar = (c1_h <= c0_h) and (c1_l >= c0_l)
            if is_inside_bar:
                if c2_l < c1_l and c2_c > c1_l and c2_c > c2_o:
                    patterns.append(ChartPattern(
                        name="FAKEY_BULLISH",
                        bias=BiasType.BULLISH,
                        quality_score=0.80,
                        neckline_price=round(c1_h, 4),
                        projected_target=round(c1_h + (c1_h - c2_l), 4),
                        invalidation_level=round(c2_l, 4),
                        candle_span=3,
                        start_index=t - 2,
                        end_index=t
                    ))
                elif c2_h > c1_h and c2_c < c1_h and c2_c < c2_o:
                    patterns.append(ChartPattern(
                        name="FAKEY_BEARISH",
                        bias=BiasType.BEARISH,
                        quality_score=0.80,
                        neckline_price=round(c1_l, 4),
                        projected_target=round(c1_l - (c2_h - c1_l), 4),
                        invalidation_level=round(c2_h, 4),
                        candle_span=3,
                        start_index=t - 2,
                        end_index=t
                    ))

        return patterns

    # ------------------------------------------------------------------------
    # Master Detection Method
    # ------------------------------------------------------------------------
    def detect_all(self, df: pd.DataFrame) -> List[ChartPattern]:
        """
        Executes comprehensive pattern detection across all geometric and multi-candle types.
        """
        all_patterns: List[ChartPattern] = []
        all_patterns.extend(self.detect_double_tops(df))
        all_patterns.extend(self.detect_double_bottoms(df))
        all_patterns.extend(self.detect_head_and_shoulders(df))
        all_patterns.extend(self.detect_triangles_and_wedges(df))
        all_patterns.extend(self.detect_multi_candle_patterns(df))
        return all_patterns
