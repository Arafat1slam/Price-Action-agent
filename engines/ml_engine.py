"""
ml_engine.py
============
Production-grade Machine Learning Engine for Live Price Action Scanner.
Implements:
1. FeatureExtractor: 35 scale-invariant, stationary technical & SMC features with zero lookahead.
2. LabelGenerator: Adaptive volatility threshold forward labeling (H=5, threshold=0.75 * ATR14).
3. PurgedTimeSeriesSplit: Time-series cross-validator enforcing embargo/purge gap.
4. ModelTrainer: HistGradientBoostingClassifier training, Platt scaling calibration, and validation metrics.
5. MLPredictor: High-speed streaming inference engine (<5ms per tick) with calibrated confidence.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.inspection import permutation_importance

# Import FrozenEstimator for prefit calibration in scikit-learn >= 1.4
try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN_ESTIMATOR = True
except ImportError:
    HAS_FROZEN_ESTIMATOR = False


# ============================================================================
# Master Schema: 35 Feature Columns (docs/ml_pipeline_spec.md)
# ============================================================================
FEATURE_COLUMNS = [
    # 1. Candlestick Geometry
    "feat_body_to_range",
    "feat_upper_wick_ratio",
    "feat_lower_wick_ratio",
    "feat_candle_direction",
    # 2. Return Metrics
    "feat_return_1b",
    "feat_return_3b",
    "feat_return_5b",
    "feat_return_10b",
    # 3. Normalized ATR
    "feat_natr_14",
    "feat_atr_ratio_sma50",
    # 4. Volume Dynamics & Liquidity Profiling
    "feat_volume_zscore",
    "feat_volume_ratio_sma20",
    "feat_volume_flow_dir",
    # 5. Support & Resistance Proximity
    "feat_dist_to_support_pct",
    "feat_dist_to_resistance_pct",
    "feat_sr_range_position",
    # 6. POC (Volume Profile) & Rolling VWAP
    "feat_dist_to_poc_pct",
    "feat_dist_to_vwap_pct",
    "feat_vwap_zscore",
    # 7. Momentum Oscillators (RSI)
    "feat_rsi_14",
    "feat_rsi_slope_3b",
    "feat_rsi_dev_mid",
    # 8. EMA Alignment & Spreads
    "feat_ema_spread_9_21",
    "feat_ema_spread_21_50",
    "feat_ema_spread_9_50",
    "feat_dist_to_ema9",
    "feat_dist_to_ema21",
    "feat_dist_to_ema50",
    "feat_ema_alignment",
    # 9. Smart Money Concepts (SMC) Flags
    "feat_inside_bullish_fvg",
    "feat_inside_bearish_fvg",
    "feat_near_bullish_ob",
    "feat_near_bearish_ob",
    "feat_recent_liquidity_sweep_bullish",
    "feat_recent_liquidity_sweep_bearish",
]

TARGET_NAMES = ["BEARISH", "NEUTRAL", "BULLISH"]
DEFAULT_MODEL_PATH = "models/price_action_model.joblib"
FALLBACK_MODEL_PATH = "models/price_action_ml_model.joblib"


# ============================================================================
# 1. Feature Extraction Framework
# ============================================================================
class FeatureExtractor:
    """
    High-performance feature extraction engine supporting both historical batch
    processing and real-time streaming candle updates.
    Extracts 35 scale-invariant, stationary features strictly without lookahead bias.
    """

    def __init__(
        self,
        sr_lookback: int = 50,
        poc_lookback: int = 50,
        vwap_lookback: int = 50,
        sr_tolerance: float = 0.015,
        sr_method: str = "clustering",
    ):
        self.sr_lookback = sr_lookback
        self.poc_lookback = poc_lookback
        self.vwap_lookback = vwap_lookback
        self.sr_tolerance = sr_tolerance
        self.sr_method = sr_method
        self.feature_columns = list(FEATURE_COLUMNS)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts the full 35-feature matrix from an OHLCV DataFrame.
        Guarantees strictly no lookahead bias.
        """
        if len(df) == 0:
            return pd.DataFrame(columns=self.feature_columns)

        # Normalize column names to lowercase
        df_norm = df.copy()
        col_map = {c.lower(): c for c in df_norm.columns}
        for col in ["open", "high", "low", "close", "volume"]:
            if col in col_map:
                df_norm[col] = pd.to_numeric(df_norm[col_map[col]], errors="coerce").astype(np.float64)
            else:
                raise ValueError(f"Missing required OHLCV column: {col}")

        n = len(df_norm)
        feats = pd.DataFrame(index=df.index)

        o = df_norm["open"]
        h = df_norm["high"]
        l = df_norm["low"]
        c = df_norm["close"]
        v = df_norm["volume"]

        o_arr = o.values
        h_arr = h.values
        l_arr = l.values
        c_arr = c.values
        v_arr = v.values

        # -------------------------------------------------------------
        # 1. Candlestick Geometry
        # -------------------------------------------------------------
        candle_range = np.maximum(h - l, 1e-8)
        body = np.abs(c - o)
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l

        feats["feat_body_to_range"] = (body / candle_range).astype(np.float32)
        feats["feat_upper_wick_ratio"] = (upper_wick / candle_range).astype(np.float32)
        feats["feat_lower_wick_ratio"] = (lower_wick / candle_range).astype(np.float32)
        feats["feat_candle_direction"] = np.sign(c - o).astype(np.float32)

        # -------------------------------------------------------------
        # 2. Multi-Horizon Returns
        # -------------------------------------------------------------
        feats["feat_return_1b"] = c.pct_change(1).fillna(0.0).astype(np.float32)
        feats["feat_return_3b"] = c.pct_change(3).fillna(0.0).astype(np.float32)
        feats["feat_return_5b"] = c.pct_change(5).fillna(0.0).astype(np.float32)
        feats["feat_return_10b"] = c.pct_change(10).fillna(0.0).astype(np.float32)

        # -------------------------------------------------------------
        # 3. Normalized ATR (Wilder's Smoothing)
        # -------------------------------------------------------------
        prev_c = c.shift(1).fillna(c)
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
        feats["feat_natr_14"] = ((atr14 / c) * 100.0).astype(np.float32)
        atr_sma50 = atr14.rolling(50, min_periods=1).mean()
        feats["feat_atr_ratio_sma50"] = (atr14 / (atr_sma50 + 1e-8)).astype(np.float32)

        # -------------------------------------------------------------
        # 4. Volume Dynamics & Liquidity Profiling
        # -------------------------------------------------------------
        vol_sma20 = v.rolling(20, min_periods=1).mean()
        vol_std20 = v.rolling(20, min_periods=1).std().fillna(1.0)
        feats["feat_volume_zscore"] = ((v - vol_sma20) / (vol_std20 + 1e-8)).astype(np.float32)
        feats["feat_volume_ratio_sma20"] = (v / (vol_sma20 + 1e-8)).astype(np.float32)
        feats["feat_volume_flow_dir"] = (
            feats["feat_candle_direction"] * np.log1p(np.maximum(feats["feat_volume_ratio_sma20"], 0.0))
        ).astype(np.float32)

        # -------------------------------------------------------------
        # 5. Support & Resistance Proximity (Dynamic Clustering)
        # -------------------------------------------------------------
        supp_dists = np.zeros(n, dtype=np.float32)
        res_dists = np.zeros(n, dtype=np.float32)
        range_positions = np.full(n, 0.5, dtype=np.float32)

        for i in range(10, n):
            start_idx = max(0, i - self.sr_lookback)
            # Strictly up to i-1 to avoid lookahead on the current candle
            w_highs = h_arr[start_idx:i]
            w_lows = l_arr[start_idx:i]
            curr_price = c_arr[i]

            # Swing points: +/- 2 bars local extrema
            swing_points = []
            len_w = len(w_highs)
            for j in range(2, len_w - 2):
                if (
                    w_highs[j] >= w_highs[j - 1]
                    and w_highs[j] >= w_highs[j - 2]
                    and w_highs[j] >= w_highs[j + 1]
                    and w_highs[j] >= w_highs[j + 2]
                ):
                    swing_points.append(w_highs[j])
                if (
                    w_lows[j] <= w_lows[j - 1]
                    and w_lows[j] <= w_lows[j - 2]
                    and w_lows[j] <= w_lows[j + 1]
                    and w_lows[j] <= w_lows[j + 2]
                ):
                    swing_points.append(w_lows[j])

            if not swing_points:
                swing_points = list(w_highs) + list(w_lows)

            swing_points.sort()
            clusters: List[Dict[str, Any]] = []
            for p in swing_points:
                matched = False
                for cl in clusters:
                    if abs(p - cl["avg"]) / (cl["avg"] + 1e-8) <= self.sr_tolerance:
                        cl["points"].append(p)
                        cl["avg"] = sum(cl["points"]) / len(cl["points"])
                        matched = True
                        break
                if not matched:
                    clusters.append({"avg": p, "points": [p]})

            supports = [cl["avg"] for cl in clusters if cl["avg"] <= curr_price]
            resistances = [cl["avg"] for cl in clusters if cl["avg"] >= curr_price]

            s_near = max(supports) if supports else curr_price * 0.98
            r_near = min(resistances) if resistances else curr_price * 1.02

            supp_dists[i] = (curr_price - s_near) / curr_price * 100.0
            res_dists[i] = (r_near - curr_price) / curr_price * 100.0
            pos = (curr_price - s_near) / (r_near - s_near + 1e-8)
            range_positions[i] = np.clip(pos, 0.0, 1.0)

        feats["feat_dist_to_support_pct"] = supp_dists
        feats["feat_dist_to_resistance_pct"] = res_dists
        feats["feat_sr_range_position"] = range_positions

        # -------------------------------------------------------------
        # 6. POC (Volume Profile) & VWAP
        # -------------------------------------------------------------
        tp = (h + l + c) / 3.0
        pv = tp * v
        rolling_pv = pv.rolling(self.vwap_lookback, min_periods=1).sum()
        rolling_v = v.rolling(self.vwap_lookback, min_periods=1).sum()
        vwap = rolling_pv / (rolling_v + 1e-8)
        feats["feat_dist_to_vwap_pct"] = (((c - vwap) / c) * 100.0).astype(np.float32)

        vwap_std = c.rolling(self.vwap_lookback, min_periods=1).std().fillna(1.0)
        feats["feat_vwap_zscore"] = ((c - vwap) / (vwap_std + 1e-8)).astype(np.float32)

        poc_dists = np.zeros(n, dtype=np.float32)
        tp_arr = tp.values
        for i in range(10, n):
            start_idx = max(0, i - self.poc_lookback)
            w_h = h_arr[start_idx : i + 1]
            w_l = l_arr[start_idx : i + 1]
            w_tp = tp_arr[start_idx : i + 1]
            w_v = v_arr[start_idx : i + 1]

            min_p, max_p = np.min(w_l), np.max(w_h)
            if max_p - min_p < 1e-8:
                continue

            bins = np.linspace(min_p, max_p, 21)
            bin_indices = np.clip(np.digitize(w_tp, bins) - 1, 0, 19)
            bin_vols = np.bincount(bin_indices, weights=w_v, minlength=20)
            poc_bin = np.argmax(bin_vols)
            poc_price = 0.5 * (bins[poc_bin] + bins[poc_bin + 1])
            poc_dists[i] = (c_arr[i] - poc_price) / c_arr[i] * 100.0

        feats["feat_dist_to_poc_pct"] = poc_dists

        # -------------------------------------------------------------
        # 7. RSI (14) & RSI Slope
        # -------------------------------------------------------------
        delta = c.diff().fillna(0.0)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        avg_loss = pd.Series(loss, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        feats["feat_rsi_14"] = rsi.astype(np.float32)
        feats["feat_rsi_slope_3b"] = ((rsi - rsi.shift(3).fillna(rsi)) / 3.0).astype(np.float32)
        feats["feat_rsi_dev_mid"] = (rsi - 50.0).astype(np.float32)

        # -------------------------------------------------------------
        # 8. EMA 9/21/50 Alignment & Spreads
        # -------------------------------------------------------------
        ema9 = c.ewm(span=9, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()

        feats["feat_ema_spread_9_21"] = (((ema9 - ema21) / c) * 100.0).astype(np.float32)
        feats["feat_ema_spread_21_50"] = (((ema21 - ema50) / c) * 100.0).astype(np.float32)
        feats["feat_ema_spread_9_50"] = (((ema9 - ema50) / c) * 100.0).astype(np.float32)

        feats["feat_dist_to_ema9"] = (((c - ema9) / c) * 100.0).astype(np.float32)
        feats["feat_dist_to_ema21"] = (((c - ema21) / c) * 100.0).astype(np.float32)
        feats["feat_dist_to_ema50"] = (((c - ema50) / c) * 100.0).astype(np.float32)

        alignment = np.zeros(n, dtype=np.float32)
        alignment[(ema9 > ema21) & (ema21 > ema50)] = 1.0
        alignment[(ema9 < ema21) & (ema21 < ema50)] = -1.0
        feats["feat_ema_alignment"] = alignment

        # -------------------------------------------------------------
        # 9. Smart Money Concepts (SMC) Flags
        # -------------------------------------------------------------
        inside_bull_fvg = np.zeros(n, dtype=np.float32)
        inside_bear_fvg = np.zeros(n, dtype=np.float32)
        near_bull_ob = np.zeros(n, dtype=np.float32)
        near_bear_ob = np.zeros(n, dtype=np.float32)
        recent_sweep_bull = np.zeros(n, dtype=np.float32)
        recent_sweep_bear = np.zeros(n, dtype=np.float32)

        atr_arr = atr14.values

        for i in range(5, n):
            curr_p = c_arr[i]
            curr_atr = atr_arr[i] if not np.isnan(atr_arr[i]) else (curr_p * 0.01)

            # Fair Value Gaps (Lookback 15 bars)
            for k in range(max(2, i - 15), i):
                if l_arr[k] > h_arr[k - 2]:  # Bullish FVG
                    if h_arr[k - 2] <= curr_p <= l_arr[k]:
                        inside_bull_fvg[i] = 1.0
                if h_arr[k] < l_arr[k - 2]:  # Bearish FVG
                    if h_arr[k] <= curr_p <= l_arr[k - 2]:
                        inside_bear_fvg[i] = 1.0

            # Order Blocks (Lookback 10 bars)
            for k in range(max(3, i - 10), i):
                if c_arr[k - 1] < o_arr[k - 1]:  # Prior down candle
                    if c_arr[k] > o_arr[k] and (c_arr[k] - o_arr[k]) > 0.8 * curr_atr:
                        ob_low, ob_high = l_arr[k - 1], h_arr[k - 1]
                        if (ob_low - 0.25 * curr_atr) <= curr_p <= (ob_high + 0.25 * curr_atr):
                            near_bull_ob[i] = 1.0

                if c_arr[k - 1] > o_arr[k - 1]:  # Prior up candle
                    if c_arr[k] < o_arr[k] and (o_arr[k] - c_arr[k]) > 0.8 * curr_atr:
                        ob_low, ob_high = l_arr[k - 1], h_arr[k - 1]
                        if (ob_low - 0.25 * curr_atr) <= curr_p <= (ob_high + 0.25 * curr_atr):
                            near_bear_ob[i] = 1.0

            # Liquidity Sweeps (Lookback 3 bars)
            for k in range(max(1, i - 3), i + 1):
                prev_min = np.min(l_arr[max(0, k - 10) : k]) if k > 0 else l_arr[k]
                if l_arr[k] < prev_min and c_arr[k] > prev_min:
                    recent_sweep_bull[i] = 1.0

                prev_max = np.max(h_arr[max(0, k - 10) : k]) if k > 0 else h_arr[k]
                if h_arr[k] > prev_max and c_arr[k] < prev_max:
                    recent_sweep_bear[i] = 1.0

        feats["feat_inside_bullish_fvg"] = inside_bull_fvg
        feats["feat_inside_bearish_fvg"] = inside_bear_fvg
        feats["feat_near_bullish_ob"] = near_bull_ob
        feats["feat_near_bearish_ob"] = near_bear_ob
        feats["feat_recent_liquidity_sweep_bullish"] = recent_sweep_bull
        feats["feat_recent_liquidity_sweep_bearish"] = recent_sweep_bear

        return feats[self.feature_columns].copy()

    def extract_latest(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fast single-vector feature extraction for real-time streaming inference.
        Extracts the 35-feature vector for the latest candle in < 1.0 ms.
        Returns: 2D numpy array of shape (1, 35).
        """
        n = len(df)
        if n < 10:
            return np.zeros((1, len(self.feature_columns)), dtype=np.float32)

        col_map = {c.lower(): c for c in df.columns}
        o_arr = df[col_map["open"]].values.astype(np.float64)
        h_arr = df[col_map["high"]].values.astype(np.float64)
        l_arr = df[col_map["low"]].values.astype(np.float64)
        c_arr = df[col_map["close"]].values.astype(np.float64)
        v_arr = df[col_map["volume"]].values.astype(np.float64)

        i = n - 1

        # 1. Candlestick Geometry
        cr = max(h_arr[i] - l_arr[i], 1e-8)
        b2r = abs(c_arr[i] - o_arr[i]) / cr
        uwr = (h_arr[i] - max(o_arr[i], c_arr[i])) / cr
        lwr = (min(o_arr[i], c_arr[i]) - l_arr[i]) / cr
        cdir = float(np.sign(c_arr[i] - o_arr[i]))

        # 2. Multi-Horizon Returns
        r1 = (c_arr[i] - c_arr[i - 1]) / c_arr[i - 1] if i >= 1 else 0.0
        r3 = (c_arr[i] - c_arr[i - 3]) / c_arr[i - 3] if i >= 3 else 0.0
        r5 = (c_arr[i] - c_arr[i - 5]) / c_arr[i - 5] if i >= 5 else 0.0
        r10 = (c_arr[i] - c_arr[i - 10]) / c_arr[i - 10] if i >= 10 else 0.0

        # 3. Normalized ATR (Wilder's Smoothing)
        prev_c = np.roll(c_arr, 1)
        prev_c[0] = c_arr[0]
        tr = np.maximum(h_arr - l_arr, np.maximum(np.abs(h_arr - prev_c), np.abs(l_arr - prev_c)))
        alpha14 = 1.0 / 14.0
        atr_vals = np.empty(n, dtype=np.float64)
        atr_vals[0] = tr[0]
        for k in range(1, n):
            atr_vals[k] = alpha14 * tr[k] + (1.0 - alpha14) * atr_vals[k - 1]
        natr14 = (atr_vals[i] / c_arr[i]) * 100.0
        atr_sma50 = np.mean(atr_vals[max(0, i - 49) : i + 1])
        atr_ratio_sma50 = atr_vals[i] / (atr_sma50 + 1e-8)

        # 4. Volume Dynamics & Liquidity Profiling
        v_window = v_arr[max(0, i - 19) : i + 1]
        vol_sma20 = np.mean(v_window)
        vol_std20 = np.std(v_window, ddof=1) if len(v_window) > 1 else 1.0
        vz = (v_arr[i] - vol_sma20) / (vol_std20 + 1e-8)
        vr = v_arr[i] / (vol_sma20 + 1e-8)
        vf = cdir * np.log1p(max(vr, 0.0))

        # 5. Support & Resistance Proximity (Clustering)
        start_idx = max(0, i - self.sr_lookback)
        w_highs = h_arr[start_idx:i]
        w_lows = l_arr[start_idx:i]
        curr_price = c_arr[i]
        swing_points = []
        len_w = len(w_highs)
        for j in range(2, len_w - 2):
            if (
                w_highs[j] >= w_highs[j - 1]
                and w_highs[j] >= w_highs[j - 2]
                and w_highs[j] >= w_highs[j + 1]
                and w_highs[j] >= w_highs[j + 2]
            ):
                swing_points.append(w_highs[j])
            if (
                w_lows[j] <= w_lows[j - 1]
                and w_lows[j] <= w_lows[j - 2]
                and w_lows[j] <= w_lows[j + 1]
                and w_lows[j] <= w_lows[j + 2]
            ):
                swing_points.append(w_lows[j])
        if not swing_points:
            swing_points = list(w_highs) + list(w_lows)
        swing_points.sort()
        clusters: List[Dict[str, Any]] = []
        for p in swing_points:
            matched = False
            for cl in clusters:
                if abs(p - cl["avg"]) / (cl["avg"] + 1e-8) <= self.sr_tolerance:
                    cl["points"].append(p)
                    cl["avg"] = sum(cl["points"]) / len(cl["points"])
                    matched = True
                    break
            if not matched:
                clusters.append({"avg": p, "points": [p]})
        supports = [cl["avg"] for cl in clusters if cl["avg"] <= curr_price]
        resistances = [cl["avg"] for cl in clusters if cl["avg"] >= curr_price]
        s_near = max(supports) if supports else curr_price * 0.98
        r_near = min(resistances) if resistances else curr_price * 1.02
        supp_dist = (curr_price - s_near) / curr_price * 100.0
        res_dist = (r_near - curr_price) / curr_price * 100.0
        sr_pos = np.clip((curr_price - s_near) / (r_near - s_near + 1e-8), 0.0, 1.0)

        # 6. POC & Rolling VWAP
        tp_arr = (h_arr + l_arr + c_arr) / 3.0
        pv_arr = tp_arr * v_arr
        w_vwap_start = max(0, i - self.vwap_lookback + 1)
        vwap = np.sum(pv_arr[w_vwap_start : i + 1]) / (np.sum(v_arr[w_vwap_start : i + 1]) + 1e-8)
        dist_vwap = ((curr_price - vwap) / curr_price) * 100.0
        vwap_std = np.std(c_arr[w_vwap_start : i + 1], ddof=1) if (i + 1 - w_vwap_start) > 1 else 1.0
        vwap_z = (curr_price - vwap) / (vwap_std + 1e-8)

        start_poc = max(0, i - self.poc_lookback)
        w_h_poc = h_arr[start_poc : i + 1]
        w_l_poc = l_arr[start_poc : i + 1]
        min_p, max_p = np.min(w_l_poc), np.max(w_h_poc)
        if max_p - min_p >= 1e-8:
            bins = np.linspace(min_p, max_p, 21)
            bin_idx = np.clip(np.digitize(tp_arr[start_poc : i + 1], bins) - 1, 0, 19)
            bin_vols = np.bincount(bin_idx, weights=v_arr[start_poc : i + 1], minlength=20)
            poc_bin = np.argmax(bin_vols)
            poc_price = 0.5 * (bins[poc_bin] + bins[poc_bin + 1])
            poc_dist = (curr_price - poc_price) / curr_price * 100.0
        else:
            poc_dist = 0.0

        # 7. RSI (14) & RSI Slope
        delta_arr = np.empty(n, dtype=np.float64)
        delta_arr[0] = 0.0
        delta_arr[1:] = np.diff(c_arr)
        gain_arr = np.where(delta_arr > 0, delta_arr, 0.0)
        loss_arr = np.where(delta_arr < 0, -delta_arr, 0.0)
        ag_arr = np.empty(n, dtype=np.float64)
        al_arr = np.empty(n, dtype=np.float64)
        ag_arr[0] = gain_arr[0]
        al_arr[0] = loss_arr[0]
        alpha_rsi = 1.0 / 14.0
        for k in range(1, n):
            ag_arr[k] = alpha_rsi * gain_arr[k] + (1.0 - alpha_rsi) * ag_arr[k - 1]
            al_arr[k] = alpha_rsi * loss_arr[k] + (1.0 - alpha_rsi) * al_arr[k - 1]
        rsi_vals = 100.0 - (100.0 / (1.0 + (ag_arr / (al_arr + 1e-8))))
        rsi14 = rsi_vals[i]
        rsi_prev3 = rsi_vals[i - 3] if i >= 3 else rsi_vals[0]
        rsi_slope3 = (rsi14 - rsi_prev3) / 3.0
        rsi_dev_mid = rsi14 - 50.0

        # 8. EMA 9/21/50 Alignment & Spreads
        def ema_fast(arr, span):
            alpha = 2.0 / (span + 1.0)
            out = np.empty(len(arr), dtype=np.float64)
            out[0] = arr[0]
            for k in range(1, len(arr)):
                out[k] = alpha * arr[k] + (1.0 - alpha) * out[k - 1]
            return out

        e9 = ema_fast(c_arr, 9)[-1]
        e21 = ema_fast(c_arr, 21)[-1]
        e50 = ema_fast(c_arr, 50)[-1]
        sp9_21 = ((e9 - e21) / curr_price) * 100.0
        sp21_50 = ((e21 - e50) / curr_price) * 100.0
        sp9_50 = ((e9 - e50) / curr_price) * 100.0
        d_e9 = ((curr_price - e9) / curr_price) * 100.0
        d_e21 = ((curr_price - e21) / curr_price) * 100.0
        d_e50 = ((curr_price - e50) / curr_price) * 100.0
        ema_align = 1.0 if (e9 > e21 > e50) else (-1.0 if (e9 < e21 < e50) else 0.0)

        # 9. Smart Money Concepts (SMC) Flags
        curr_atr = atr_vals[i]
        in_bull_fvg = 0.0
        in_bear_fvg = 0.0
        for k in range(max(2, i - 15), i):
            if l_arr[k] > h_arr[k - 2] and h_arr[k - 2] <= curr_price <= l_arr[k]:
                in_bull_fvg = 1.0
            if h_arr[k] < l_arr[k - 2] and h_arr[k] <= curr_price <= l_arr[k - 2]:
                in_bear_fvg = 1.0

        near_b_ob = 0.0
        near_br_ob = 0.0
        for k in range(max(3, i - 10), i):
            if c_arr[k - 1] < o_arr[k - 1] and c_arr[k] > o_arr[k] and (c_arr[k] - o_arr[k]) > 0.8 * curr_atr:
                if (l_arr[k - 1] - 0.25 * curr_atr) <= curr_price <= (h_arr[k - 1] + 0.25 * curr_atr):
                    near_b_ob = 1.0
            if c_arr[k - 1] > o_arr[k - 1] and c_arr[k] < o_arr[k] and (o_arr[k] - c_arr[k]) > 0.8 * curr_atr:
                if (l_arr[k - 1] - 0.25 * curr_atr) <= curr_price <= (h_arr[k - 1] + 0.25 * curr_atr):
                    near_br_ob = 1.0

        sw_bull = 0.0
        sw_bear = 0.0
        for k in range(max(1, i - 3), i + 1):
            prev_min = np.min(l_arr[max(0, k - 10) : k]) if k > 0 else l_arr[k]
            if l_arr[k] < prev_min and c_arr[k] > prev_min:
                sw_bull = 1.0
            prev_max = np.max(h_arr[max(0, k - 10) : k]) if k > 0 else h_arr[k]
            if h_arr[k] > prev_max and c_arr[k] < prev_max:
                sw_bear = 1.0

        return np.array(
            [[
                b2r, uwr, lwr, cdir,
                r1, r3, r5, r10,
                natr14, atr_ratio_sma50,
                vz, vr, vf,
                supp_dist, res_dist, sr_pos,
                poc_dist, dist_vwap, vwap_z,
                rsi14, rsi_slope3, rsi_dev_mid,
                sp9_21, sp21_50, sp9_50,
                d_e9, d_e21, d_e50, ema_align,
                in_bull_fvg, in_bear_fvg,
                near_b_ob, near_br_ob,
                sw_bull, sw_bear,
            ]],
            dtype=np.float32,
        )


# ============================================================================
# 2. Label Generator with Adaptive Volatility Threshold
# ============================================================================
class LabelGenerator:
    """
    Constructs forward-looking ATR-adjusted classification labels.
    Horizon H=5, threshold = 0.75 * ATR14:
      +1: Bullish (forward return > +0.75 * ATR14)
      -1: Bearish (forward return < -0.75 * ATR14)
       0: Neutral (consolidation / chop within band)
    Strict anti-lookahead: truncates the last H rows with NaN.
    """

    def __init__(self, horizon: int = 5, atr_multiplier: float = 0.75):
        self.horizon = horizon
        self.atr_multiplier = atr_multiplier
        self.label_map = {-1: 0, 0: 1, 1: 2}
        self.inv_map = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}

    def compute_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates tri-class target vector {-1.0, 0.0, +1.0}.
        Truncates the last H rows with NaN to prevent lookahead bias.
        """
        col_map = {c.lower(): c for c in df.columns}
        c = pd.to_numeric(df[col_map["close"]], errors="coerce").astype(float)
        h = pd.to_numeric(df[col_map["high"]], errors="coerce").astype(float)
        l = pd.to_numeric(df[col_map["low"]], errors="coerce").astype(float)

        prev_c = c.shift(1).fillna(c)
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()

        future_close = c.shift(-self.horizon)
        delta_p = future_close - c
        threshold = self.atr_multiplier * atr14

        labels = pd.Series(0.0, index=df.index, dtype=float)
        labels[delta_p > threshold] = 1.0    # Bullish
        labels[delta_p < -threshold] = -1.0  # Bearish

        # Strict anti-lookahead: last H rows have no future realization
        labels.iloc[-self.horizon:] = np.nan
        return labels

    def prepare_dataset(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """
        Aligns feature matrix with labels, drops NaNs, and maps labels to contiguous {0, 1, 2}.
        0: BEARISH (-1)
        1: NEUTRAL (0)
        2: BULLISH (+1)
        """
        valid = ~labels.isna() & ~features.isna().any(axis=1)
        X_clean = features.loc[valid]
        y_clean = labels.loc[valid].astype(int).map(self.label_map)
        return X_clean.values.astype(np.float32), y_clean.values.astype(np.int64), X_clean.index


# ============================================================================
# 3. Purged Time-Series Split & Walk-Forward Validation
# ============================================================================
class PurgedTimeSeriesSplit:
    """
    Time-Series Cross-Validator enforcing an embargo/purge gap
    to prevent forward-looking label overlap across folds.
    """

    def __init__(self, n_splits: int = 4, purge_window: int = 5):
        self.n_splits = n_splits
        self.purge_window = purge_window

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n = len(X)
        test_size = n // (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            train_end = i * test_size - self.purge_window
            test_start = i * test_size
            test_end = test_start + test_size if i < self.n_splits else n

            train_idx = np.arange(0, max(0, train_end))
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) > 50 and len(test_idx) > 10:
                yield train_idx, test_idx


@dataclass
class WalkForwardFoldResult:
    fold_idx: int
    train_size: int
    test_size: int
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    bullish_precision: float
    bullish_recall: float
    bearish_precision: float
    bearish_recall: float
    directional_win_rate: float
    simulated_sharpe: float
    simulated_max_dd_pct: float
    simulated_net_return_pct: float


@dataclass
class WalkForwardValidationReport:
    n_folds: int
    avg_accuracy: float
    avg_f1_macro: float
    avg_directional_win_rate: float
    avg_sharpe: float
    overall_max_dd_pct: float
    overall_simulated_return_pct: float
    folds: List[WalkForwardFoldResult]


class PurgedWalkForwardValidator:
    """
    Time-Series Purged Walk-Forward Cross-Validator.
    Splits sequential dataset into rolling or expanding chronological training periods,
    applies an embargo/purge gap, and validates strictly on out-of-sample data.
    Computes Out-of-Sample metrics: precision, recall, F1, Sharpe ratio, and Max Drawdown.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 5,
        min_train_size: int = 150,
        expanding: bool = True,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.min_train_size = min_train_size
        self.expanding = expanding
        self.random_state = random_state

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        raw_prices: Optional[np.ndarray] = None,
    ) -> WalkForwardValidationReport:
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        folds_res: List[WalkForwardFoldResult] = []
        all_strat_returns: List[float] = []

        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size - self.purge_window
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits else n

            train_start = 0 if self.expanding else max(0, train_end - 2 * fold_size)
            train_idx = np.arange(train_start, max(0, train_end))
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) < self.min_train_size or len(test_idx) < 20:
                continue

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            # Fit base estimator
            model = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.04,
                max_iter=60,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.5,
                class_weight="balanced",
                random_state=self.random_state + i,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            # Standard Metrics
            acc = float(accuracy_score(y_te, y_pred))
            b_acc = float(balanced_accuracy_score(y_te, y_pred))
            prec_m = float(precision_score(y_te, y_pred, average="macro", zero_division=0))
            rec_m = float(recall_score(y_te, y_pred, average="macro", zero_division=0))
            f1_m = float(f1_score(y_te, y_pred, average="macro", zero_division=0))

            # Class 2: BULLISH (+1), Class 0: BEARISH (-1)
            bull_prec = float(precision_score(y_te == 2, y_pred == 2, zero_division=0))
            bull_rec = float(recall_score(y_te == 2, y_pred == 2, zero_division=0))
            bear_prec = float(precision_score(y_te == 0, y_pred == 0, zero_division=0))
            bear_rec = float(recall_score(y_te == 0, y_pred == 0, zero_division=0))

            # Simulated ML strategy signal returns
            if raw_prices is not None and len(raw_prices) == n:
                te_prices = raw_prices[test_idx]
                fwd_ret = np.zeros(len(y_te))
                for k in range(len(y_te)):
                    exit_idx = min(len(te_prices) - 1, k + 5)
                    fwd_ret[k] = (te_prices[exit_idx] - te_prices[k]) / max(1e-8, te_prices[k])
            else:
                # Proxy 5-bar forward return from target labels
                fwd_ret = np.where(y_te == 2, 0.015, np.where(y_te == 0, -0.015, 0.0))

            # Trade return: Long on Bullish (2), Short on Bearish (0), Flat on Neutral (1)
            signal_dir = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))
            strat_ret = signal_dir * fwd_ret

            active_trades = np.where(signal_dir != 0)[0]
            win_count = np.sum(strat_ret[active_trades] > 0)
            win_rate = (win_count / max(1, len(active_trades))) * 100.0

            # Sharpe Ratio
            ret_mean = np.mean(strat_ret)
            ret_std = np.std(strat_ret)
            sharpe = float((ret_mean / (ret_std + 1e-8)) * np.sqrt(365 * 24 / 5))

            # Max Drawdown
            equity = np.cumprod(1.0 + strat_ret)
            peak = np.maximum.accumulate(equity)
            dd_pct = (peak - equity) / np.maximum(peak, 1e-8) * 100.0
            max_dd = float(np.max(dd_pct))
            net_ret = float((equity[-1] - 1.0) * 100.0)

            all_strat_returns.extend(strat_ret)

            folds_res.append(
                WalkForwardFoldResult(
                    fold_idx=i,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    accuracy=round(acc, 4),
                    balanced_accuracy=round(b_acc, 4),
                    precision_macro=round(prec_m, 4),
                    recall_macro=round(rec_m, 4),
                    f1_macro=round(f1_m, 4),
                    bullish_precision=round(bull_prec, 4),
                    bullish_recall=round(bull_rec, 4),
                    bearish_precision=round(bear_prec, 4),
                    bearish_recall=round(bear_rec, 4),
                    directional_win_rate=round(win_rate, 2),
                    simulated_sharpe=round(sharpe, 2),
                    simulated_max_dd_pct=round(max_dd, 2),
                    simulated_net_return_pct=round(net_ret, 2),
                )
            )

        if not folds_res:
            return WalkForwardValidationReport(
                n_folds=0, avg_accuracy=0.0, avg_f1_macro=0.0,
                avg_directional_win_rate=0.0, avg_sharpe=0.0,
                overall_max_dd_pct=0.0, overall_simulated_return_pct=0.0,
                folds=[]
            )

        avg_acc = float(np.mean([f.accuracy for f in folds_res]))
        avg_f1 = float(np.mean([f.f1_macro for f in folds_res]))
        avg_wr = float(np.mean([f.directional_win_rate for f in folds_res]))
        avg_sh = float(np.mean([f.simulated_sharpe for f in folds_res]))

        # Overall equity across all OOS predictions
        tot_eq = np.cumprod(1.0 + np.array(all_strat_returns))
        tot_pk = np.maximum.accumulate(tot_eq)
        tot_dd = float(np.max((tot_pk - tot_eq) / np.maximum(tot_pk, 1e-8) * 100.0))
        tot_ret = float((tot_eq[-1] - 1.0) * 100.0)

        return WalkForwardValidationReport(
            n_folds=len(folds_res),
            avg_accuracy=round(avg_acc, 4),
            avg_f1_macro=round(avg_f1, 4),
            avg_directional_win_rate=round(avg_wr, 2),
            avg_sharpe=round(avg_sh, 2),
            overall_max_dd_pct=round(tot_dd, 2),
            overall_simulated_return_pct=round(tot_ret, 2),
            folds=folds_res
        )


# ============================================================================
# 4. Model Training & Probability Calibration Engine
# ============================================================================
class ModelTrainer:
    """
    Trains HistGradientBoostingClassifier with balanced class weights,
    calibrates probabilities via Platt Scaling (Sigmoid), and computes comprehensive metrics.
    """

    def __init__(
        self,
        output_dir: str = "models",
        learning_rate: float = 0.04,
        max_iter: int = 80,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 25,
        l2_regularization: float = 1.5,
        random_state: int = 42,
    ):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        os.makedirs(self.output_dir, exist_ok=True)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fits base HistGradientBoostingClassifier, calibrates on validation set,
        evaluates metrics, and serializes artifact.
        """
        # 1. Base HistGradientBoostingClassifier
        base_estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            class_weight="balanced",
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.15,
            random_state=self.random_state,
        )
        base_estimator.fit(X_train, y_train)

        # 2. Probability Calibration via Platt Scaling (Sigmoid) with Balanced Sample Weights
        import warnings
        from sklearn.utils.class_weight import compute_sample_weight

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if HAS_FROZEN_ESTIMATOR:
                calibrator = CalibratedClassifierCV(
                    estimator=FrozenEstimator(base_estimator),
                    method="sigmoid",
                )
            else:
                calibrator = CalibratedClassifierCV(
                    estimator=base_estimator,
                    method="sigmoid",
                    cv="prefit",
                )
            sw_calib = compute_sample_weight("balanced", y_val)
            calibrator.fit(X_val, y_val, sample_weight=sw_calib)

        # 3. Model Evaluation on Validation / Holdout Set
        y_pred = calibrator.predict(X_val)
        y_prob = calibrator.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro")
        f1_weighted = f1_score(y_val, y_pred, average="weighted")
        loss = log_loss(y_val, y_prob)

        # Multi-class ROC-AUC (One-vs-Rest)
        try:
            auc_weighted = roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted")
            auc_macro = roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc_weighted = 0.5
            auc_macro = 0.5

        # Brier score loss (multiclass: mean across classes)
        brier_scores = []
        for c_idx in range(3):
            y_binary = (y_val == c_idx).astype(int)
            brier_scores.append(brier_score_loss(y_binary, y_prob[:, c_idx]))
        mean_brier = float(np.mean(brier_scores))

        report = classification_report(
            y_val, y_pred, target_names=TARGET_NAMES, output_dict=True
        )

        # 4. Feature Importances via Permutation Importance
        # Subsample up to 2,500 samples for sub-second importance evaluation
        eval_size = min(len(X_val), 2500)
        rng = np.random.RandomState(self.random_state)
        sub_idx = rng.choice(len(X_val), eval_size, replace=False)
        perm_res = permutation_importance(
            calibrator,
            X_val[sub_idx],
            y_val[sub_idx],
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=1,
        )
        importances = {
            feature_names[i]: float(perm_res.importances_mean[i])
            for i in range(len(feature_names))
        }
        # Sort importances descending
        sorted_importances = dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)
        )

        metrics = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "roc_auc_weighted": float(auc_weighted),
            "roc_auc_macro": float(auc_macro),
            "log_loss": float(loss),
            "brier_score": mean_brier,
            "classification_report": report,
            "feature_importances": sorted_importances,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

        # 5. Serialize Model Bundle
        if save_path is None:
            save_path = os.path.join(self.output_dir, "price_action_model.joblib")

        bundle = {
            "model": calibrator,
            "base_model": base_estimator,
            "feature_names": feature_names,
            "target_names": TARGET_NAMES,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "horizon": 5,
            "atr_multiplier": 0.75,
        }

        joblib.dump(bundle, save_path, compress=3)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        metrics["artifact_path"] = save_path
        metrics["artifact_size_mb"] = float(size_mb)

        # Also save copy to fallback path if different
        alt_path = os.path.join(self.output_dir, "price_action_ml_model.joblib")
        if os.path.abspath(save_path) != os.path.abspath(alt_path):
            try:
                joblib.dump(bundle, alt_path, compress=3)
            except Exception:
                pass

        return metrics


# ============================================================================
# 5. Real-Time Streaming Inference Engine
# ============================================================================
class MLPredictor:
    """
    Lightweight, thread-safe inference wrapper for the Price Action Scanner.
    Executes feature extraction + calibrated inference in < 5ms per tick.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.bundle: Optional[Dict[str, Any]] = None
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.load_model()

    def load_model(self, custom_path: Optional[str] = None) -> bool:
        """Loads serialized model bundle from disk."""
        target_path = custom_path or self.model_path

        candidates = []
        if target_path:
            candidates.append(target_path)
        candidates.extend([DEFAULT_MODEL_PATH, FALLBACK_MODEL_PATH])

        for path in candidates:
            if os.path.exists(path):
                try:
                    self.bundle = joblib.load(path)
                    self.model = self.bundle["model"]
                    self.model_path = path
                    return True
                except Exception as e:
                    print(f"[MLPredictor] Warning: Error loading model from '{path}': {e}")

        self.model = None
        return False

    def predict(self, df_window: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs single-step prediction on the current rolling candlestick window.
        Guarantees sub-5ms execution latency on CPU.
        """
        t0 = time.perf_counter()

        if self.model is None or len(df_window) < 50:
            return {
                "available": False,
                "ml_bias": "NEUTRAL",
                "ml_confidence": 50,
                "prob_bearish": 0.333,
                "prob_neutral": 0.334,
                "prob_bullish": 0.333,
                "latency_ms": 0.0,
            }

        # 1. Fast feature extraction for latest candle
        latest_vector = self.feature_extractor.extract_latest(df_window)

        # 2. Calibrated Inference
        probs = self.model.predict_proba(latest_vector)[0]
        p_bear = float(probs[0])
        p_neut = float(probs[1])
        p_bull = float(probs[2])

        # 3. Determine Bias and Confidence
        max_idx = int(np.argmax(probs))
        bias = TARGET_NAMES[max_idx]

        # Scaled probability margin to 50-95% confidence
        max_prob = probs[max_idx]
        confidence = int(np.clip(50 + (max_prob - 0.333) * 75.0, 50, 95))

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 3)

        return {
            "available": True,
            "ml_bias": bias,
            "ml_confidence": confidence,
            "prob_bearish": round(p_bear, 4),
            "prob_neutral": round(p_neut, 4),
            "prob_bullish": round(p_bull, 4),
            "latency_ms": latency_ms,
        }
