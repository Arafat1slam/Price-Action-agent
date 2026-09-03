"""
engines/confluence_engine.py - Multi-Timeframe Confluence Synthesis Engine
Combines:
  1. Smart Money Concepts (FVG, Order Blocks, Liquidity Sweeps, BOS/CHoCH)
  2. Advanced Analytical Indicators (Volume Profile POC/VAH/VAL, VWAP bands, Wyckoff VSA, KDE S/R, Fibonacci)
  3. Classical & Geometric Chart Patterns (Double Top/Bottom, H&S, Triangles, Wedges, Multi-candle)
  4. Machine Learning Probability Model (Calibrated HistGradientBoosting / Heuristic inference)
  5. Multi-Timeframe Weighting (HTF 4h/1d directional alignment + LTF 15m/1h triggers)
Evaluates calibrated institutional confidence score (50% - 95%) and directional bias.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from core.models import (
    BiasType,
    Candle,
    ChartPattern,
    ConfluenceReport,
    FairValueGap as CanonicalFVG,
    KDELevel,
    MarketStructureState,
    MLInferenceResult,
    OrderBlock as CanonicalOB,
    SetupType,
    TimeframeConfluence,
    VolumeProfileZone,
)
from engines.smc_engine import (
    SMCEngine,
    SMCAnalysisReport,
    FVGType,
    OBType,
    SweepType,
    SMCTrend,
)
from engines.indicators_engine import (
    VolumeProfileEngine,
    VWAPEngine,
    VSAEngine,
    KDESupportResistanceEngine,
    FibonacciEngine,
    IndicatorConfluenceEngine,
)
from engines.chart_pattern_engine import ChartPatternEngine
from engines.ml_predictor import MLPredictor


class ConfluenceEngine:
    """
    Multi-Timeframe Confluence Synthesis Engine.
    Synthesizes structural, statistical, pattern-based, and ML signals into
    a unified confidence score (50% - 95%) and institutional bias.
    """

    # Multi-factor weights as defined in architecture specification
    WEIGHT_SMC = 0.25
    WEIGHT_HTF = 0.20
    WEIGHT_SR_VP = 0.15
    WEIGHT_PAT = 0.15
    WEIGHT_ML = 0.15
    WEIGHT_VOL_MOM = 0.10

    def __init__(
        self,
        smc_engine: Optional[SMCEngine] = None,
        chart_pattern_engine: Optional[ChartPatternEngine] = None,
        ml_predictor: Optional[MLPredictor] = None,
        vp_engine: Optional[VolumeProfileEngine] = None,
        vwap_engine: Optional[VWAPEngine] = None,
        vsa_engine: Optional[VSAEngine] = None,
        kde_engine: Optional[KDESupportResistanceEngine] = None,
        fib_engine: Optional[FibonacciEngine] = None,
    ):
        self.smc_engine = smc_engine or SMCEngine()
        self.chart_pattern_engine = chart_pattern_engine or ChartPatternEngine()
        self.ml_predictor = ml_predictor or MLPredictor()
        self.vp_engine = vp_engine or VolumeProfileEngine()
        self.vwap_engine = vwap_engine or VWAPEngine()
        self.vsa_engine = vsa_engine or VSAEngine()
        self.kde_engine = kde_engine or KDESupportResistanceEngine()
        self.fib_engine = fib_engine or FibonacciEngine()
        self.indicator_confluence_engine = IndicatorConfluenceEngine()

    def analyze(
        self,
        symbol: str,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        current_price: Optional[float] = None,
        is_closed: bool = True,
        primary_timeframe: str = "1h",
    ) -> ConfluenceReport:
        """
        Executes comprehensive multi-timeframe confluence analysis.
        Args:
            symbol: Trading pair (e.g. BTCUSDT)
            data: Single primary DataFrame or dict of {timeframe: DataFrame}
            current_price: Optional explicit current price
            is_closed: Whether current candle is closed
            primary_timeframe: Primary execution timeframe
        """
        # 1. Normalize Multi-Timeframe Data
        tf_data: Dict[str, pd.DataFrame] = {}
        if isinstance(data, dict):
            tf_data = data
            primary_df = tf_data.get(primary_timeframe)
            if primary_df is None and tf_data:
                # Fallback to first available timeframe
                primary_timeframe, primary_df = next(iter(tf_data.items()))
        else:
            primary_df = data
            tf_data[primary_timeframe] = primary_df
            # Synthesize or derive HTF if bars are sufficient and datetime index exists
            self._populate_derived_timeframes(primary_df, primary_timeframe, tf_data)

        if primary_df is None or len(primary_df) < 5:
            # Insufficient data fallback
            return self._build_empty_report(symbol, current_price or 0.0, is_closed)

        df = primary_df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        curr_price = float(df["close"].iloc[-1]) if current_price is None else float(current_price)
        curr_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else datetime.now(timezone.utc)

        # 2. Smart Money Concepts (SMC) Analysis
        smc_report = self.smc_engine.analyze(df)
        s_smc, relied_feature_tested = self._evaluate_smc_score(smc_report, curr_price)

        # 3. Higher Timeframe (HTF) Alignment
        s_htf, htf_bias, tf_breakdowns = self._evaluate_htf_alignment(tf_data, primary_timeframe, df)

        # 4. Advanced Indicators (Volume Profile, VWAP, KDE S/R, Fibonacci, VSA)
        s_sr_vp, vp_zone = self._evaluate_indicators_score(df, curr_price)

        # 5. Classical Chart Patterns
        detected_patterns = self.chart_pattern_engine.detect_all(df)
        s_pat = self._evaluate_chart_patterns_score(detected_patterns)

        # 6. Machine Learning Inference
        ml_res = self.ml_predictor.predict(df)
        s_ml = float(np.clip(ml_res.prob_bullish - ml_res.prob_bearish, -1.0, 1.0))

        # 7. Volume & Momentum Verification
        s_vol_mom = self._evaluate_vol_mom_score(df)

        # 8. Raw Confluence Score Calculation
        s_raw = (
            self.WEIGHT_SMC * s_smc +
            self.WEIGHT_HTF * s_htf +
            self.WEIGHT_SR_VP * s_sr_vp +
            self.WEIGHT_PAT * s_pat +
            self.WEIGHT_ML * s_ml +
            self.WEIGHT_VOL_MOM * s_vol_mom
        )
        s_raw = float(np.clip(s_raw, -1.0, 1.0))

        # 9. Dynamic Multipliers
        # a) Forming candle uncertainty penalty
        m_forming = 1.0 if is_closed else 0.88

        # b) HTF counter-trend penalty
        m_htf_conflict = 1.0
        if (s_raw > 0 and s_htf <= -0.5) or (s_raw < 0 and s_htf >= 0.5):
            m_htf_conflict = 0.50

        # c) Multi-signal confluence booster (at least 4 out of 6 strictly agree on sign)
        components = [s_smc, s_htf, s_sr_vp, s_pat, s_ml, s_vol_mom]
        aligned_count = 0
        if abs(s_raw) > 0.05:
            target_sign = 1 if s_raw > 0 else -1
            aligned_count = sum(1 for c in components if (c > 0.05 if target_sign == 1 else c < -0.05))

        m_confluence = 1.15 if aligned_count >= 4 else 1.0

        # d) Mitigation decay (if relied feature tested >= 2 times)
        m_mitigation = 0.70 if relied_feature_tested else 1.0

        total_multiplier = m_forming * m_htf_conflict * m_confluence * m_mitigation
        s_adjusted = float(np.clip(s_raw * total_multiplier, -1.0, 1.0))

        # 10. Calibrated Confidence Score (50% - 95%) & Final Bias
        confidence_score = int(np.clip(50 + 45.0 * abs(s_adjusted), 50, 95))

        if s_adjusted >= 0.65:
            overall_bias = BiasType.STRONG_BULLISH
        elif s_adjusted >= 0.15:
            overall_bias = BiasType.BULLISH
        elif s_adjusted <= -0.65:
            overall_bias = BiasType.STRONG_BEARISH
        elif s_adjusted <= -0.15:
            overall_bias = BiasType.BEARISH
        else:
            overall_bias = BiasType.NEUTRAL

        # LTF Execution Trigger
        ltf_trigger = BiasType.NEUTRAL
        if s_smc >= 0.20 or s_vol_mom >= 0.30:
            ltf_trigger = BiasType.BULLISH
        elif s_smc <= -0.20 or s_vol_mom <= -0.30:
            ltf_trigger = BiasType.BEARISH

        # Canonical SMC state & conversions
        canonical_state = self._build_canonical_smc_state(smc_report, df)
        canonical_fvgs = self._to_canonical_fvgs(smc_report.active_bullish_fvgs + smc_report.active_bearish_fvgs)
        canonical_obs = self._to_canonical_obs(smc_report.active_bullish_obs + smc_report.active_bearish_obs)

        report = ConfluenceReport(
            symbol=symbol,
            timestamp=curr_ts,
            current_price=round(curr_price, 4),
            overall_bias=overall_bias,
            confidence_score=confidence_score,
            htf_bias=htf_bias,
            ltf_trigger=ltf_trigger,
            timeframe_breakdown=tf_breakdowns,
            smc_state=canonical_state,
            active_patterns=detected_patterns,
            active_fvgs=canonical_fvgs,
            active_obs=canonical_obs,
            volume_profile=vp_zone,
            ml_prediction=ml_res,
        )

        # Attach diagnostic metrics for UI & backtesting
        setattr(report, "raw_score", round(s_raw, 4))
        setattr(report, "adjusted_score", round(s_adjusted, 4))
        setattr(report, "component_scores", {
            "smc": round(s_smc, 3),
            "htf": round(s_htf, 3),
            "sr_vp": round(s_sr_vp, 3),
            "patterns": round(s_pat, 3),
            "ml": round(s_ml, 3),
            "vol_mom": round(s_vol_mom, 3)
        })
        setattr(report, "multipliers", {
            "m_forming": round(m_forming, 2),
            "m_htf_conflict": round(m_htf_conflict, 2),
            "m_confluence": round(m_confluence, 2),
            "m_mitigation": round(m_mitigation, 2),
            "total": round(total_multiplier, 3)
        })

        return report

    # ------------------------------------------------------------------------
    # Component Evaluators
    # ------------------------------------------------------------------------

    def _evaluate_smc_score(self, smc_report: SMCAnalysisReport, current_price: float) -> Tuple[float, bool]:
        """
        Calculates S_SMC component:
        S_SMC = 0.35 * I_BOS_CHOCH + 0.30 * I_OB_Touch + 0.25 * I_FVG_Consequent + 0.10 * I_Sweep
        """
        # 1. BOS / CHoCH
        i_struct = 0.0
        if smc_report.recent_structure_events:
            last_event = smc_report.recent_structure_events[-1]
            if "BULLISH" in last_event.event_type.value:
                i_struct = 1.0
            elif "BEARISH" in last_event.event_type.value:
                i_struct = -1.0
        elif smc_report.trend == SMCTrend.BULLISH:
            i_struct = 0.7
        elif smc_report.trend == SMCTrend.BEARISH:
            i_struct = -0.7

        # 2. OB Touch
        i_ob = 0.0
        relied_feature_tested = False
        for ob in smc_report.active_bullish_obs:
            if ob.bottom * 0.998 <= current_price <= ob.top * 1.005:
                i_ob = 1.0
                if ob.touches >= 2:
                    relied_feature_tested = True
                break

        if i_ob == 0.0:
            for ob in smc_report.active_bearish_obs:
                if ob.bottom * 0.995 <= current_price <= ob.top * 1.002:
                    i_ob = -1.0
                    if ob.touches >= 2:
                        relied_feature_tested = True
                    break

        # 3. FVG Consequent Encroachment Retest
        i_fvg = 0.0
        for fvg in smc_report.active_bullish_fvgs:
            if fvg.bottom * 0.998 <= current_price <= fvg.top * 1.002:
                # Near CE 50%
                dist_to_ce = abs(current_price - fvg.ce) / max(1e-6, fvg.initial_size)
                i_fvg = 1.0 if dist_to_ce <= 0.5 else 0.7
                if fvg.mitigation_percentage >= 0.5:
                    relied_feature_tested = True
                break

        if i_fvg == 0.0:
            for fvg in smc_report.active_bearish_fvgs:
                if fvg.bottom * 0.998 <= current_price <= fvg.top * 1.002:
                    dist_to_ce = abs(current_price - fvg.ce) / max(1e-6, fvg.initial_size)
                    i_fvg = -1.0 if dist_to_ce <= 0.5 else -0.7
                    if fvg.mitigation_percentage >= 0.5:
                        relied_feature_tested = True
                    break

        # 4. Liquidity Sweeps
        i_sweep = 0.0
        if smc_report.recent_sweeps:
            last_sweep = smc_report.recent_sweeps[-1]
            if last_sweep.sweep_type == SweepType.SSL_SWEEP:
                # Sell-side sweep is bullish reversal
                i_sweep = 1.0
            elif last_sweep.sweep_type == SweepType.BSL_SWEEP:
                # Buy-side sweep is bearish reversal
                i_sweep = -1.0

        s_smc = (0.35 * i_struct) + (0.30 * i_ob) + (0.25 * i_fvg) + (0.10 * i_sweep)
        return float(np.clip(s_smc, -1.0, 1.0)), relied_feature_tested

    def _evaluate_htf_alignment(
        self,
        tf_data: Dict[str, pd.DataFrame],
        primary_tf: str,
        primary_df: pd.DataFrame
    ) -> Tuple[float, BiasType, Dict[str, TimeframeConfluence]]:
        """
        Calculates S_HTF component:
        S_HTF = 0.60 * Bias_4h + 0.40 * Bias_1d
        Also produces detailed TimeframeConfluence breakdowns.
        """
        breakdowns: Dict[str, TimeframeConfluence] = {}
        all_tfs = ["1d", "4h", "1h", "15m"]

        scores = {}
        for tf in all_tfs:
            df_tf = tf_data.get(tf)
            if df_tf is not None and len(df_tf) >= 5:
                score, bias, signals = self._analyze_single_tf(df_tf, tf)
            else:
                # Estimate from primary_df using multi-scale analysis if timeframe missing
                score, bias, signals = self._estimate_tf_from_primary(primary_df, tf)

            scores[tf] = score
            breakdowns[tf] = TimeframeConfluence(
                timeframe=tf,
                bias=bias,
                score=round(score, 3),
                key_signals=signals
            )

        # S_HTF = 0.60 * Bias_4h + 0.40 * Bias_1d
        s_htf = 0.60 * scores.get("4h", 0.0) + 0.40 * scores.get("1d", 0.0)
        s_htf = float(np.clip(s_htf, -1.0, 1.0))

        if s_htf >= 0.50:
            htf_bias = BiasType.STRONG_BULLISH
        elif s_htf >= 0.15:
            htf_bias = BiasType.BULLISH
        elif s_htf <= -0.50:
            htf_bias = BiasType.STRONG_BEARISH
        elif s_htf <= -0.15:
            htf_bias = BiasType.BEARISH
        else:
            htf_bias = BiasType.NEUTRAL

        return s_htf, htf_bias, breakdowns

    def _analyze_single_tf(self, df: pd.DataFrame, tf_name: str) -> Tuple[float, BiasType, List[str]]:
        """Analyzes a specific timeframe dataframe for trend, EMA position, and structure."""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(df)
        signals = []

        curr_c = close[-1]
        span_fast = min(20, n)
        span_slow = min(50, n)
        ema_fast = pd.Series(close).ewm(span=span_fast).mean().iloc[-1]
        ema_slow = pd.Series(close).ewm(span=span_slow).mean().iloc[-1]

        score = 0.0
        if curr_c > ema_fast and ema_fast >= ema_slow:
            score += 0.50
            signals.append(f"Price above {span_fast}/{span_slow} EMA")
        elif curr_c < ema_fast and ema_fast <= ema_slow:
            score -= 0.50
            signals.append(f"Price below {span_fast}/{span_slow} EMA")
        elif curr_c > ema_fast:
            score += 0.20
        elif curr_c < ema_fast:
            score -= 0.20

        # Swing HH/HL or LH/LL
        if n >= 10:
            half = n // 2
            h1 = np.max(high[:half])
            h2 = np.max(high[half:])
            l1 = np.min(low[:half])
            l2 = np.min(low[half:])

            if h2 > h1 and l2 > l1:
                score += 0.50
                signals.append("Uptrend (HH/HL Structure)")
            elif h2 < h1 and l2 < l1:
                score -= 0.50
                signals.append("Downtrend (LH/LL Structure)")
            else:
                signals.append("Range / Consolidation")

        score = float(np.clip(score, -1.0, 1.0))
        if score >= 0.5:
            bias = BiasType.STRONG_BULLISH if score >= 0.75 else BiasType.BULLISH
        elif score <= -0.5:
            bias = BiasType.STRONG_BEARISH if score <= -0.75 else BiasType.BEARISH
        else:
            bias = BiasType.NEUTRAL

        return score, bias, signals

    def _estimate_tf_from_primary(self, df: pd.DataFrame, tf_name: str) -> Tuple[float, BiasType, List[str]]:
        """Estimates HTF/LTF context from primary DataFrame via multi-scale windowing."""
        close = df["close"].values
        n = len(df)
        curr_c = close[-1]
        signals = []

        if tf_name == "1d":
            mult = min(n, 120)
            ema = pd.Series(close).ewm(span=mult).mean().iloc[-1]
            score = 0.7 if curr_c > ema else -0.7
            signals.append("Macro EMA Position" if curr_c > ema else "Macro EMA Resistance")
        elif tf_name == "4h":
            mult = min(n, 40)
            ema = pd.Series(close).ewm(span=mult).mean().iloc[-1]
            score = 0.6 if curr_c > ema else -0.6
            signals.append("Intermediate Trend Support" if curr_c > ema else "Intermediate Trend Resistance")
        elif tf_name == "15m":
            mult = min(n, 10)
            ema = pd.Series(close).ewm(span=mult).mean().iloc[-1]
            score = 0.5 if curr_c > ema else -0.5
            signals.append("Execution Momentum Aligned" if curr_c > ema else "Execution Momentum Opposed")
        else:
            score = 0.0
            signals.append("Baseline Structure")

        score = float(np.clip(score, -1.0, 1.0))
        bias = BiasType.BULLISH if score > 0.2 else (BiasType.BEARISH if score < -0.2 else BiasType.NEUTRAL)
        return score, bias, signals

    def _populate_derived_timeframes(self, primary_df: pd.DataFrame, primary_tf: str, tf_data: Dict[str, pd.DataFrame]):
        """Derives higher timeframe resampled frames if primary dataframe has datetime index or timestamp."""
        if "timestamp" not in primary_df.columns or len(primary_df) < 50:
            return
        try:
            df_ts = primary_df.copy()
            df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
            df_ts = df_ts.set_index("timestamp")

            ohlc_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }
            if "4h" not in tf_data:
                res_4h = df_ts.resample("4h").agg(ohlc_dict).dropna().reset_index()
                if len(res_4h) >= 5:
                    tf_data["4h"] = res_4h
            if "1d" not in tf_data:
                res_1d = df_ts.resample("1D").agg(ohlc_dict).dropna().reset_index()
                if len(res_1d) >= 5:
                    tf_data["1d"] = res_1d
        except Exception:
            pass

    def _evaluate_indicators_score(self, df: pd.DataFrame, current_price: float) -> Tuple[float, Optional[VolumeProfileZone]]:
        """
        Calculates S_SR_VP component via Volume Profile, VWAP, KDE S/R, and Fibonacci.
        """
        score = 0.0
        vp_zone: Optional[VolumeProfileZone] = None

        try:
            # 1. Volume Profile
            vp_res = self.vp_engine.compute_profile(df)
            vp_zone = VolumeProfileZone(
                poc_price=round(vp_res.poc_price, 4),
                vah_price=round(vp_res.vah_price, 4),
                val_price=round(vp_res.val_price, 4),
                total_volume=round(vp_res.total_volume, 2),
                hvn_levels=[round(h, 4) for h in vp_res.hvn_levels],
                lvn_levels=[round(l, 4) for l in vp_res.lvn_levels],
            )

            # Position in Value Area: near VAL is bullish bounce; near VAH is bearish rejection
            va_range = max(1e-6, vp_res.vah_price - vp_res.val_price)
            if current_price <= vp_res.val_price * 1.008:
                score += 0.50  # At or near Value Area Low
            elif current_price >= vp_res.vah_price * 0.992:
                score -= 0.50  # At or near Value Area High
            elif abs(current_price - vp_res.poc_price) / current_price <= 0.005:
                # At POC - check trend
                score += 0.20 if current_price >= vp_res.poc_price else -0.20
            else:
                rel_pos = (current_price - vp_res.val_price) / va_range
                score += (0.5 - rel_pos) * 0.5

            # 2. VWAP
            vwap_res = self.vwap_engine.compute(df)
            if len(vwap_res.vwap) > 0:
                vwap_val = float(vwap_res.vwap.iloc[-1])
                l1 = float(vwap_res.lower_1sigma.iloc[-1])
                u1 = float(vwap_res.upper_1sigma.iloc[-1])
                l2 = float(vwap_res.lower_2sigma.iloc[-1])
                u2 = float(vwap_res.upper_2sigma.iloc[-1])

                if current_price <= l2:
                    score += 0.40  # Oversold / Exhaustion band bounce
                elif current_price >= u2:
                    score -= 0.40  # Overbought / Exhaustion band rejection
                elif current_price >= vwap_val:
                    score += 0.15
                else:
                    score -= 0.15

            # 3. Fibonacci Golden Pocket
            fib_res = self.fib_engine.compute_retracements(df)
            if fib_res.golden_pocket_low <= current_price <= fib_res.golden_pocket_high:
                score += 0.35 if fib_res.direction == "UP" else -0.35

        except Exception:
            score = 0.0

        return float(np.clip(score, -1.0, 1.0)), vp_zone

    def _evaluate_chart_patterns_score(self, patterns: List[ChartPattern]) -> float:
        """
        Calculates S_PAT component:
        S_PAT = sum_k (Quality_k * Direction_k), clamped to [-1.0, 1.0]
        """
        if not patterns:
            return 0.0

        total_score = 0.0
        for pat in patterns:
            direction = 1.0 if pat.bias in [BiasType.BULLISH, BiasType.STRONG_BULLISH] else (
                -1.0 if pat.bias in [BiasType.BEARISH, BiasType.STRONG_BEARISH] else 0.0
            )
            total_score += pat.quality_score * direction

        return float(np.clip(total_score, -1.0, 1.0))

    def _evaluate_vol_mom_score(self, df: pd.DataFrame) -> float:
        """
        Calculates S_VOL_MOM component:
        S_VOL_MOM = tanh((RVOL - 1.0) / 1.5) * sign(Momentum)
        """
        if len(df) < 5:
            return 0.0

        vol = df["volume"].values
        close = df["close"].values
        open_ = df["open"].values

        n = len(df)
        sma_len = min(20, n)
        avg_vol = np.mean(vol[-sma_len:])
        curr_vol = vol[-1]
        rvol = curr_vol / max(1e-8, avg_vol)

        # Momentum direction: 3-period price return + current candle direction
        ret3 = (close[-1] - close[-min(4, n)]) / max(1e-8, close[-min(4, n)])
        curr_dir = np.sign(close[-1] - open_[-1])
        mom_dir = 1.0 if (ret3 > 0 or (ret3 == 0 and curr_dir > 0)) else -1.0

        vol_scale = math.tanh((rvol - 1.0) / 1.5)
        s_vol_mom = vol_scale * mom_dir
        return float(np.clip(s_vol_mom, -1.0, 1.0))

    # ------------------------------------------------------------------------
    # Model Mappings & Canonical Converters
    # ------------------------------------------------------------------------

    def _build_canonical_smc_state(self, smc_report: SMCAnalysisReport, df: pd.DataFrame) -> MarketStructureState:
        """Constructs canonical MarketStructureState."""
        highs = df["high"].values
        lows = df["low"].values
        sh = float(np.max(highs[-15:])) if len(highs) >= 15 else float(np.max(highs))
        sl = float(np.min(lows[-15:])) if len(lows) >= 15 else float(np.min(lows))

        recent_bos = None
        recent_choch = None
        for evt in reversed(smc_report.recent_structure_events):
            evt_str = evt.event_type.value
            if "BOS" in evt_str and recent_bos is None:
                recent_bos = evt_str
            elif "CHOCH" in evt_str and recent_choch is None:
                recent_choch = evt_str

        trend_map = {
            SMCTrend.BULLISH: "UPTREND",
            SMCTrend.BEARISH: "DOWNTREND",
            SMCTrend.RANGING: "RANGE"
        }
        return MarketStructureState(
            trend=trend_map.get(smc_report.trend, "RANGE"),
            recent_bos=recent_bos,
            recent_choch=recent_choch,
            last_swing_high=sh,
            last_swing_low=sl
        )

    def _to_canonical_fvgs(self, fvgs: List[Any]) -> List[CanonicalFVG]:
        """Converts internal SMC FVGs to canonical core.models.FairValueGap."""
        canonical = []
        for f in fvgs:
            bias = BiasType.BULLISH if f.gap_type == FVGType.BULLISH else BiasType.BEARISH
            canonical.append(
                CanonicalFVG(
                    top=round(f.top, 4),
                    bottom=round(f.bottom, 4),
                    midpoint=round(f.ce, 4),
                    bias=bias,
                    created_time=f.creation_timestamp,
                    is_mitigated=(f.state.value != "UNMITIGATED"),
                    mitigation_pct=round(f.mitigation_percentage * 100.0, 1),
                    volume_delta=0.0
                )
            )
        return canonical

    def _to_canonical_obs(self, obs: List[Any]) -> List[CanonicalOB]:
        """Converts internal SMC Order Blocks to canonical core.models.OrderBlock."""
        canonical = []
        for o in obs:
            bias = BiasType.BULLISH if o.ob_type == OBType.BULLISH else BiasType.BEARISH
            canonical.append(
                CanonicalOB(
                    top=round(o.top, 4),
                    bottom=round(o.bottom, 4),
                    bias=bias,
                    created_time=o.timestamp,
                    volume=round(o.volume, 2),
                    is_mitigated=(o.state.value in ["TESTED", "INVALIDATED", "BREAKER"]),
                    is_invalidated=(o.state.value == "INVALIDATED")
                )
            )
        return canonical

    def _build_empty_report(self, symbol: str, current_price: float, is_closed: bool) -> ConfluenceReport:
        """Returns neutral placeholder report when data is insufficient."""
        return ConfluenceReport(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            current_price=round(current_price, 4),
            overall_bias=BiasType.NEUTRAL,
            confidence_score=50,
            htf_bias=BiasType.NEUTRAL,
            ltf_trigger=BiasType.NEUTRAL,
            timeframe_breakdown={},
            smc_state=None,
            active_patterns=[],
            active_fvgs=[],
            active_obs=[],
            volume_profile=None,
            ml_prediction=None
        )
