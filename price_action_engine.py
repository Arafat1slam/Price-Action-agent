import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from core.models import (
    BiasType,
    ChartPattern,
    ConfluenceReport,
    MLInferenceResult,
    SetupType,
    TradeSetup,
)
from engines.smc_engine import SMCEngine, SMCAnalysisReport
from engines.chart_pattern_engine import ChartPatternEngine
from engines.indicators_engine import (
    VolumeProfileEngine,
    VWAPEngine,
    VSAEngine,
    KDESupportResistanceEngine,
    FibonacciEngine,
    IndicatorConfluenceEngine,
)
from engines.ml_predictor import MLPredictor
from engines.confluence_engine import ConfluenceEngine
from engines.trade_setup_engine import TradeSetupEngine

@dataclass
class PatternSignal:
    name: str
    bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float  # 0.0 to 1.0
    description: str

@dataclass
class SupportResistanceLevel:
    price: float
    touches: int
    level_type: str  # "SUPPORT" or "RESISTANCE"
    distance_pct: float

@dataclass
class AnalysisResult:
    symbol: str
    timeframe: str
    current_price: float
    is_candle_closed: bool
    bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: int  # 0 - 100 %
    patterns: List[str]
    trend: str
    trend_detail: str
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]
    rough_target: Optional[float]
    volume_state: str
    momentum_streak: int
    momentum_state: str
    timestamp: str
    confluence_report: Optional[ConfluenceReport] = None
    trade_setup: Optional[TradeSetup] = None
    smc_report: Optional[SMCAnalysisReport] = None
    chart_patterns: List[ChartPattern] = field(default_factory=list)
    ml_result: Optional[MLInferenceResult] = None

    def to_cli_display(self) -> Dict[str, Any]:
        status_label = "Confirmed closed candle" if self.is_candle_closed else "Live candle forming"
        pattern_str = ", ".join(self.patterns) if self.patterns else "None significant"
        
        support_str = f"{self.nearest_support:,.2f}" if self.nearest_support else "N/A"
        resistance_str = f"{self.nearest_resistance:,.2f}" if self.nearest_resistance else "N/A"
        target_str = f"{self.rough_target:,.2f}" if self.rough_target else "N/A"

        return {
            "timestamp": self.timestamp,
            "status": status_label,
            "price": f"{self.current_price:,.2f}",
            "bias": self.bias,
            "confidence": f"{self.confidence}%",
            "patterns": pattern_str,
            "trend": self.trend_detail,
            "support": support_str,
            "resistance": resistance_str,
            "target": target_str,
            "volume": self.volume_state,
            "momentum": self.momentum_state,
            "trade_setup": self.trade_setup
        }

class PriceActionEngine:
    """
    Core rule-based & statistical Price Action Analysis Engine.
    Maintains rolling candlestick window and evaluates patterns, trend, S/R, volume & momentum.
    """

    def __init__(self, max_candles: int = 150):
        self.max_candles = max_candles
        self.df: pd.DataFrame = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "is_closed"]
        )
        self.mtf_buffers: Dict[str, pd.DataFrame] = {}
        self.smc_engine = SMCEngine()
        self.chart_pattern_engine = ChartPatternEngine()
        self.ml_predictor = MLPredictor()
        self.confluence_engine = ConfluenceEngine(
            smc_engine=self.smc_engine,
            chart_pattern_engine=self.chart_pattern_engine,
            ml_predictor=self.ml_predictor,
        )
        self.trade_setup_engine = TradeSetupEngine()

    def set_history(self, df: pd.DataFrame, timeframe: str = "1h"):
        """Seed the engine with historical candles for a specific or default timeframe."""
        if df.empty:
            return
        df_copy = df.tail(self.max_candles).copy().reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").astype(float)
        self.df = df_copy
        self.mtf_buffers[timeframe.lower()] = df_copy

    def set_multi_history(self, dfs: Dict[str, pd.DataFrame], primary_tf: str = "1h"):
        """Seed the engine with multi-timeframe candle datasets (e.g. 5m, 15m, 1h, 4h)."""
        self.mtf_buffers = {}
        for tf, df in dfs.items():
            if df is not None and not df.empty:
                df_copy = df.tail(self.max_candles).copy().reset_index(drop=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df_copy.columns:
                        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").astype(float)
                self.mtf_buffers[tf.lower()] = df_copy

        p_tf = primary_tf.lower()
        if p_tf in self.mtf_buffers:
            self.df = self.mtf_buffers[p_tf]
        elif self.mtf_buffers:
            first_tf = next(iter(self.mtf_buffers))
            self.df = self.mtf_buffers[first_tf]

    def update_candle(self, candle: Dict[str, Any], timeframe: Optional[str] = None):
        """
        Updates the engine's rolling buffer with a live streaming candle update.
        Maintains both primary buffer and multi-timeframe buffers (5m, 15m, 1h, 4h).
        """
        intv = (timeframe or candle.get("interval") or "1h").lower()
        if intv not in self.mtf_buffers:
            self.mtf_buffers[intv] = pd.DataFrame([candle])
        else:
            tf_df = self.mtf_buffers[intv]
            if not tf_df.empty:
                last_idx = len(tf_df) - 1
                last_ts = tf_df.loc[last_idx, "timestamp"]
                if candle["timestamp"] == last_ts:
                    for key in ["open", "high", "low", "close", "volume", "is_closed"]:
                        if key in candle:
                            tf_df.loc[last_idx, key] = candle[key]
                elif candle["timestamp"] > last_ts:
                    new_row = pd.DataFrame([candle])
                    self.mtf_buffers[intv] = pd.concat([tf_df, new_row], ignore_index=True)
                    if len(self.mtf_buffers[intv]) > self.max_candles:
                        self.mtf_buffers[intv] = self.mtf_buffers[intv].iloc[-self.max_candles:].reset_index(drop=True)
            else:
                self.mtf_buffers[intv] = pd.DataFrame([candle])

        # Also update primary df
        if self.df.empty:
            new_row = pd.DataFrame([candle])
            self.df = new_row
            return

        last_idx = len(self.df) - 1
        last_ts = self.df.loc[last_idx, "timestamp"]
        if candle["timestamp"] == last_ts:
            for key in ["open", "high", "low", "close", "volume", "is_closed"]:
                if key in candle:
                    self.df.loc[last_idx, key] = candle[key]
        elif candle["timestamp"] > last_ts:
            new_row = pd.DataFrame([candle])
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            if len(self.df) > self.max_candles:
                self.df = self.df.iloc[-self.max_candles:].reset_index(drop=True)

    def analyze(self, symbol: str, timeframe: str) -> Optional[AnalysisResult]:
        """Runs full price action analysis on the current rolling buffer."""
        if len(self.df) < 5:
            return None

        df = self.df.copy()
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        current_price = float(curr["close"])
        is_closed = bool(curr.get("is_closed", False))
        ts_str = pd.to_datetime(curr["timestamp"]).strftime("%H:%M:%S")

        # 1. Candlestick Pattern Detection
        pattern_signals = self._detect_patterns(curr, prev, is_closed)

        # 2. Trend Structure Analysis
        trend_info = self._detect_trend(df)

        # 3. Support & Resistance Detection
        sr_info = self._detect_support_resistance(df, current_price)

        # 4. Volume Confirmation
        vol_info = self._detect_volume(df, curr)

        # 5. Momentum / Continuation
        mom_info = self._detect_momentum(df)

        # 6. Signal Synthesis & Confidence Score
        synthesis = self._synthesize(
            patterns=pattern_signals,
            trend=trend_info,
            sr=sr_info,
            volume=vol_info,
            momentum=mom_info,
            current_price=current_price,
            is_closed=is_closed
        )

        pattern_names = [p.name for p in pattern_signals]

        # 7. Advanced Engines Synthesis (Confluence, SMC, Patterns, ML, Trade Setup)
        confluence_rep: Optional[ConfluenceReport] = None
        trade_setup: Optional[TradeSetup] = None
        smc_rep: Optional[SMCAnalysisReport] = None
        chart_pats: List[ChartPattern] = []
        ml_res: Optional[MLInferenceResult] = None

        try:
            confluence_data = self.mtf_buffers if self.mtf_buffers else df
            confluence_rep = self.confluence_engine.analyze(
                symbol=symbol,
                data=confluence_data,
                current_price=current_price,
                is_closed=is_closed,
                primary_timeframe=timeframe,
            )
            trade_setup = self.trade_setup_engine.generate_setup(
                confluence_report=confluence_rep,
                df=df,
                symbol=symbol,
            )
            smc_rep = self.smc_engine.analyze(df)
            chart_pats = self.chart_pattern_engine.detect_all(df)
            ml_res = self.ml_predictor.predict(df)
        except Exception:
            pass

        final_bias = synthesis["bias"]
        final_confidence = synthesis["confidence"]
        if confluence_rep is not None:
            final_confidence = max(synthesis["confidence"], confluence_rep.confidence_score)

        return AnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            is_candle_closed=is_closed,
            bias=final_bias,
            confidence=final_confidence,
            patterns=pattern_names,
            trend=trend_info["trend"],
            trend_detail=trend_info["detail"],
            nearest_support=sr_info.get("nearest_support"),
            nearest_resistance=sr_info.get("nearest_resistance"),
            rough_target=synthesis["target"],
            volume_state=vol_info["state"],
            momentum_streak=mom_info["streak"],
            momentum_state=mom_info["state"],
            timestamp=ts_str,
            confluence_report=confluence_rep,
            trade_setup=trade_setup,
            smc_report=smc_rep,
            chart_patterns=chart_pats,
            ml_result=ml_res,
        )

    def _detect_patterns(self, curr: pd.Series, prev: pd.Series, is_closed: bool) -> List[PatternSignal]:
        signals: List[PatternSignal] = []
        suffix = "" if is_closed else " (forming)"

        o, h, l, c = float(curr["open"]), float(curr["high"]), float(curr["low"]), float(curr["close"])
        prev_o, prev_h, prev_l, prev_c = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])

        candle_range = max(h - l, 1e-9)
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        prev_body = abs(prev_c - prev_o)

        # 1. Doji Detection
        if body <= 0.12 * candle_range:
            if lower_wick >= 0.5 * candle_range and upper_wick <= 0.15 * candle_range:
                signals.append(PatternSignal(f"Dragonfly Doji{suffix}", "BULLISH", 0.7, "Long lower shadow rejection"))
            elif upper_wick >= 0.5 * candle_range and lower_wick <= 0.15 * candle_range:
                signals.append(PatternSignal(f"Gravestone Doji{suffix}", "BEARISH", 0.7, "Long upper shadow rejection"))
            else:
                signals.append(PatternSignal(f"Doji{suffix}", "NEUTRAL", 0.5, "Indecision candle"))

        # 2. Pin Bar / Long Wick Rejection
        if lower_wick >= 0.60 * candle_range and body <= 0.30 * candle_range:
            signals.append(PatternSignal(f"Bullish Pin Bar{suffix}", "BULLISH", 0.8, "Bullish rejection of lower prices"))
        elif upper_wick >= 0.60 * candle_range and body <= 0.30 * candle_range:
            signals.append(PatternSignal(f"Bearish Pin Bar{suffix}", "BEARISH", 0.8, "Bearish rejection of higher prices"))

        # 3. Hammer & Inverted Hammer
        elif lower_wick >= 2.0 * body and upper_wick <= 0.25 * body and body >= 0.10 * candle_range:
            signals.append(PatternSignal(f"Hammer{suffix}", "BULLISH", 0.75, "Bullish hammer pattern"))
        elif upper_wick >= 2.0 * body and lower_wick <= 0.25 * body and body >= 0.10 * candle_range:
            signals.append(PatternSignal(f"Inverted Hammer{suffix}", "BEARISH", 0.75, "Inverted hammer rejection"))

        # 4. Bullish / Bearish Engulfing
        if prev_c < prev_o and c > o:  # Red followed by Green
            if c >= prev_o and o <= prev_c and body > prev_body:
                signals.append(PatternSignal(f"Bullish Engulfing{suffix}", "BULLISH", 0.85, "Bullish engulfing candle"))
        elif prev_c > prev_o and c < o:  # Green followed by Red
            if o >= prev_c and c <= prev_o and body > prev_body:
                signals.append(PatternSignal(f"Bearish Engulfing{suffix}", "BEARISH", 0.85, "Bearish engulfing candle"))

        # 5. Inside Bar & Outside Bar
        if h <= prev_h and l >= prev_l:
            signals.append(PatternSignal(f"Inside Bar{suffix}", "NEUTRAL", 0.6, "Consolidation inside previous bar range"))
        elif h >= prev_h and l <= prev_l and body > prev_body:
            direction = "BULLISH" if c > o else "BEARISH"
            signals.append(PatternSignal(f"Outside Bar{suffix}", direction, 0.7, "Volatility expansion bar"))

        return signals

    def _detect_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates swing highs/lows and classifies trend as Uptrend (HH/HL),
        Downtrend (LH/LL), or Range/Sideways.
        """
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        # Identify swing highs and swing lows (3-period window)
        swing_highs = []
        swing_lows = []

        for i in range(2, n - 2):
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                swing_lows.append(lows[i])

        # Moving average trend check
        ema20 = pd.Series(closes).ewm(span=min(20, n)).mean().iloc[-1]
        ema50 = pd.Series(closes).ewm(span=min(50, n)).mean().iloc[-1]

        trend = "SIDEWAYS"
        detail = "Range / Sideways"
        strength = 0.5

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1] > swing_highs[-2]
            hl = swing_lows[-1] > swing_lows[-2]
            lh = swing_highs[-1] < swing_highs[-2]
            ll = swing_lows[-1] < swing_lows[-2]

            if hh and hl and ema20 >= ema50:
                trend = "UPTREND"
                detail = "Uptrend (HH/HL)"
                strength = 0.85
            elif lh and ll and ema20 <= ema50:
                trend = "DOWNTREND"
                detail = "Downtrend (LH/LL)"
                strength = 0.85
            elif hh and hl:
                trend = "UPTREND"
                detail = "Uptrend (HH/HL leaning)"
                strength = 0.70
            elif lh and ll:
                trend = "DOWNTREND"
                detail = "Downtrend (LH/LL leaning)"
                strength = 0.70

        if trend == "SIDEWAYS":
            if ema20 >= ema50 * 1.01 and closes[-1] > closes[0]:
                trend = "UPTREND"
                detail = "Uptrend (Strong momentum)"
                strength = 0.80
            elif ema20 <= ema50 * 0.99 and closes[-1] < closes[0]:
                trend = "DOWNTREND"
                detail = "Downtrend (Strong momentum)"
                strength = 0.80
            elif ema20 > ema50 * 1.002:
                detail = "Sideways (Bullish bias)"
            elif ema20 < ema50 * 0.998:
                detail = "Sideways (Bearish bias)"

        return {"trend": trend, "detail": detail, "strength": strength, "ema20": ema20, "ema50": ema50}

    def _detect_support_resistance(self, df: pd.DataFrame, current_price: float, tolerance_pct: float = 0.015) -> Dict[str, Any]:
        """
        Clusters swing highs and lows into support and resistance levels.
        """
        highs = df["high"].tolist()
        lows = df["low"].tolist()

        points = highs + lows
        if not points:
            return {"nearest_support": None, "nearest_resistance": None, "levels": []}

        # Cluster points within tolerance
        points = sorted(points)
        clusters = []
        for p in points:
            matched = False
            for c in clusters:
                if abs(p - c["avg"]) / c["avg"] <= tolerance_pct:
                    c["points"].append(p)
                    c["avg"] = sum(c["points"]) / len(c["points"])
                    matched = True
                    break
            if not matched:
                clusters.append({"avg": p, "points": [p]})

        # Filter significant levels (at least 2 touches)
        significant = [c for c in clusters if len(c["points"]) >= 2]
        if not significant:
            significant = clusters

        levels: List[SupportResistanceLevel] = []
        for c in significant:
            price = c["avg"]
            level_type = "RESISTANCE" if price > current_price else "SUPPORT"
            dist_pct = abs(price - current_price) / current_price * 100.0
            levels.append(SupportResistanceLevel(price=price, touches=len(c["points"]), level_type=level_type, distance_pct=dist_pct))

        supports = [l.price for l in levels if l.level_type == "SUPPORT"]
        resistances = [l.price for l in levels if l.level_type == "RESISTANCE"]

        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None

        return {
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "levels": levels
        }

    def _detect_volume(self, df: pd.DataFrame, curr: pd.Series) -> Dict[str, Any]:
        """
        Compares current volume to 20-period volume SMA.
        """
        vol_series = df["volume"]
        sma_len = min(20, len(vol_series))
        avg_vol = vol_series.iloc[-sma_len:].mean()
        curr_vol = float(curr["volume"])

        rvol = curr_vol / (avg_vol + 1e-9)

        if rvol >= 1.5:
            state = f"Volume Spike ({rvol:.1f}x avg)"
            score = 1.0
        elif rvol >= 1.1:
            state = f"Volume Rising ({rvol:.1f}x avg)"
            score = 0.75
        elif rvol < 0.6:
            state = "Volume Low / Drying Up"
            score = 0.3
        else:
            state = "Volume Normal"
            score = 0.5

        return {"state": state, "rvol": rvol, "score": score}

    def _detect_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates consecutive directional runs and exhaustion.
        """
        n = len(df)
        streak = 0
        streak_direction = None

        for i in range(n - 1, -1, -1):
            c = df.iloc[i]["close"]
            o = df.iloc[i]["open"]
            direction = "UP" if c >= o else "DOWN"
            if streak_direction is None:
                streak_direction = direction
                streak = 1
            elif direction == streak_direction:
                streak += 1
            else:
                break

        # Check momentum slowdown (shrinking bodies over last 3 candles)
        slowdown = False
        if n >= 4:
            b1 = abs(df.iloc[-1]["close"] - df.iloc[-1]["open"])
            b2 = abs(df.iloc[-2]["close"] - df.iloc[-2]["open"])
            b3 = abs(df.iloc[-3]["close"] - df.iloc[-3]["open"])
            if b3 > b2 > b1:
                slowdown = True

        state = f"{streak} consecutive {'bullish' if streak_direction == 'UP' else 'bearish'} candles"
        if slowdown and streak >= 3:
            state += " (Momentum slowing/exhaustion)"

        return {
            "streak": streak,
            "direction": streak_direction,
            "slowdown": slowdown,
            "state": state
        }

    def _synthesize(
        self,
        patterns: List[PatternSignal],
        trend: Dict[str, Any],
        sr: Dict[str, Any],
        volume: Dict[str, Any],
        momentum: Dict[str, Any],
        current_price: float,
        is_closed: bool
    ) -> Dict[str, Any]:
        """
        Combines independent components into final Bias, Confidence %, and Rough Target.
        """
        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0

        # 1. Candlestick Patterns (Weight 30%)
        pattern_wt = 0.30
        total_weight += pattern_wt
        if patterns:
            p_bull = sum(p.strength for p in patterns if p.bias == "BULLISH")
            p_bear = sum(p.strength for p in patterns if p.bias == "BEARISH")
            p_total = p_bull + p_bear
            if p_total > 0:
                bullish_score += pattern_wt * (p_bull / p_total)
                bearish_score += pattern_wt * (p_bear / p_total)
            else:
                bullish_score += pattern_wt * 0.5
                bearish_score += pattern_wt * 0.5
        else:
            bullish_score += pattern_wt * 0.5
            bearish_score += pattern_wt * 0.5

        # 2. Trend Structure (Weight 30%)
        trend_wt = 0.30
        total_weight += trend_wt
        if trend["trend"] == "UPTREND":
            bullish_score += trend_wt * trend["strength"]
            bearish_score += trend_wt * (1.0 - trend["strength"])
        elif trend["trend"] == "DOWNTREND":
            bearish_score += trend_wt * trend["strength"]
            bullish_score += trend_wt * (1.0 - trend["strength"])
        else:
            bullish_score += trend_wt * 0.5
            bearish_score += trend_wt * 0.5

        # 3. Support / Resistance (Weight 20%)
        sr_wt = 0.20
        total_weight += sr_wt
        supp = sr.get("nearest_support")
        res = sr.get("nearest_resistance")
        if supp and res:
            range_total = res - supp
            if range_total > 0:
                pos_in_range = (current_price - supp) / range_total
                # Near support -> favors bounce (bullish), near resistance -> favors rejection (bearish)
                bullish_score += sr_wt * (1.0 - pos_in_range)
                bearish_score += sr_wt * pos_in_range
            else:
                bullish_score += sr_wt * 0.5
                bearish_score += sr_wt * 0.5
        else:
            bullish_score += sr_wt * 0.5
            bearish_score += sr_wt * 0.5

        # 4. Volume Confirmation (Weight 10%)
        vol_wt = 0.10
        total_weight += vol_wt
        if volume["rvol"] >= 1.3:
            # Volume confirms dominant move
            if bullish_score >= bearish_score:
                bullish_score += vol_wt * 0.8
                bearish_score += vol_wt * 0.2
            else:
                bearish_score += vol_wt * 0.8
                bullish_score += vol_wt * 0.2
        else:
            bullish_score += vol_wt * 0.5
            bearish_score += vol_wt * 0.5

        # 5. Momentum (Weight 10%)
        mom_wt = 0.10
        total_weight += mom_wt
        if momentum["direction"] == "UP":
            if momentum["slowdown"]:
                bearish_score += mom_wt * 0.6
                bullish_score += mom_wt * 0.4
            else:
                bullish_score += mom_wt * 0.8
                bearish_score += mom_wt * 0.2
        elif momentum["direction"] == "DOWN":
            if momentum["slowdown"]:
                bullish_score += mom_wt * 0.6
                bearish_score += mom_wt * 0.4
            else:
                bearish_score += mom_wt * 0.8
                bullish_score += mom_wt * 0.2

        # Final Bias
        score_diff = abs(bullish_score - bearish_score)
        if score_diff < 0.08:
            bias = "NEUTRAL"
            base_confidence = 50 + int(score_diff * 100)
        elif bullish_score > bearish_score:
            bias = "BULLISH"
            base_confidence = int(55 + (bullish_score / total_weight) * 38)
        else:
            bias = "BEARISH"
            base_confidence = int(55 + (bearish_score / total_weight) * 38)

        # Forming candle penalty for confidence
        if not is_closed:
            base_confidence = max(50, base_confidence - 5)

        confidence = max(50, min(95, base_confidence))

        # Rough Price Target Calculation
        target = None
        if bias == "BULLISH":
            if res and res > current_price:
                target = res
            else:
                # Estimate 1.5% target if beyond resistance
                target = current_price * 1.015
        elif bias == "BEARISH":
            if supp and supp < current_price:
                target = supp
            else:
                target = current_price * 0.985
        else:
            target = None

        return {
            "bias": bias,
            "confidence": confidence,
            "target": target
        }
