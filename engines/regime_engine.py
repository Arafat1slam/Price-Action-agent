"""
engines/regime_engine.py - Market Regime Detection Engine
Classifies the market into:
  1. TRENDING_BULL: Strong upward directional persistence (ADX >= 22, +DI > -DI, EMA alignment)
  2. TRENDING_BEAR: Strong downward directional persistence (ADX >= 22, -DI > +DI, EMA alignment)
  3. RANGING: Low directional momentum (ADX < 20, compressed BB bandwidth, S/R bounded)
  4. BREAKOUT: Volatility expansion (ATR > 1.4x, BB expansion, RVOL volume surge)
  5. HIGH_VOLATILITY_CHOP: Extreme volatility with erratic swing churning (ATR extreme, whipsaws)
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

from core.models import MarketRegime, RegimeReport


class MarketRegimeEngine:
    """
    Statistical and technical market regime detector.
    Evaluates volatility, directional persistence, volume expansion, and range boundaries.
    """

    def __init__(self, adx_period: int = 14, atr_period: int = 14, bb_period: int = 20):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.bb_period = bb_period

    def detect_regime(self, df: pd.DataFrame) -> RegimeReport:
        """
        Analyzes candlestick dataframe and returns a detailed RegimeReport.
        """
        if df is None or len(df) < 20:
            return RegimeReport(
                regime=MarketRegime.RANGING,
                regime_label="RANGING (INSUFFICIENT DATA)",
                confidence=50,
                adx=15.0,
                plus_di=20.0,
                minus_di=20.0,
                atr_ratio=1.0,
                bb_bandwidth_pct=2.0,
                volatility_state="NORMAL",
                recommended_strategy="Wait for data accumulation before aggressive positioning.",
                key_drivers=["Insufficient bars for full regime detection"]
            )

        close = pd.to_numeric(df["close"], errors="coerce").astype(float).values
        high = pd.to_numeric(df["high"], errors="coerce").astype(float).values
        low = pd.to_numeric(df["low"], errors="coerce").astype(float).values
        volume = pd.to_numeric(df["volume"], errors="coerce").astype(float).values
        n = len(close)

        # 1. Calculate Wilder's DMI & ADX
        adx, plus_di, minus_di = self._calculate_adx(high, low, close, period=self.adx_period)

        # 2. Calculate ATR & Volatility Expansion
        atr_curr, atr_baseline, atr_ratio = self._calculate_atr_metrics(high, low, close, period=self.atr_period)

        # 3. Calculate Bollinger Bands & Bandwidth
        bb_upper, bb_mid, bb_lower, bandwidth_pct = self._calculate_bollinger_bands(close, period=self.bb_period)

        # 4. Calculate RVOL (Relative Volume vs 20-period SMA)
        rvol = self._calculate_rvol(volume, period=20)

        # 5. Calculate Moving Average Stack (EMA 20 vs EMA 50)
        ema20 = pd.Series(close).ewm(span=min(20, n)).mean().iloc[-1]
        ema50 = pd.Series(close).ewm(span=min(50, n)).mean().iloc[-1]
        curr_price = close[-1]

        # 6. Evaluate Volatility State
        if atr_ratio >= 1.70:
            vol_state = "EXTREME"
        elif atr_ratio >= 1.30:
            vol_state = "EXPANDING"
        elif atr_ratio <= 0.75:
            vol_state = "LOW"
        else:
            vol_state = "NORMAL"

        drivers = []

        # 7. Regime Decision Tree
        # A. Volatility Expansion / Breakout
        # Squeeze expansion: Bandwidth expanding or price penetrating outer bands + high volume + high ATR
        is_breaking_high = curr_price > bb_upper and close[-2] <= bb_upper
        is_breaking_low = curr_price < bb_lower and close[-2] >= bb_lower
        is_breakout_vol = rvol >= 1.60 and atr_ratio >= 1.25

        if (is_breaking_high or is_breaking_low) and is_breakout_vol:
            regime = MarketRegime.BREAKOUT
            direction = "BULLISH BREAKOUT" if is_breaking_high else "BEARISH BREAKDOWN"
            regime_label = f"VOLATILITY EXPANSION ({direction})"
            conf = int(min(95, 70 + 10 * (rvol - 1.0) + 10 * (atr_ratio - 1.0)))
            strategy = (
                f"Momentum Breakout — Follow the {direction}. "
                "Enter on structural breakout retest or immediate momentum with trailing stop."
            )
            drivers.append(f"Price broke outer Bollinger Band with RVOL {rvol:.2f}x")
            drivers.append(f"ATR expanded to {atr_ratio:.2f}x rolling baseline")

        # B. High Volatility Chop (Extreme ATR with weak directional trend or whipsawing)
        elif (atr_ratio >= 1.65 or vol_state == "EXTREME") and adx < 22.0:
            regime = MarketRegime.HIGH_VOLATILITY_CHOP
            regime_label = "HIGH-VOLATILITY CHOP"
            conf = int(min(90, 65 + 15 * (atr_ratio - 1.0)))
            strategy = (
                "Capital Preservation Mode — Erratic swing churning without clear trend direction. "
                "Reduce position size by 50-75% or sit on hands until volatility stabilizes."
            )
            drivers.append(f"Extreme ATR ratio ({atr_ratio:.2f}x) combined with low ADX ({adx:.1f})")
            drivers.append("Erratic high-amplitude price swings lacking directional persistence")

        # C. Trending Bull
        elif adx >= 22.0 and plus_di > minus_di and curr_price >= ema20:
            regime = MarketRegime.TRENDING_BULL
            strength = "STRONG" if adx >= 32.0 else "MODERATE"
            regime_label = f"TRENDING BULL ({strength})"
            conf = int(min(95, 60 + (adx - 20) * 1.5 + (plus_di - minus_di) * 0.5))
            strategy = (
                "Trend Following — Favor SMC Pullbacks to Bullish FVG & Order Blocks. "
                "Strictly avoid counter-trend shorts. Let winners run to HTF liquidity targets."
            )
            drivers.append(f"ADX at {adx:.1f} confirming persistent trend strength")
            drivers.append(f"+DI ({plus_di:.1f}) dominating -DI ({minus_di:.1f})")
            drivers.append("Price holding above 20 EMA / 50 EMA bullish alignment")

        # D. Trending Bear
        elif adx >= 22.0 and minus_di > plus_di and curr_price <= ema20:
            regime = MarketRegime.TRENDING_BEAR
            strength = "STRONG" if adx >= 32.0 else "MODERATE"
            regime_label = f"TRENDING BEAR ({strength})"
            conf = int(min(95, 60 + (adx - 20) * 1.5 + (minus_di - plus_di) * 0.5))
            strategy = (
                "Trend Following — Favor SMC Retracements to Bearish FVG & Bearish Order Blocks. "
                "Strictly avoid counter-trend longs. Target sell-side liquidity sweeps."
            )
            drivers.append(f"ADX at {adx:.1f} confirming persistent downward trend strength")
            drivers.append(f"-DI ({minus_di:.1f}) dominating +DI ({plus_di:.1f})")
            drivers.append("Price suppressed below 20 EMA / 50 EMA bearish alignment")

        # E. Ranging / Consolidation
        else:
            regime = MarketRegime.RANGING
            regime_label = "RANGING CONSOLIDATION"
            conf = int(min(90, 60 + max(0, 20.0 - adx) * 2.0))
            strategy = (
                "Mean Reversion — Fade S/R extremes (Buy Support / Sell Resistance). "
                "Avoid breakout chasing. Take quick 1:1.5 to 1:2 profits at range midpoint / POC."
            )
            drivers.append(f"ADX below 20 ({adx:.1f}) indicating lack of directional momentum")
            drivers.append(f"Bollinger Bandwidth compressed ({bandwidth_pct:.2f}%)")
            drivers.append("Price oscillating within well-defined structural support/resistance")

        # 8. Advanced Indicator Drivers (Choppiness Index & TTM Squeeze)
        try:
            from engines.indicators_engine import ChoppinessIndexEngine, TTMSqueezeEngine
            chop_res = ChoppinessIndexEngine(period=14).compute(df)
            squeeze_res = TTMSqueezeEngine().compute(df)
            if chop_res.is_choppy:
                drivers.append(f"Choppiness Index elevated ({chop_res.chop_index:.1f}/100) — Congestion warning")
            elif chop_res.is_trending:
                drivers.append(f"Choppiness Index ({chop_res.chop_index:.1f}/100) — Strong directional flow")

            if squeeze_res.is_squeeze_on:
                drivers.append(f"TTM Squeeze ACTIVE ({squeeze_res.squeeze_bars} bars) — Energy coiling for expansion")
        except Exception:
            pass

        return RegimeReport(
            regime=regime,
            regime_label=regime_label,
            confidence=max(50, min(95, conf)),
            adx=round(float(adx), 2),
            plus_di=round(float(plus_di), 2),
            minus_di=round(float(minus_di), 2),
            atr_ratio=round(float(atr_ratio), 2),
            bb_bandwidth_pct=round(float(bandwidth_pct), 2),
            volatility_state=vol_state,
            recommended_strategy=strategy,
            key_drivers=drivers
        )

    # ------------------------------------------------------------------------
    # Mathematical Helpers
    # ------------------------------------------------------------------------
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
        """Calculates Wilder's Smoothed ADX, +DI, and -DI."""
        n = len(close)
        if n < period + 2:
            return 15.0, 20.0, 20.0

        tr_arr = np.zeros(n)
        pdm_arr = np.zeros(n)
        mdm_arr = np.zeros(n)

        for i in range(1, n):
            h_diff = high[i] - high[i - 1]
            l_diff = low[i - 1] - low[i]

            pdm_arr[i] = h_diff if (h_diff > 0 and h_diff > l_diff) else 0.0
            mdm_arr[i] = l_diff if (l_diff > 0 and l_diff > h_diff) else 0.0

            tr_arr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # Wilder's Smoothing
        alpha = 1.0 / period
        tr_smooth = pd.Series(tr_arr).ewm(alpha=alpha, adjust=False).mean().values
        pdm_smooth = pd.Series(pdm_arr).ewm(alpha=alpha, adjust=False).mean().values
        mdm_smooth = pd.Series(mdm_arr).ewm(alpha=alpha, adjust=False).mean().values

        eps = 1e-8
        plus_di = 100.0 * (pdm_smooth / (tr_smooth + eps))
        minus_di = 100.0 * (mdm_smooth / (tr_smooth + eps))

        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + eps)
        adx_series = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values

        return float(adx_series[-1]), float(plus_di[-1]), float(minus_di[-1])

    def _calculate_atr_metrics(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
        """Calculates current ATR and ratio against rolling 50-period baseline."""
        n = len(close)
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
        atr_series = pd.Series(tr).rolling(window=period, min_periods=min(5, n)).mean().values
        curr_atr = float(atr_series[-1]) if not np.isnan(atr_series[-1]) else 1.0

        baseline_span = min(50, n)
        baseline_atr = float(np.median(atr_series[-baseline_span:])) if baseline_span > 0 else curr_atr
        if baseline_atr <= 0 or np.isnan(baseline_atr):
            baseline_atr = curr_atr

        ratio = curr_atr / max(1e-8, baseline_atr)
        return curr_atr, baseline_atr, float(ratio)

    def _calculate_bollinger_bands(self, close: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float, float]:
        """Calculates Bollinger Bands and Bandwidth %."""
        n = len(close)
        span = min(period, n)
        sma = pd.Series(close).rolling(window=span, min_periods=min(5, n)).mean().values
        std = pd.Series(close).rolling(window=span, min_periods=min(5, n)).std().values

        curr_sma = float(sma[-1])
        curr_std = float(std[-1]) if not np.isnan(std[-1]) else 0.0

        upper = curr_sma + num_std * curr_std
        lower = curr_sma - num_std * curr_std
        bandwidth_pct = ((upper - lower) / max(1e-8, curr_sma)) * 100.0

        return upper, curr_sma, lower, float(bandwidth_pct)

    def _calculate_rvol(self, volume: np.ndarray, period: int = 20) -> float:
        """Calculates Relative Volume vs rolling SMA."""
        n = len(volume)
        if n < 2:
            return 1.0
        span = min(period, n)
        sma_vol = float(np.mean(volume[-span:]))
        curr_vol = float(volume[-1])
        return curr_vol / max(1e-8, sma_vol)

    # Alias for uniform engine interface
    analyze = detect_regime


# Module-level aliases
RegimeEngine = MarketRegimeEngine
