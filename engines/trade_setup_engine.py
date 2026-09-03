"""
engines/trade_setup_engine.py - Institutional Risk-Reward & Trade Setup Engine
Generates actionable, risk-quantified trade execution plans:
  - Setup Types: SMC Pullback FVG, SMC Order Block, Liquidity Sweep Reversal,
    Value Area Mean Reversion, Chart Pattern Breakout
  - Direction: LONG / SHORT
  - Entry Price (P_entry)
  - Invalidation Stop Loss (P_SL)
  - Take Profit 1 (P_TP1) & Take Profit 2 (P_TP2)
  - Strict Risk-Reward Ratio enforcement: Effective R:R >= 1.80
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from core.models import (
    BiasType,
    ConfluenceReport,
    SetupType,
    TradeSetup,
    FairValueGap,
    OrderBlock,
    ChartPattern,
)


class TradeSetupEngine:
    """
    Evaluates multi-factor confluence and market structure to formulate
    actionable trade plans satisfying strict institutional risk-reward constraints.
    """

    MIN_EFFECTIVE_RR = 1.80

    def __init__(self, min_effective_rr: float = 1.80, atr_multiplier_sl: float = 0.20):
        self.min_effective_rr = min_effective_rr
        self.atr_multiplier_sl = atr_multiplier_sl

    def generate_setup(
        self,
        confluence_report: ConfluenceReport,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> Optional[TradeSetup]:
        """
        Synthesizes active confluence report and OHLCV history into the highest-expectancy
        valid TradeSetup with effective R:R >= 1.80.
        Returns None if no candidate passes the strict institutional filter gate.
        """
        candidates = self.find_all_setups(confluence_report, df, symbol)
        if not candidates:
            return None

        # Priority weighting: SMC Core Continuation (FVG / OB) > Liquidity Sweep > Chart Pattern > Value Area
        setup_priority = {
            SetupType.SMC_PULLBACK_FVG: 5,
            SetupType.SMC_ORDER_BLOCK: 4,
            SetupType.LIQUIDITY_SWEEP_REVERSAL: 3,
            SetupType.CHART_PATTERN_BREAKOUT: 2,
            SetupType.VALUE_AREA_MEAN_REVERSION: 1,
        }

        # Rank candidates by: 1. Setup priority, 2. Effective R:R
        candidates.sort(
            key=lambda s: (
                setup_priority.get(s.setup_type, 0),
                s.effective_rr >= self.min_effective_rr,
                s.effective_rr
            ),
            reverse=True
        )
        best = candidates[0]
        return best if best.effective_rr >= self.min_effective_rr else None

    def find_all_setups(
        self,
        confluence_report: ConfluenceReport,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> List[TradeSetup]:
        """
        Scans all 5 setup archetypes and returns all structurally valid trade setups.
        Enforces top-down macro directional permissioning.
        """
        if len(df) < 14:
            return []

        sym = symbol or confluence_report.symbol
        current_price = confluence_report.current_price
        timestamp = confluence_report.timestamp
        overall_bias = confluence_report.overall_bias
        confidence = confluence_report.confidence_score

        atr14 = self._compute_atr(df, period=14)
        atr_buffer = self.atr_multiplier_sl * atr14

        setups: List[TradeSetup] = []

        # 1. SMC Pullback & FVG Mitigation Setup
        fvg_setup = self._check_smc_fvg_setup(confluence_report, df, sym, current_price, timestamp, atr14, atr_buffer, confidence)
        if fvg_setup:
            setups.append(fvg_setup)

        # 2. SMC Order Block Retest Setup
        ob_setup = self._check_smc_ob_setup(confluence_report, df, sym, current_price, timestamp, atr14, atr_buffer, confidence)
        if ob_setup:
            setups.append(ob_setup)

        # 3. Liquidity Sweep Reversal Setup
        sweep_setup = self._check_liquidity_sweep_setup(confluence_report, df, sym, current_price, timestamp, atr14, atr_buffer, confidence)
        if sweep_setup:
            setups.append(sweep_setup)

        # 4. Value Area Mean Reversion Setup
        va_setup = self._check_value_area_setup(confluence_report, df, sym, current_price, timestamp, atr14, atr_buffer, confidence)
        if va_setup:
            setups.append(va_setup)

        # 5. Chart Pattern Breakout Setup
        pattern_setup = self._check_chart_pattern_setup(confluence_report, df, sym, current_price, timestamp, atr14, atr_buffer, confidence)
        if pattern_setup:
            setups.append(pattern_setup)

        # Filter strictly for positive risk distance, valid R:R >= MIN_EFFECTIVE_RR, and directional alignment
        valid_setups = []
        for s in setups:
            if s.effective_rr < self.min_effective_rr:
                continue
            if overall_bias in [BiasType.BULLISH, BiasType.STRONG_BULLISH] and s.direction != "LONG":
                continue
            if overall_bias in [BiasType.BEARISH, BiasType.STRONG_BEARISH] and s.direction != "SHORT":
                continue
            valid_setups.append(s)

        return valid_setups

    # ------------------------------------------------------------------------
    # 1. SMC FVG Mitigation Setup Archetype
    # ------------------------------------------------------------------------
    def _check_smc_fvg_setup(
        self,
        rep: ConfluenceReport,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
        timestamp: Any,
        atr14: float,
        atr_buffer: float,
        confidence: int,
    ) -> Optional[TradeSetup]:
        if not rep.active_fvgs:
            return None

        # Find closest unmitigated FVG
        for fvg in rep.active_fvgs:
            # Long Setup on Bullish FVG
            if fvg.bias == BiasType.BULLISH and rep.overall_bias in [BiasType.BULLISH, BiasType.STRONG_BULLISH, BiasType.NEUTRAL]:
                if fvg.bottom * 0.995 <= current_price <= fvg.top * 1.015:
                    entry = round(fvg.midpoint, 4)  # Consequent Encroachment 50%
                    sl = round(fvg.bottom - atr_buffer, 4)
                    risk = entry - sl
                    if risk <= 0:
                        continue

                    # Targets: TP1 nearest resistance / POC; TP2 3.0R or HTF target
                    tp1 = round(entry + max(1.5 * risk, 1.2 * atr14), 4)
                    tp2 = round(entry + max(2.8 * risk, 2.5 * atr14), 4)
                    if rep.volume_profile and rep.volume_profile.vah_price > entry:
                        tp2 = max(tp2, round(rep.volume_profile.vah_price, 4))

                    return self._create_trade_setup(
                        symbol=symbol,
                        timestamp=timestamp,
                        setup_type=SetupType.SMC_PULLBACK_FVG,
                        direction="LONG",
                        entry=entry,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        confidence=confidence,
                        rationale=[
                            f"Price retraced into Bullish FVG [${fvg.bottom:,.2f} - ${fvg.top:,.2f}]",
                            f"Entry anchored at Consequent Encroachment (50%) midline ${fvg.midpoint:,.2f}",
                            f"Invalidation placed below FVG boundary with 0.2*ATR buffer (${sl:,.2f})"
                        ],
                        invalidation_reason=f"Closed below Bullish FVG lower threshold at ${fvg.bottom:,.2f}"
                    )

            # Short Setup on Bearish FVG
            elif fvg.bias == BiasType.BEARISH and rep.overall_bias in [BiasType.BEARISH, BiasType.STRONG_BEARISH, BiasType.NEUTRAL]:
                if fvg.bottom * 0.985 <= current_price <= fvg.top * 1.005:
                    entry = round(fvg.midpoint, 4)
                    sl = round(fvg.top + atr_buffer, 4)
                    risk = sl - entry
                    if risk <= 0:
                        continue

                    tp1 = round(entry - max(1.5 * risk, 1.2 * atr14), 4)
                    tp2 = round(entry - max(2.8 * risk, 2.5 * atr14), 4)
                    if rep.volume_profile and rep.volume_profile.val_price < entry:
                        tp2 = min(tp2, round(rep.volume_profile.val_price, 4))

                    return self._create_trade_setup(
                        symbol=symbol,
                        timestamp=timestamp,
                        setup_type=SetupType.SMC_PULLBACK_FVG,
                        direction="SHORT",
                        entry=entry,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        confidence=confidence,
                        rationale=[
                            f"Price retraced into Bearish FVG [${fvg.bottom:,.2f} - ${fvg.top:,.2f}]",
                            f"Entry anchored at Consequent Encroachment (50%) midline ${fvg.midpoint:,.2f}",
                            f"Invalidation placed above FVG boundary with 0.2*ATR buffer (${sl:,.2f})"
                        ],
                        invalidation_reason=f"Closed above Bearish FVG upper threshold at ${fvg.top:,.2f}"
                    )
        return None

    # ------------------------------------------------------------------------
    # 2. SMC Order Block Retest Setup Archetype
    # ------------------------------------------------------------------------
    def _check_smc_ob_setup(
        self,
        rep: ConfluenceReport,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
        timestamp: Any,
        atr14: float,
        atr_buffer: float,
        confidence: int,
    ) -> Optional[TradeSetup]:
        if not rep.active_obs:
            return None

        for ob in rep.active_obs:
            if ob.bias == BiasType.BULLISH and rep.overall_bias in [BiasType.BULLISH, BiasType.STRONG_BULLISH, BiasType.NEUTRAL]:
                if ob.bottom * 0.995 <= current_price <= ob.top * 1.015:
                    entry = round(ob.top, 4)  # Entry at top boundary of demand block
                    sl = round(ob.bottom - atr_buffer, 4)
                    risk = entry - sl
                    if risk <= 0:
                        continue

                    tp1 = round(entry + max(1.5 * risk, 1.2 * atr14), 4)
                    tp2 = round(entry + max(2.8 * risk, 2.5 * atr14), 4)

                    return self._create_trade_setup(
                        symbol=symbol,
                        timestamp=timestamp,
                        setup_type=SetupType.SMC_ORDER_BLOCK,
                        direction="LONG",
                        entry=entry,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        confidence=confidence,
                        rationale=[
                            f"Institutional +OB Demand zone retest [${ob.bottom:,.2f} - ${ob.top:,.2f}]",
                            f"Displacement confirmation with volume ${ob.volume:,.1f}",
                            f"Stop Loss secured below OB invalidation level (${sl:,.2f})"
                        ],
                        invalidation_reason=f"Closed past Bullish Order Block low at ${ob.bottom:,.2f}"
                    )

            elif ob.bias == BiasType.BEARISH and rep.overall_bias in [BiasType.BEARISH, BiasType.STRONG_BEARISH, BiasType.NEUTRAL]:
                if ob.bottom * 0.985 <= current_price <= ob.top * 1.005:
                    entry = round(ob.bottom, 4)  # Entry at bottom boundary of supply block
                    sl = round(ob.top + atr_buffer, 4)
                    risk = sl - entry
                    if risk <= 0:
                        continue

                    tp1 = round(entry - max(1.5 * risk, 1.2 * atr14), 4)
                    tp2 = round(entry - max(2.8 * risk, 2.5 * atr14), 4)

                    return self._create_trade_setup(
                        symbol=symbol,
                        timestamp=timestamp,
                        setup_type=SetupType.SMC_ORDER_BLOCK,
                        direction="SHORT",
                        entry=entry,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        confidence=confidence,
                        rationale=[
                            f"Institutional -OB Supply zone retest [${ob.bottom:,.2f} - ${ob.top:,.2f}]",
                            f"Displacement confirmation with volume ${ob.volume:,.1f}",
                            f"Stop Loss secured above OB invalidation level (${sl:,.2f})"
                        ],
                        invalidation_reason=f"Closed past Bearish Order Block high at ${ob.top:,.2f}"
                    )
        return None

    # ------------------------------------------------------------------------
    # 3. Liquidity Sweep & CHoCH Reversal Archetype
    # ------------------------------------------------------------------------
    def _check_liquidity_sweep_setup(
        self,
        rep: ConfluenceReport,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
        timestamp: Any,
        atr14: float,
        atr_buffer: float,
        confidence: int,
    ) -> Optional[TradeSetup]:
        lows = df["low"].values
        highs = df["high"].values
        closes = df["close"].values
        opens = df["open"].values

        # Check for recent lower wick sweep (SSL) or upper wick sweep (BSL)
        if len(df) < 5:
            return None

        # Look back 5 candles for significant wick rejection
        for i in range(-1, -min(6, len(df)), -1):
            c = closes[i]
            o = opens[i]
            h = highs[i]
            l = lows[i]
            rng = max(1e-8, h - l)
            lw = (min(o, c) - l) / rng
            uw = (h - max(o, c)) / rng

            # Bullish SSL Sweep (Turtle Soup long reversal)
            if lw >= 0.40 and l == min(lows[-15:]):
                if rep.overall_bias in [BiasType.BULLISH, BiasType.STRONG_BULLISH, BiasType.NEUTRAL]:
                    entry = round(current_price, 4)
                    sl = round(l - atr_buffer, 4)
                    risk = entry - sl
                    if risk <= 0:
                        continue

                    tp1 = round(entry + max(1.5 * risk, 1.2 * atr14), 4)
                    tp2 = round(entry + max(2.8 * risk, 2.5 * atr14), 4)

                    return self._create_trade_setup(
                        symbol=symbol,
                        timestamp=timestamp,
                        setup_type=SetupType.LIQUIDITY_SWEEP_REVERSAL,
                        direction="LONG",
                        entry=entry,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        confidence=confidence,
                        rationale=[
                            f"Sell-Side Liquidity (SSL) sweep at swing low ${l:,.2f}",
                            f"Significant wick rejection of {lw*100:.1f}% indicates stop-run absorption",
                            f"Stop Loss anchored below the sweep extreme at ${sl:,.2f}"
                        ],
                        invalidation_reason=f"Breach and acceptance below sweep low ${l:,.2f}"
                    )

            # Bearish BSL Sweep (Turtle Soup short reversal)
            elif uw >= 0.40 and h == max(highs[-15:]):
                if rep.overall_bias in [BiasType.BEARISH, BiasType.STRONG_BEARISH, BiasType.NEUTRAL]:
                    entry = round(current_price, 4)
                    sl = round(h + atr_buffer, 4)
                    risk = sl - entry
                    if risk <= 0:
                        continue

                    tp1 = round(entry - max(1.5 * risk, 1.2 * atr14), 4)
                    tp2 = round(entry - max(2.8 * risk, 2.5 * atr14), 4)

                    return self._create_trade_setup(
                        symbol=symbol,
                        timestamp=timestamp,
                        setup_type=SetupType.LIQUIDITY_SWEEP_REVERSAL,
                        direction="SHORT",
                        entry=entry,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        confidence=confidence,
                        rationale=[
                            f"Buy-Side Liquidity (BSL) sweep at swing high ${h:,.2f}",
                            f"Significant wick rejection of {uw*100:.1f}% indicates institutional distribution",
                            f"Stop Loss anchored above the sweep extreme at ${sl:,.2f}"
                        ],
                        invalidation_reason=f"Breach and acceptance above sweep high ${h:,.2f}"
                    )
        return None

    # ------------------------------------------------------------------------
    # 4. Value Area Mean Reversion Archetype
    # ------------------------------------------------------------------------
    def _check_value_area_setup(
        self,
        rep: ConfluenceReport,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
        timestamp: Any,
        atr14: float,
        atr_buffer: float,
        confidence: int,
    ) -> Optional[TradeSetup]:
        vp = rep.volume_profile
        if not vp:
            return None

        # Long: Bounce from Value Area Low targeting POC & VAH
        if current_price <= vp.val_price * 1.008 and current_price >= vp.val_price * 0.985:
            entry = round(current_price, 4)
            sl = round(vp.val_price - atr_buffer, 4)
            risk = entry - sl
            if risk <= 0:
                return None

            tp1 = round(vp.poc_price, 4)
            tp2 = round(vp.vah_price, 4)

            # Ensure minimum spacing
            if tp1 <= entry:
                tp1 = round(entry + 1.5 * risk, 4)
            if tp2 <= tp1:
                tp2 = round(entry + 2.8 * risk, 4)

            return self._create_trade_setup(
                symbol=symbol,
                timestamp=timestamp,
                setup_type=SetupType.VALUE_AREA_MEAN_REVERSION,
                direction="LONG",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                confidence=confidence,
                rationale=[
                    f"Auction Market Theory: Bounce off Value Area Low (${vp.val_price:,.2f})",
                    f"Targeting high-volume mean reversion to POC (${vp.poc_price:,.2f}) and VAH (${vp.vah_price:,.2f})",
                    f"Stop Loss placed beyond Value Area Low (${sl:,.2f})"
                ],
                invalidation_reason=f"Auction acceptance outside Value Area below ${vp.val_price:,.2f}"
            )

        # Short: Rejection from Value Area High targeting POC & VAL
        elif current_price >= vp.vah_price * 0.992 and current_price <= vp.vah_price * 1.015:
            entry = round(current_price, 4)
            sl = round(vp.vah_price + atr_buffer, 4)
            risk = sl - entry
            if risk <= 0:
                return None

            tp1 = round(vp.poc_price, 4)
            tp2 = round(vp.val_price, 4)

            if tp1 >= entry:
                tp1 = round(entry - 1.5 * risk, 4)
            if tp2 >= tp1:
                tp2 = round(entry - 2.8 * risk, 4)

            return self._create_trade_setup(
                symbol=symbol,
                timestamp=timestamp,
                setup_type=SetupType.VALUE_AREA_MEAN_REVERSION,
                direction="SHORT",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                confidence=confidence,
                rationale=[
                    f"Auction Market Theory: Rejection off Value Area High (${vp.vah_price:,.2f})",
                    f"Targeting mean reversion to POC (${vp.poc_price:,.2f}) and VAL (${vp.val_price:,.2f})",
                    f"Stop Loss placed beyond Value Area High (${sl:,.2f})"
                ],
                invalidation_reason=f"Auction acceptance outside Value Area above ${vp.vah_price:,.2f}"
            )
        return None

    # ------------------------------------------------------------------------
    # 5. Chart Pattern Breakout Archetype
    # ------------------------------------------------------------------------
    def _check_chart_pattern_setup(
        self,
        rep: ConfluenceReport,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
        timestamp: Any,
        atr14: float,
        atr_buffer: float,
        confidence: int,
    ) -> Optional[TradeSetup]:
        if not rep.active_patterns:
            return None

        # Prefer highest quality detected pattern
        best_pat = max(rep.active_patterns, key=lambda p: p.quality_score)
        if best_pat.quality_score < 0.70:
            return None

        if best_pat.bias in [BiasType.BULLISH, BiasType.STRONG_BULLISH]:
            entry = round(best_pat.neckline_price or current_price, 4)
            sl = round(best_pat.invalidation_level - atr_buffer, 4)
            risk = entry - sl
            if risk <= 0:
                return None

            tp1 = round(entry + max(1.6 * risk, 1.2 * atr14), 4)
            min_tp2 = entry + max(2.8 * risk, 2.5 * atr14)
            target = max(best_pat.projected_target or 0.0, min_tp2)
            tp2 = round(target, 4)

            return self._create_trade_setup(
                symbol=symbol,
                timestamp=timestamp,
                setup_type=SetupType.CHART_PATTERN_BREAKOUT,
                direction="LONG",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                confidence=confidence,
                rationale=[
                    f"Classical Chart Pattern: {best_pat.name} (Quality: {best_pat.quality_score*100:.0f}%)",
                    f"Breakout above neckline at ${entry:,.2f}",
                    f"Invalidation level anchored at ${sl:,.2f}"
                ],
                invalidation_reason=f"Breakdown below {best_pat.name} base invalidation at ${best_pat.invalidation_level:,.2f}"
            )

        elif best_pat.bias in [BiasType.BEARISH, BiasType.STRONG_BEARISH]:
            entry = round(best_pat.neckline_price or current_price, 4)
            sl = round(best_pat.invalidation_level + atr_buffer, 4)
            risk = sl - entry
            if risk <= 0:
                return None

            tp1 = round(entry - max(1.6 * risk, 1.2 * atr14), 4)
            min_tp2 = entry - max(2.8 * risk, 2.5 * atr14)
            target = min(best_pat.projected_target or 1e12, min_tp2)
            tp2 = round(target, 4)

            return self._create_trade_setup(
                symbol=symbol,
                timestamp=timestamp,
                setup_type=SetupType.CHART_PATTERN_BREAKOUT,
                direction="SHORT",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                confidence=confidence,
                rationale=[
                    f"Classical Chart Pattern: {best_pat.name} (Quality: {best_pat.quality_score*100:.0f}%)",
                    f"Breakdown below neckline at ${entry:,.2f}",
                    f"Invalidation level anchored at ${sl:,.2f}"
                ],
                invalidation_reason=f"Breakout above {best_pat.name} top invalidation at ${best_pat.invalidation_level:,.2f}"
            )
        return None

    # ------------------------------------------------------------------------
    # Helper Construction & Vector Math
    # ------------------------------------------------------------------------
    def _create_trade_setup(
        self,
        symbol: str,
        timestamp: Any,
        setup_type: SetupType,
        direction: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        confidence: int,
        rationale: List[str],
        invalidation_reason: str,
    ) -> TradeSetup:
        risk = abs(entry - sl)
        if risk <= 1e-8:
            risk = 1e-8

        reward_tp1 = abs(tp1 - entry)
        reward_tp2 = abs(tp2 - entry)

        rr_tp1 = round(reward_tp1 / risk, 2)
        rr_tp2 = round(reward_tp2 / risk, 2)
        effective_rr = round(0.5 * rr_tp1 + 0.5 * rr_tp2, 2)

        setup_id = f"SETUP-{uuid.uuid4().hex[:8].upper()}"

        return TradeSetup(
            setup_id=setup_id,
            symbol=symbol,
            timestamp=timestamp,
            setup_type=setup_type,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            tp1_price=tp1,
            tp2_price=tp2,
            risk_reward_tp1=rr_tp1,
            risk_reward_tp2=rr_tp2,
            effective_rr=effective_rr,
            confidence_score=confidence,
            rationale=rationale,
            invalidation_reason=invalidation_reason,
        )

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates latest Average True Range."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(df)
        if n == 0:
            return 1.0
        if n == 1:
            return max(1e-6, float(high[0] - low[0]))

        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        atr = float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))
        return max(1e-6, atr)
