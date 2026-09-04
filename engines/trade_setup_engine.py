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
    ExecutionState,
    TradeStyle,
    TradeQualityScore,
    QualityGrade,
    PositionState,
    ActivePosition,
    RegimeReport,
    MarketRegime,
    MLInferenceResult,
)
from core.security import verify_author_integrity


class TradeSetupEngine:
    """
    Evaluates multi-factor confluence and market structure to formulate
    actionable trade plans satisfying strict institutional risk-reward constraints.
    """

    MIN_EFFECTIVE_RR = 1.80

    def __init__(self, min_effective_rr: float = 1.80, atr_multiplier_sl: float = 0.20):
        verify_author_integrity()
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
                        invalidation_reason=f"Closed below Bullish FVG lower threshold at ${fvg.bottom:,.2f}",
                        current_price=current_price,
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
                        invalidation_reason=f"Closed above Bearish FVG upper threshold at ${fvg.top:,.2f}",
                        current_price=current_price,
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
                        invalidation_reason=f"Closed past Bullish Order Block low at ${ob.bottom:,.2f}",
                        current_price=current_price,
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
                        invalidation_reason=f"Closed past Bearish Order Block high at ${ob.top:,.2f}",
                        current_price=current_price,
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
                        invalidation_reason=f"Breach and acceptance below sweep low ${l:,.2f}",
                        current_price=current_price,
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
                        invalidation_reason=f"Breach and acceptance above sweep high ${h:,.2f}",
                        current_price=current_price,
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
                invalidation_reason=f"Auction acceptance outside Value Area below ${vp.val_price:,.2f}",
                current_price=current_price,
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
                invalidation_reason=f"Auction acceptance outside Value Area above ${vp.vah_price:,.2f}",
                current_price=current_price,
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
                invalidation_reason=f"Breakdown below {best_pat.name} base invalidation at ${best_pat.invalidation_level:,.2f}",
                current_price=current_price,
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
                invalidation_reason=f"Breakout above {best_pat.name} top invalidation at ${best_pat.invalidation_level:,.2f}",
                current_price=current_price,
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
        current_price: Optional[float] = None,
        candle: Optional[Dict[str, float]] = None,
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

        # Probability of TP1 & TP2 (higher R:R requires higher barrier to hit)
        base_prob = confidence * 0.92
        rr_penalty = max(0.0, (rr_tp1 - 1.0) * 6.0)
        tp1_probability = int(np.clip(base_prob - rr_penalty, 40, 92))
        spread_penalty = max(0.0, (rr_tp2 - rr_tp1) * 8.0)
        tp2_probability = int(np.clip(tp1_probability - spread_penalty, 25, 80))

        # Recommended risk % based on institutional confluence confidence
        if confidence >= 80:
            recommended_risk_pct = 2.0
        elif confidence >= 70:
            recommended_risk_pct = 1.5
        elif confidence >= 60:
            recommended_risk_pct = 1.0
        else:
            recommended_risk_pct = 0.5

        # Suggested position size based on standard $10,000 reference portfolio
        account_size = 10000.0
        dollar_risk = account_size * (recommended_risk_pct / 100.0)
        risk_pct_trade = risk / entry if entry > 0 else 0.01
        position_size_usd = round(min(50000.0, max(100.0, dollar_risk / risk_pct_trade)), 2)

        setup = TradeSetup(
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
            tp1_probability=tp1_probability,
            tp2_probability=tp2_probability,
            recommended_risk_pct=recommended_risk_pct,
            position_size_usd=position_size_usd,
        )

        eff_price = current_price if current_price is not None else entry
        return self.update_execution_state(setup, eff_price, candle)

    def update_execution_state(
        self,
        setup: TradeSetup,
        current_price: float,
        candle: Optional[Dict[str, float]] = None,
    ) -> TradeSetup:
        """
        Dynamically updates the entry confirmation state machine based on latest live price tick:
        - Checks distance from current price to entry target.
        - Emits WAITING_FOR_PRICE (Do Not Chase) when price has not retraced to entry.
        - Emits CONFIRMED_ENTRY_TRIGGER when price reaches entry zone with candle reaction.
        - Emits INVALIDATED when price breaches Stop Loss.
        - Emits TARGET_HIT when price reaches TP1 or TP2.
        """
        entry = setup.entry_price
        sl = setup.stop_loss
        tp1 = setup.tp1_price
        tp2 = setup.tp2_price
        is_long = setup.direction == "LONG"

        dist_pct = ((current_price - entry) / entry) * 100 if entry > 0 else 0.0
        setup.entry_distance_pct = round(dist_pct, 2)

        # Dynamic entry tolerance: 0.15% of entry price (e.g., $120 on BTC, $0.20 on SOL)
        tolerance = 0.0015 * entry

        # Candle details if available
        candle_detail = ""
        if candle:
            o = float(candle.get("open", current_price))
            h = float(candle.get("high", current_price))
            l = float(candle.get("low", current_price))
            c = float(candle.get("close", current_price))
            v = float(candle.get("volume", 0.0))
            rng = max(1e-8, h - l)
            if is_long:
                lower_wick = (min(o, c) - l) / rng
                is_green = c >= o
                if lower_wick >= 0.20 or is_green:
                    candle_detail = " (Bullish wick/bounce reaction confirmed)"
            else:
                upper_wick = (h - max(o, c)) / rng
                is_red = c <= o
                if upper_wick >= 0.20 or is_red:
                    candle_detail = " (Bearish wick/rejection reaction confirmed)"

        if is_long:
            if current_price >= tp2:
                setup.execution_state = ExecutionState.TARGET_HIT.value
                setup.entry_action = f"🎯 TARGET 2 REACHED at ${current_price:,.2f}! (+{abs((tp2-entry)/entry)*100:.1f}%) Close 100% position."
            elif current_price >= tp1:
                setup.execution_state = ExecutionState.TARGET_HIT.value
                setup.entry_action = f"🎯 TARGET 1 HIT at ${current_price:,.2f}! Secure 50% profit. Move SL to Breakeven (${entry:,.2f})."
            elif current_price <= sl:
                setup.execution_state = ExecutionState.INVALIDATED.value
                setup.entry_action = f"❌ SETUP INVALIDATED — Price breached Stop Loss (${sl:,.2f}). Capital protected."
            else:
                # Between SL and TP1
                if current_price > entry + tolerance:
                    diff_val = current_price - entry
                    setup.execution_state = ExecutionState.WAITING_FOR_PRICE.value
                    setup.entry_action = (
                        f"⏳ DO NOT CHASE — Wait for pullback to entry ${entry:,.2f} "
                        f"(${current_price:,.2f} is +{abs(dist_pct):.2f}% higher / ${diff_val:,.2f} away)"
                    )
                elif current_price < entry - tolerance:
                    setup.execution_state = ExecutionState.IN_ENTRY_ZONE.value
                    setup.entry_action = (
                        f"⚠️ IN DISCOUNT ZONE (${current_price:,.2f}) — Watch for bullish rejection "
                        f"to confirm entry around ${entry:,.2f} (SL: ${sl:,.2f})"
                    )
                else:
                    setup.execution_state = ExecutionState.CONFIRMED_ENTRY_TRIGGER.value
                    setup.entry_action = (
                        f"🚨 CONFIRMED ENTRY TRIGGER — Price in entry zone (${current_price:,.2f})! "
                        f"EXECUTE LONG NOW!{candle_detail} (SL: ${sl:,.2f} | TP1: ${tp1:,.2f})"
                    )
        else: # SHORT
            if current_price <= tp2:
                setup.execution_state = ExecutionState.TARGET_HIT.value
                setup.entry_action = f"🎯 TARGET 2 REACHED at ${current_price:,.2f}! (+{abs((entry-tp2)/entry)*100:.1f}%) Close 100% position."
            elif current_price <= tp1:
                setup.execution_state = ExecutionState.TARGET_HIT.value
                setup.entry_action = f"🎯 TARGET 1 HIT at ${current_price:,.2f}! Secure 50% profit. Move SL to Breakeven (${entry:,.2f})."
            elif current_price >= sl:
                setup.execution_state = ExecutionState.INVALIDATED.value
                setup.entry_action = f"❌ SETUP INVALIDATED — Price surged above Stop Loss (${sl:,.2f}). Capital protected."
            else:
                # Between SL and TP1
                if current_price < entry - tolerance:
                    diff_val = entry - current_price
                    setup.execution_state = ExecutionState.WAITING_FOR_PRICE.value
                    setup.entry_action = (
                        f"⏳ DO NOT CHASE — Wait for rally to entry ${entry:,.2f} "
                        f"(${current_price:,.2f} is -{abs(dist_pct):.2f}% lower / ${diff_val:,.2f} away)"
                    )
                elif current_price > entry + tolerance:
                    setup.execution_state = ExecutionState.IN_ENTRY_ZONE.value
                    setup.entry_action = (
                        f"⚠️ IN PREMIUM ZONE (${current_price:,.2f}) — Watch for bearish rejection "
                        f"to confirm entry around ${entry:,.2f} (SL: ${sl:,.2f})"
                    )
                else:
                    setup.execution_state = ExecutionState.CONFIRMED_ENTRY_TRIGGER.value
                    setup.entry_action = (
                        f"🚨 CONFIRMED ENTRY TRIGGER — Price in entry zone (${current_price:,.2f})! "
                        f"EXECUTE SHORT NOW!{candle_detail} (SL: ${sl:,.2f} | TP1: ${tp1:,.2f})"
                    )

        return setup

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

    # ========================================================================
    # Phase 5: Unified Trade Quality Scoring (0–100)
    # ========================================================================
    def score_trade_quality(
        self,
        setup: TradeSetup,
        confluence_report: Optional[ConfluenceReport] = None,
        regime_report: Optional[RegimeReport] = None,
        ml_result: Optional[MLInferenceResult] = None,
    ) -> TradeQualityScore:
        """
        Computes a unified 0–100 trade quality score from 5 dimensions:
          SMC (25%) + Volume/VSA (20%) + Structure/Regime (20%) + MTF (20%) + ML (15%)
        Maps the composite to a letter grade: A+ (≥85), A (≥75), B (≥65), C (≥45), FILTERED (<45).
        """
        smc_score = self._score_smc_component(setup, confluence_report)
        volume_score = self._score_volume_component(confluence_report)
        structure_score = self._score_structure_component(setup, regime_report)
        mtf_score = self._score_mtf_component(confluence_report)
        ml_score = self._score_ml_component(setup, ml_result)

        # Weighted composite
        raw = (
            smc_score * 0.25
            + volume_score * 0.20
            + structure_score * 0.20
            + mtf_score * 0.20
            + ml_score * 0.15
        )
        raw = round(min(100.0, max(0.0, raw)), 1)

        grade = self._map_grade(raw)

        breakdown = {
            "SMC (25%)": round(smc_score * 0.25, 1),
            "Volume (20%)": round(volume_score * 0.20, 1),
            "Structure (20%)": round(structure_score * 0.20, 1),
            "MTF (20%)": round(mtf_score * 0.20, 1),
            "ML (15%)": round(ml_score * 0.15, 1),
        }

        qs = TradeQualityScore(
            raw_score=raw,
            grade=grade,
            smc_score=round(smc_score, 1),
            volume_score=round(volume_score, 1),
            structure_score=round(structure_score, 1),
            mtf_score=round(mtf_score, 1),
            ml_score=round(ml_score, 1),
            component_breakdown=breakdown,
        )
        setup.quality_score = qs
        return qs

    @staticmethod
    def _map_grade(raw: float) -> QualityGrade:
        if raw >= 85:
            return QualityGrade.A_PLUS
        elif raw >= 75:
            return QualityGrade.A
        elif raw >= 65:
            return QualityGrade.B
        elif raw >= 45:
            return QualityGrade.C
        else:
            return QualityGrade.FILTERED

    def _score_smc_component(
        self, setup: TradeSetup, rep: Optional[ConfluenceReport]
    ) -> float:
        """SMC component (0–100): FVG, Order Block, Liquidity Sweep, BOS/CHoCH."""
        score = 0.0

        # Setup type bonus
        type_bonus = {
            SetupType.SMC_PULLBACK_FVG: 40,
            SetupType.SMC_ORDER_BLOCK: 38,
            SetupType.LIQUIDITY_SWEEP_REVERSAL: 30,
            SetupType.CHART_PATTERN_BREAKOUT: 15,
            SetupType.VALUE_AREA_MEAN_REVERSION: 10,
        }
        score += type_bonus.get(setup.setup_type, 10)

        if rep:
            # Active FVGs near entry
            if rep.active_fvgs:
                for fvg in rep.active_fvgs[:3]:
                    if not fvg.is_mitigated:
                        score += 10
                        break

            # Active Order Blocks near entry
            if rep.active_obs:
                for ob in rep.active_obs[:3]:
                    if not ob.is_mitigated and not ob.is_invalidated:
                        score += 10
                        break

            # Market structure alignment (BOS/CHoCH)
            if rep.smc_state:
                if setup.direction == "LONG" and rep.smc_state.trend == "UPTREND":
                    score += 15
                elif setup.direction == "SHORT" and rep.smc_state.trend == "DOWNTREND":
                    score += 15
                elif rep.smc_state.recent_choch:
                    # CHoCH reversal alignment
                    if setup.direction == "LONG" and "BULLISH" in (rep.smc_state.recent_choch or ""):
                        score += 20
                    elif setup.direction == "SHORT" and "BEARISH" in (rep.smc_state.recent_choch or ""):
                        score += 20

            # R:R quality bonus
            if setup.effective_rr >= 3.0:
                score += 15
            elif setup.effective_rr >= 2.5:
                score += 10
            elif setup.effective_rr >= 2.0:
                score += 5

        return min(100.0, score)

    def _score_volume_component(self, rep: Optional[ConfluenceReport]) -> float:
        """Volume/VSA component (0–100): Volume profile, VSA patterns."""
        score = 50.0  # Baseline

        if rep:
            # Volume Profile alignment
            if rep.volume_profile:
                vp = rep.volume_profile
                if vp.total_volume > 0:
                    score += 15
                if vp.poc_price > 0:
                    score += 10

            # Overall confluence confidence as proxy for volume quality
            if rep.confidence_score >= 80:
                score += 20
            elif rep.confidence_score >= 65:
                score += 10

        return min(100.0, score)

    def _score_structure_component(
        self, setup: TradeSetup, regime: Optional[RegimeReport]
    ) -> float:
        """Structure/Regime component (0–100): Market regime alignment."""
        score = 40.0  # Baseline

        if regime:
            # Regime-direction alignment
            if setup.direction == "LONG":
                if regime.regime == MarketRegime.TRENDING_BULL:
                    score += 40
                elif regime.regime == MarketRegime.BREAKOUT:
                    score += 25
                elif regime.regime == MarketRegime.RANGING:
                    score += 10
                elif regime.regime == MarketRegime.HIGH_VOLATILITY_CHOP:
                    score -= 15
                elif regime.regime == MarketRegime.TRENDING_BEAR:
                    score -= 20
            else:  # SHORT
                if regime.regime == MarketRegime.TRENDING_BEAR:
                    score += 40
                elif regime.regime == MarketRegime.BREAKOUT:
                    score += 25
                elif regime.regime == MarketRegime.RANGING:
                    score += 10
                elif regime.regime == MarketRegime.HIGH_VOLATILITY_CHOP:
                    score -= 15
                elif regime.regime == MarketRegime.TRENDING_BULL:
                    score -= 20

            # ADX strength bonus
            if regime.adx >= 30:
                score += 15
            elif regime.adx >= 20:
                score += 5

        return min(100.0, max(0.0, score))

    def _score_mtf_component(self, rep: Optional[ConfluenceReport]) -> float:
        """Multi-Timeframe component (0–100): HTF/LTF alignment."""
        score = 30.0  # Baseline

        if rep and rep.timeframe_breakdown:
            aligned = 0
            total = 0
            overall = rep.overall_bias

            for tf_name, tf_conf in rep.timeframe_breakdown.items():
                total += 1
                if tf_conf.bias == overall:
                    aligned += 1
                elif tf_conf.bias == BiasType.NEUTRAL:
                    aligned += 0.5

            if total > 0:
                alignment_pct = aligned / total
                score += alignment_pct * 60

            # HTF-LTF alignment bonus
            if rep.htf_bias == rep.ltf_trigger:
                score += 10

        elif rep:
            # Fallback: use confidence score
            score = rep.confidence_score * 0.8

        return min(100.0, max(0.0, score))

    def _score_ml_component(
        self, setup: TradeSetup, ml: Optional[MLInferenceResult]
    ) -> float:
        """ML Prediction component (0–100): Model probability alignment."""
        if not ml:
            return 50.0  # Neutral when no ML available

        score = 20.0

        if setup.direction == "LONG":
            prob = ml.prob_bullish
        else:
            prob = ml.prob_bearish

        # Scale probability to score contribution
        # prob >= 0.65 → high score; prob <= 0.30 → penalty
        if prob >= 0.70:
            score += 60
        elif prob >= 0.55:
            score += 40
        elif prob >= 0.45:
            score += 20
        elif prob >= 0.35:
            score += 0
        else:
            score -= 10

        # Model confidence bonus
        if ml.model_confidence >= 0.25:
            score += 20
        elif ml.model_confidence >= 0.15:
            score += 10

        return min(100.0, max(0.0, score))


# ============================================================================
# Phase 6: Position-State Manager (Whipsaw Prevention & Reversal Warning)
# ============================================================================

class PositionStateManager:
    """
    Manages live position state to prevent whipsaw (opposite signals while
    trade is open) and detect early reversal warnings when trade reaches
    >= 1:1 R:R with reversal patterns emerging.
    """

    def __init__(self):
        verify_author_integrity()
        self.active_position: Optional[ActivePosition] = None
        self._trade_history: List[ActivePosition] = []

    @property
    def is_flat(self) -> bool:
        return self.active_position is None

    @property
    def has_open_trade(self) -> bool:
        return self.active_position is not None

    def open_position(self, setup: TradeSetup, current_price: float) -> ActivePosition:
        """Creates and registers a new open position from a trade setup."""
        pos = ActivePosition(
            position_id=setup.setup_id,
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            tp1_price=setup.tp1_price,
            tp2_price=setup.tp2_price,
            entry_time=setup.timestamp,
            current_price=current_price,
            state=PositionState.OPEN_LONG if setup.direction == "LONG" else PositionState.OPEN_SHORT,
        )
        self.active_position = pos
        return pos

    def close_position(self, reason: str = "Manual close") -> Optional[ActivePosition]:
        """Closes the active position and moves it to history."""
        if self.active_position is None:
            return None
        closed = self.active_position
        closed.state = PositionState.FLAT
        self._trade_history.append(closed)
        self.active_position = None
        return closed

    def update_position(
        self,
        current_price: float,
        candle: Optional[Dict[str, float]] = None,
    ) -> Optional[ActivePosition]:
        """
        Updates position P&L, R:R, and checks for early reversal warnings.
        Returns the updated position or None if flat.
        """
        pos = self.active_position
        if pos is None:
            return None

        pos.current_price = current_price
        risk = abs(pos.entry_price - pos.stop_loss)
        if risk <= 1e-8:
            risk = 1e-8

        # Calculate current R:R
        if pos.direction == "LONG":
            pnl = current_price - pos.entry_price
        else:
            pnl = pos.entry_price - current_price

        pos.current_rr = round(pnl / risk, 2)
        pos.current_pnl_pct = round((pnl / pos.entry_price) * 100, 2)

        # Track peak R:R
        if pos.current_rr > pos.peak_rr:
            pos.peak_rr = pos.current_rr

        # Check for TP1 hit → partial close state
        if pos.direction == "LONG" and current_price >= pos.tp1_price:
            pos.state = PositionState.PARTIAL_TP1
        elif pos.direction == "SHORT" and current_price <= pos.tp1_price:
            pos.state = PositionState.PARTIAL_TP1

        # Check for SL hit → auto close
        if pos.direction == "LONG" and current_price <= pos.stop_loss:
            return self.close_position("Stop Loss hit")
        elif pos.direction == "SHORT" and current_price >= pos.stop_loss:
            return self.close_position("Stop Loss hit")

        # Check for TP2 hit → auto close
        if pos.direction == "LONG" and current_price >= pos.tp2_price:
            return self.close_position("TP2 hit")
        elif pos.direction == "SHORT" and current_price <= pos.tp2_price:
            return self.close_position("TP2 hit")

        # Early Reversal Warning: >= 1:1 R:R and signs of pullback
        pos.reversal_warning = False
        pos.reversal_reason = ""
        if pos.current_rr >= 1.0 and candle:
            h = float(candle.get("high", current_price))
            l = float(candle.get("low", current_price))
            o = float(candle.get("open", current_price))
            c = float(candle.get("close", current_price))
            rng = max(1e-8, h - l)

            if pos.direction == "LONG":
                upper_wick = (h - max(o, c)) / rng
                is_red = c < o
                # Pullback from peak: peak_rr was high but now dropping
                rr_drawdown = pos.peak_rr - pos.current_rr
                if (upper_wick >= 0.40 and is_red) or rr_drawdown >= 0.5:
                    pos.reversal_warning = True
                    reasons = []
                    if upper_wick >= 0.40:
                        reasons.append(f"Strong upper wick rejection ({upper_wick*100:.0f}%)")
                    if is_red:
                        reasons.append("Bearish candle forming")
                    if rr_drawdown >= 0.5:
                        reasons.append(f"R:R dropping from peak {pos.peak_rr:.1f}R → {pos.current_rr:.1f}R")
                    pos.reversal_reason = " | ".join(reasons)
            else:  # SHORT
                lower_wick = (min(o, c) - l) / rng
                is_green = c > o
                rr_drawdown = pos.peak_rr - pos.current_rr
                if (lower_wick >= 0.40 and is_green) or rr_drawdown >= 0.5:
                    pos.reversal_warning = True
                    reasons = []
                    if lower_wick >= 0.40:
                        reasons.append(f"Strong lower wick bounce ({lower_wick*100:.0f}%)")
                    if is_green:
                        reasons.append("Bullish candle forming")
                    if rr_drawdown >= 0.5:
                        reasons.append(f"R:R dropping from peak {pos.peak_rr:.1f}R → {pos.current_rr:.1f}R")
                    pos.reversal_reason = " | ".join(reasons)

        return pos

    def should_suppress_signal(self, new_direction: str) -> bool:
        """
        Returns True if a new signal should be suppressed (whipsaw prevention).
        Suppresses opposite-direction signals while a trade is open.
        """
        if self.active_position is None:
            return False
        return self.active_position.direction != new_direction

    def get_trade_history(self) -> List[ActivePosition]:
        """Returns list of completed trades."""
        return list(self._trade_history)

