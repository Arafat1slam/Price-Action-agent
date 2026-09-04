"""
core/models.py - Canonical Data Models for Price Action Agent v2.0
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

class BiasType(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"

class SetupType(str, Enum):
    SMC_PULLBACK_FVG = "SMC_PULLBACK_FVG"
    SMC_ORDER_BLOCK = "SMC_ORDER_BLOCK"
    LIQUIDITY_SWEEP_REVERSAL = "LIQUIDITY_SWEEP_REVERSAL"
    CHART_PATTERN_BREAKOUT = "CHART_PATTERN_BREAKOUT"
    VALUE_AREA_MEAN_REVERSION = "VALUE_AREA_MEAN_REVERSION"

class MarketRegime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"
    HIGH_VOLATILITY_CHOP = "HIGH_VOLATILITY_CHOP"

@dataclass
class RegimeReport:
    regime: MarketRegime
    regime_label: str
    confidence: int               # 0 - 100%
    adx: float                    # Average Directional Index (14)
    plus_di: float                # +DI
    minus_di: float               # -DI
    atr_ratio: float              # Current ATR vs 20-period baseline
    bb_bandwidth_pct: float       # Bollinger Bandwidth %
    volatility_state: str         # "LOW", "NORMAL", "EXPANDING", "EXTREME"
    recommended_strategy: str     # Strategic trade guidance
    key_drivers: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool
    timeframe: str
    symbol: str

@dataclass
class FairValueGap:
    top: float
    bottom: float
    midpoint: float  # Consequent Encroachment (50%)
    bias: BiasType   # BULLISH (gap up) or BEARISH (gap down)
    created_time: Any
    is_mitigated: bool = False
    mitigation_pct: float = 0.0
    volume_delta: float = 0.0

@dataclass
class OrderBlock:
    top: float
    bottom: float
    bias: BiasType
    created_time: Any
    volume: float
    is_mitigated: bool = False
    is_invalidated: bool = False

@dataclass
class LiquiditySweep:
    level_breached: float
    sweep_type: str  # "BUY_SIDE" (BSL) or "SELL_SIDE" (SSL)
    wick_rejection_pct: float
    volume_ratio: float
    timestamp: Any

@dataclass
class MarketStructureState:
    trend: str               # "UPTREND", "DOWNTREND", "RANGE"
    recent_bos: Optional[str] # "BULLISH_BOS", "BEARISH_BOS"
    recent_choch: Optional[str] # "BULLISH_CHOCH", "BEARISH_CHOCH"
    last_swing_high: float
    last_swing_low: float

@dataclass
class ChartPattern:
    name: str                # e.g., "DOUBLE_BOTTOM", "HEAD_AND_SHOULDERS", "FALLING_WEDGE"
    bias: BiasType
    quality_score: float     # 0.0 to 1.0
    neckline_price: Optional[float]
    projected_target: Optional[float]
    invalidation_level: float
    candle_span: int
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VolumeProfileZone:
    poc_price: float         # Point of Control
    vah_price: float         # Value Area High
    val_price: float         # Value Area Low
    total_volume: float
    hvn_levels: List[float] = field(default_factory=list)  # High Volume Nodes
    lvn_levels: List[float] = field(default_factory=list)  # Low Volume Nodes

@dataclass
class KDELevel:
    price: float
    density: float
    level_type: str          # "SUPPORT" or "RESISTANCE"
    touches: int

@dataclass
class MLInferenceResult:
    prob_bullish: float      # Model probability of hitting +2R before -1R
    prob_bearish: float
    prob_neutral: float
    model_confidence: float  # Margin over uniform distribution
    feature_contributions: Dict[str, float] = field(default_factory=dict)

@dataclass
class TimeframeConfluence:
    timeframe: str
    bias: BiasType
    score: float             # -1.0 (max bearish) to +1.0 (max bullish)
    key_signals: List[str] = field(default_factory=list)

@dataclass
class ConfluenceReport:
    symbol: str
    timestamp: Any
    current_price: float
    overall_bias: BiasType
    confidence_score: int    # 0 to 100%
    htf_bias: BiasType
    ltf_trigger: BiasType
    timeframe_breakdown: Dict[str, TimeframeConfluence] = field(default_factory=dict)
    smc_state: Optional[MarketStructureState] = None
    active_patterns: List[ChartPattern] = field(default_factory=list)
    active_fvgs: List[FairValueGap] = field(default_factory=list)
    active_obs: List[OrderBlock] = field(default_factory=list)
    volume_profile: Optional[VolumeProfileZone] = None
    ml_prediction: Optional[MLInferenceResult] = None

class TradeStyle(str, Enum):
    SCALP = "SCALP"
    INTRADAY = "INTRADAY"
    SWING = "SWING"

class ExecutionState(str, Enum):
    WAITING_FOR_PRICE = "WAITING_FOR_PRICE"
    IN_ENTRY_ZONE = "IN_ENTRY_ZONE"
    CONFIRMED_ENTRY_TRIGGER = "CONFIRMED_ENTRY_TRIGGER"
    INVALIDATED = "INVALIDATED"
    TARGET_HIT = "TARGET_HIT"

@dataclass
class TradeSetup:
    setup_id: str
    symbol: str
    timestamp: Any
    setup_type: SetupType
    direction: str           # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    risk_reward_tp1: float
    risk_reward_tp2: float
    effective_rr: float
    confidence_score: int
    rationale: List[str] = field(default_factory=list)
    invalidation_reason: str = ""
    tp1_probability: int = 70               # Estimated % chance of hitting TP1
    tp2_probability: int = 55               # Estimated % chance of hitting TP2
    recommended_risk_pct: float = 1.0       # Recommended portfolio risk % (e.g. 1.5%)
    position_size_usd: float = 1000.0       # Suggested position sizing
    execution_state: str = "WAITING_FOR_PRICE" # "WAITING_FOR_PRICE", "IN_ENTRY_ZONE", "CONFIRMED_ENTRY_TRIGGER"
    entry_distance_pct: float = 0.0         # Distance from current price to entry (%): e.g. -0.45%
    entry_action: str = ""                  # Actionable instruction (WAIT vs ENTER NOW)
    quality_score: Optional['TradeQualityScore'] = None  # Unified 0-100 quality score
    tp3_price: float = 0.0                  # Institutional runner target (e.g. 1:4.0 R:R)
    risk_reward_tp3: float = 0.0            # R:R for TP3
    trailing_stop: float = 0.0              # Dynamic ATR trailing stop level


class QualityGrade(str, Enum):
    A_PLUS = "A+"    # ≥ 85 — Elite institutional setup
    A = "A"          # ≥ 75 — High-quality setup
    B = "B"          # ≥ 65 — Good setup, moderate confidence
    C = "C"          # ≥ 45 — Marginal setup, use caution
    FILTERED = "FILTERED"  # < 45 — Below quality threshold, do not trade


@dataclass
class TradeQualityScore:
    """Unified 0–100 trade quality score combining 5 analytical dimensions."""
    raw_score: float             # 0.0 – 100.0 composite score
    grade: QualityGrade          # Letter grade mapped from raw_score
    smc_score: float             # 0–100 SMC component (25% weight)
    volume_score: float          # 0–100 Volume/VSA component (20% weight)
    structure_score: float       # 0–100 Structure/Regime component (20% weight)
    mtf_score: float             # 0–100 Multi-Timeframe component (20% weight)
    ml_score: float              # 0–100 ML Prediction component (15% weight)
    component_breakdown: Dict[str, float] = field(default_factory=dict)  # Weighted contributions


class PositionState(str, Enum):
    FLAT = "FLAT"                # No open position
    OPEN_LONG = "OPEN_LONG"      # Long position active
    OPEN_SHORT = "OPEN_SHORT"    # Short position active
    PARTIAL_TP1 = "PARTIAL_TP1"  # TP1 hit, partial close, trailing remainder
    PARTIAL_TP2 = "PARTIAL_TP2"  # TP2 hit, second scale-out, trailing runner


@dataclass
class ActivePosition:
    """Tracks a live open position for whipsaw prevention and reversal warnings."""
    position_id: str
    symbol: str
    direction: str               # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    entry_time: Any
    current_price: float = 0.0
    current_pnl_pct: float = 0.0
    current_rr: float = 0.0      # Current R:R achieved (e.g. 1.5 means 1.5R profit)
    state: PositionState = PositionState.OPEN_LONG
    reversal_warning: bool = False
    reversal_reason: str = ""
    peak_rr: float = 0.0         # Highest R:R reached during this trade
    tp3_price: float = 0.0
    trailing_stop: float = 0.0