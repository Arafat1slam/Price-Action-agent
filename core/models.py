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