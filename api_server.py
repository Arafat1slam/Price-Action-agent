"""
api_server.py - FastAPI REST Server for Price Action Engine
Exposes analysis, trade setup, and streaming endpoints for the ARIS Desktop UI.
"""

import os
import sys
import asyncio
import json
import traceback
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEFAULT_TIMEFRAME,
    DEFAULT_CANDLE_LIMIT,
    SUPPORTED_TIMEFRAMES,
    logger,
)
from symbol_mapper import resolve_symbol, suggest_symbols
from binance_client import BinanceClient
from price_action_engine import PriceActionEngine, AnalysisResult

app = FastAPI(
    title="ARIS Price Action API",
    description="Institutional Price Action Analysis Engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
binance_client = BinanceClient()
engine = PriceActionEngine()


# ─── Response Models ────────────────────────────────────────────────

class FVGResponse(BaseModel):
    top: float
    bottom: float
    midpoint: float
    bias: str
    is_mitigated: bool = False
    mitigation_pct: float = 0.0

class OrderBlockResponse(BaseModel):
    top: float
    bottom: float
    bias: str
    volume: float = 0.0
    is_mitigated: bool = False
    is_breaker: bool = False

class LiquiditySweepResponse(BaseModel):
    level_breached: float
    sweep_type: str
    wick_rejection_pct: float = 0.0
    volume_ratio: float = 0.0

class StructureEventResponse(BaseModel):
    event_type: str
    broken_level: float

class SMCResponse(BaseModel):
    active_bullish_fvgs: List[FVGResponse] = []
    active_bearish_fvgs: List[FVGResponse] = []
    active_bullish_obs: List[OrderBlockResponse] = []
    active_bearish_obs: List[OrderBlockResponse] = []
    recent_sweeps: List[LiquiditySweepResponse] = []
    structure_events: List[StructureEventResponse] = []

class VolumeProfileResponse(BaseModel):
    poc_price: float = 0.0
    vah_price: float = 0.0
    val_price: float = 0.0
    total_volume: float = 0.0
    hvn_levels: List[float] = []
    lvn_levels: List[float] = []

class MLResponse(BaseModel):
    prob_bullish: float = 0.5
    prob_bearish: float = 0.5
    prob_neutral: float = 0.0
    model_confidence: float = 0.0

class QualityScoreResponse(BaseModel):
    raw_score: float = 0.0
    grade: str = "FILTERED"
    smc_score: float = 0.0
    volume_score: float = 0.0
    structure_score: float = 0.0
    mtf_score: float = 0.0
    ml_score: float = 0.0

class TradeSetupResponse(BaseModel):
    setup_id: str = ""
    setup_type: str = ""
    direction: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    risk_reward_tp1: float = 0.0
    risk_reward_tp2: float = 0.0
    effective_rr: float = 0.0
    confidence_score: int = 0
    rationale: List[str] = []
    invalidation_reason: str = ""
    tp1_probability: int = 0
    tp2_probability: int = 0
    recommended_risk_pct: float = 0.0
    position_size_usd: float = 0.0
    execution_state: str = "WAITING_FOR_PRICE"
    entry_distance_pct: float = 0.0
    entry_action: str = ""
    quality_score: Optional[QualityScoreResponse] = None
    trailing_stop: float = 0.0

class TimeframeConfluenceResponse(BaseModel):
    timeframe: str
    bias: str
    score: float = 0.0
    key_signals: List[str] = []

class ConfluenceResponse(BaseModel):
    overall_bias: str = "NEUTRAL"
    confidence_score: int = 50
    htf_bias: str = "NEUTRAL"
    ltf_trigger: str = "NEUTRAL"
    timeframe_breakdown: Dict[str, TimeframeConfluenceResponse] = {}

class ChartPatternResponse(BaseModel):
    name: str
    bias: str
    quality_score: float = 0.0
    neckline_price: Optional[float] = None
    projected_target: Optional[float] = None
    invalidation_level: float = 0.0

class RegimeResponse(BaseModel):
    regime: str = "RANGING"
    regime_label: str = "Range-Bound"
    confidence: int = 50
    adx: float = 0.0
    atr_ratio: float = 0.0
    bb_bandwidth_pct: float = 0.0
    volatility_state: str = "NORMAL"
    recommended_strategy: str = ""
    key_drivers: List[str] = []

class AnalysisResponse(BaseModel):
    symbol: str
    timeframe: str
    current_price: float
    is_candle_closed: bool
    bias: str
    confidence: int
    patterns: List[str]
    trend: str
    trend_detail: str
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    rough_target: Optional[float] = None
    volume_state: str = ""
    momentum_streak: int = 0
    momentum_state: str = ""
    timestamp: str = ""
    smc: Optional[SMCResponse] = None
    volume_profile: Optional[VolumeProfileResponse] = None
    ml_prediction: Optional[MLResponse] = None
    trade_setup: Optional[TradeSetupResponse] = None
    confluence: Optional[ConfluenceResponse] = None
    chart_patterns: List[ChartPatternResponse] = []
    regime: Optional[RegimeResponse] = None


def serialize_analysis(res: AnalysisResult) -> dict:
    """Convert AnalysisResult to JSON-serializable dict."""
    data = {
        "symbol": res.symbol,
        "timeframe": res.timeframe,
        "current_price": res.current_price,
        "is_candle_closed": res.is_candle_closed,
        "bias": res.bias if isinstance(res.bias, str) else res.bias.value,
        "confidence": res.confidence,
        "patterns": res.patterns or [],
        "trend": res.trend,
        "trend_detail": res.trend_detail,
        "nearest_support": res.nearest_support,
        "nearest_resistance": res.nearest_resistance,
        "rough_target": res.rough_target,
        "volume_state": res.volume_state or "",
        "momentum_streak": res.momentum_streak or 0,
        "momentum_state": res.momentum_state or "",
        "timestamp": res.timestamp or datetime.now(timezone.utc).isoformat(),
    }

    # SMC Report
    smc = res.smc_report
    if smc:
        smc_data = {"active_bullish_fvgs": [], "active_bearish_fvgs": [], "active_bullish_obs": [], "active_bearish_obs": [], "recent_sweeps": [], "structure_events": []}
        for fvg in getattr(smc, "active_bullish_fvgs", []):
            smc_data["active_bullish_fvgs"].append({"top": fvg.top, "bottom": fvg.bottom, "midpoint": fvg.midpoint, "bias": "BULLISH", "is_mitigated": getattr(fvg, "is_mitigated", False), "mitigation_pct": getattr(fvg, "mitigation_pct", 0.0)})
        for fvg in getattr(smc, "active_bearish_fvgs", []):
            smc_data["active_bearish_fvgs"].append({"top": fvg.top, "bottom": fvg.bottom, "midpoint": fvg.midpoint, "bias": "BEARISH", "is_mitigated": getattr(fvg, "is_mitigated", False), "mitigation_pct": getattr(fvg, "mitigation_pct", 0.0)})
        for ob in getattr(smc, "active_bullish_obs", []):
            smc_data["active_bullish_obs"].append({"top": ob.top, "bottom": ob.bottom, "bias": "BULLISH", "volume": getattr(ob, "volume", 0), "is_mitigated": getattr(ob, "is_mitigated", False), "is_breaker": getattr(ob, "is_breaker", False)})
        for ob in getattr(smc, "active_bearish_obs", []):
            smc_data["active_bearish_obs"].append({"top": ob.top, "bottom": ob.bottom, "bias": "BEARISH", "volume": getattr(ob, "volume", 0), "is_mitigated": getattr(ob, "is_mitigated", False), "is_breaker": getattr(ob, "is_breaker", False)})
        for sw in getattr(smc, "recent_sweeps", []):
            smc_data["recent_sweeps"].append({"level_breached": getattr(sw, "sweep_level", getattr(sw, "level_breached", 0)), "sweep_type": getattr(sw, "sweep_type", "").value if hasattr(getattr(sw, "sweep_type", ""), "value") else str(getattr(sw, "sweep_type", "")), "wick_rejection_pct": getattr(sw, "wick_rejection_pct", 0), "volume_ratio": getattr(sw, "volume_ratio", 0)})
        for ev in getattr(smc, "recent_structure_events", []):
            smc_data["structure_events"].append({"event_type": ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type), "broken_level": getattr(ev, "broken_level", 0)})
        data["smc"] = smc_data
    else:
        data["smc"] = None

    # Confluence Report
    conf = res.confluence_report
    if conf:
        conf_data = {
            "overall_bias": conf.overall_bias.value if hasattr(conf.overall_bias, "value") else str(conf.overall_bias),
            "confidence_score": conf.confidence_score,
            "htf_bias": conf.htf_bias.value if hasattr(conf.htf_bias, "value") else str(conf.htf_bias),
            "ltf_trigger": conf.ltf_trigger.value if hasattr(conf.ltf_trigger, "value") else str(conf.ltf_trigger),
            "timeframe_breakdown": {},
        }
        for tf_key, tf_val in (conf.timeframe_breakdown or {}).items():
            conf_data["timeframe_breakdown"][tf_key] = {
                "timeframe": tf_val.timeframe,
                "bias": tf_val.bias.value if hasattr(tf_val.bias, "value") else str(tf_val.bias),
                "score": getattr(tf_val, "score", 0.0),
                "key_signals": getattr(tf_val, "key_signals", []),
            }
        data["confluence"] = conf_data
    else:
        data["confluence"] = None

    # Trade Setup
    setup = res.trade_setup
    if setup:
        qs = setup.quality_score
        qs_data = None
        if qs:
            qs_data = {
                "raw_score": qs.raw_score,
                "grade": qs.grade.value if hasattr(qs.grade, "value") else str(qs.grade),
                "smc_score": qs.smc_score,
                "volume_score": qs.volume_score,
                "structure_score": qs.structure_score,
                "mtf_score": qs.mtf_score,
                "ml_score": qs.ml_score,
            }
        data["trade_setup"] = {
            "setup_id": setup.setup_id,
            "setup_type": setup.setup_type.value if hasattr(setup.setup_type, "value") else str(setup.setup_type),
            "direction": setup.direction,
            "entry_price": setup.entry_price,
            "stop_loss": setup.stop_loss,
            "tp1_price": setup.tp1_price,
            "tp2_price": setup.tp2_price,
            "tp3_price": getattr(setup, "tp3_price", 0.0),
            "risk_reward_tp1": setup.risk_reward_tp1,
            "risk_reward_tp2": setup.risk_reward_tp2,
            "effective_rr": setup.effective_rr,
            "confidence_score": setup.confidence_score,
            "rationale": setup.rationale or [],
            "invalidation_reason": setup.invalidation_reason or "",
            "tp1_probability": setup.tp1_probability,
            "tp2_probability": setup.tp2_probability,
            "recommended_risk_pct": setup.recommended_risk_pct,
            "position_size_usd": setup.position_size_usd,
            "execution_state": setup.execution_state,
            "entry_distance_pct": setup.entry_distance_pct,
            "entry_action": setup.entry_action or "",
            "quality_score": qs_data,
            "trailing_stop": getattr(setup, "trailing_stop", 0.0),
        }
    else:
        data["trade_setup"] = None

    # ML Result
    ml = res.ml_result
    if ml:
        data["ml_prediction"] = {
            "prob_bullish": ml.prob_bullish,
            "prob_bearish": ml.prob_bearish,
            "prob_neutral": ml.prob_neutral,
            "model_confidence": ml.model_confidence,
        }
    else:
        data["ml_prediction"] = None

    # Chart Patterns
    data["chart_patterns"] = []
    for cp in (res.chart_patterns or []):
        data["chart_patterns"].append({
            "name": cp.name,
            "bias": cp.bias.value if hasattr(cp.bias, "value") else str(cp.bias),
            "quality_score": cp.quality_score,
            "neckline_price": cp.neckline_price,
            "projected_target": cp.projected_target,
            "invalidation_level": cp.invalidation_level,
        })

    # Regime Report
    regime = res.regime_report
    if regime:
        data["regime"] = {
            "regime": regime.regime.value if hasattr(regime.regime, "value") else str(regime.regime),
            "regime_label": regime.regime_label,
            "confidence": regime.confidence,
            "adx": regime.adx,
            "atr_ratio": regime.atr_ratio,
            "bb_bandwidth_pct": regime.bb_bandwidth_pct,
            "volatility_state": regime.volatility_state,
            "recommended_strategy": regime.recommended_strategy,
            "key_drivers": regime.key_drivers or [],
        }
    else:
        data["regime"] = None

    # Volume Profile from confluence
    if conf and conf.volume_profile:
        vp = conf.volume_profile
        data["volume_profile"] = {
            "poc_price": vp.poc_price,
            "vah_price": vp.vah_price,
            "val_price": vp.val_price,
            "total_volume": vp.total_volume,
            "hvn_levels": vp.hvn_levels or [],
            "lvn_levels": vp.lvn_levels or [],
        }
    else:
        data["volume_profile"] = None

    return data


# ─── REST Endpoints ─────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "online", "engine": "Price Action v2.0", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/timeframes")
async def get_timeframes():
    return {"timeframes": SUPPORTED_TIMEFRAMES, "default": DEFAULT_TIMEFRAME}


@app.get("/api/analyze")
async def analyze(
    symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
    timeframe: str = Query("1h", description="Candlestick timeframe"),
    limit: int = Query(150, description="Number of candles"),
):
    """Run full Price Action analysis on a symbol."""
    try:
        resolved = resolve_symbol(symbol)
        if not resolved:
            return {"error": f"Symbol '{symbol}' not found. Try: {suggest_symbols(symbol)}"}

        klines = binance_client.get_klines(resolved, timeframe, limit)
        if not klines or len(klines) < 20:
            return {"error": f"Insufficient data for {resolved} ({timeframe})"}

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        result: AnalysisResult = engine.analyze(df, resolved, timeframe)
        return serialize_analysis(result)

    except Exception as e:
        logger.error(f"Analysis error: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


@app.get("/api/symbols")
async def search_symbols(q: str = Query("", description="Search query")):
    """Search for tradable symbols."""
    try:
        if q:
            suggestions = suggest_symbols(q)
            return {"symbols": suggestions[:20]}
        # Return popular pairs
        return {"symbols": [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
            "LINKUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
        ]}
    except Exception as e:
        return {"symbols": [], "error": str(e)}


# ─── WebSocket for Live Streaming Analysis ──────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket endpoint for live streaming analysis.
    Client sends: {"symbol": "BTCUSDT", "timeframe": "15m"}
    Server responds with periodic analysis updates.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            symbol = msg.get("symbol", "BTCUSDT")
            timeframe = msg.get("timeframe", "15m")

            try:
                resolved = resolve_symbol(symbol)
                if resolved:
                    klines = binance_client.get_klines(resolved, timeframe, 150)
                    if klines and len(klines) >= 20:
                        df = pd.DataFrame(klines, columns=[
                            "timestamp", "open", "high", "low", "close", "volume",
                            "close_time", "quote_volume", "trades", "taker_buy_base",
                            "taker_buy_quote", "ignore"
                        ])
                        for col in ["open", "high", "low", "close", "volume"]:
                            df[col] = df[col].astype(float)

                        result = engine.analyze(df, resolved, timeframe)
                        await websocket.send_json(serialize_analysis(result))
                    else:
                        await websocket.send_json({"error": "Insufficient data"})
                else:
                    await websocket.send_json({"error": f"Unknown symbol: {symbol}"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PRICE_ACTION_PORT", "8899"))
    print(f"🚀 ARIS Price Action API starting on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
