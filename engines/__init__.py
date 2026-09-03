"""
engines - Unified Price Action Analysis & Institutional Confluence Suite
"""

from engines.smc_engine import SMCEngine, SMCAnalysisReport
from engines.indicators_engine import (
    VolumeProfileEngine,
    VWAPEngine,
    VSAEngine,
    KDESupportResistanceEngine,
    FibonacciEngine,
    IndicatorConfluenceEngine,
    evaluate_confluence,
)
from engines.chart_pattern_engine import ChartPatternEngine
from engines.ml_predictor import MLPredictor, FeatureExtractor
from engines.ml_engine import LabelGenerator, ModelTrainer, PurgedTimeSeriesSplit
from engines.confluence_engine import ConfluenceEngine
from engines.trade_setup_engine import TradeSetupEngine
from engines.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    SMCPriceActionStrategy,
    TradeRecord,
    Position,
    TradeSignal,
    TradeSide,
    ExitReason,
)

__all__ = [
    "SMCEngine",
    "SMCAnalysisReport",
    "VolumeProfileEngine",
    "VWAPEngine",
    "VSAEngine",
    "KDESupportResistanceEngine",
    "FibonacciEngine",
    "IndicatorConfluenceEngine",
    "evaluate_confluence",
    "ChartPatternEngine",
    "MLPredictor",
    "FeatureExtractor",
    "LabelGenerator",
    "ModelTrainer",
    "PurgedTimeSeriesSplit",
    "ConfluenceEngine",
    "TradeSetupEngine",
    "BacktestEngine",
    "BacktestConfig",
    "BacktestMetrics",
    "BacktestResult",
    "SMCPriceActionStrategy",
    "TradeRecord",
    "Position",
    "TradeSignal",
    "TradeSide",
    "ExitReason",
]

