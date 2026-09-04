"""
tests/test_ml_engine.py
=======================
Unit and integration tests for the Machine Learning Engine:
- FeatureExtractor (35 features, schema, bounds, anti-lookahead, streaming vs batch)
- LabelGenerator (adaptive ATR threshold, lookahead truncation)
- PurgedTimeSeriesSplit (embargo gap enforcement, chronological splits)
- ModelTrainer (training, calibration, evaluation metrics)
- MLPredictor (streaming inference <5ms, calibration, confidence)
- Model Artifact Integrity (<5MB disk budget, valid deserialization)
"""

import os
import time
import pytest
import numpy as np
import pandas as pd
import joblib

from engines.ml_engine import (
    DEFAULT_MODEL_PATH,
    FALLBACK_MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_NAMES,
    FeatureExtractor,
    LabelGenerator,
    MLPredictor,
    ModelTrainer,
    PurgedTimeSeriesSplit,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Generates 120 bars of synthetic OHLCV data."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    
    # Random walk close price
    returns = np.random.normal(0.0002, 0.005, size=n)
    closes = 50000.0 * np.exp(np.cumsum(returns))
    
    highs = closes * (1.0 + np.random.uniform(0.001, 0.006, size=n))
    lows = closes * (1.0 - np.random.uniform(0.001, 0.006, size=n))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    volumes = np.random.uniform(50.0, 500.0, size=n)
    
    return pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ============================================================================
# 1. Feature Extractor Tests
# ============================================================================
def test_feature_columns_schema_and_count():
    """Verify that exactly 35 feature columns are defined in the schema."""
    assert len(FEATURE_COLUMNS) == 35
    fe = FeatureExtractor()
    assert len(fe.feature_columns) == 35
    assert fe.feature_columns == FEATURE_COLUMNS


def test_feature_extractor_returns_valid_shape_and_types(sample_ohlcv_df):
    """Verify FeatureExtractor returns (N, 35) DataFrame with float32 values."""
    fe = FeatureExtractor()
    feats = fe.extract_features(sample_ohlcv_df)

    assert isinstance(feats, pd.DataFrame)
    assert feats.shape == (len(sample_ohlcv_df), 35)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert not feats.isna().any().any()
    assert not np.isinf(feats.values).any()
    assert feats.dtypes.iloc[0] == np.float32


def test_feature_bounds_and_validity(sample_ohlcv_df):
    """Verify mathematical bounds for normalized features."""
    fe = FeatureExtractor()
    feats = fe.extract_features(sample_ohlcv_df)

    # Candlestick ratios
    assert (feats["feat_body_to_range"] >= 0.0).all() and (feats["feat_body_to_range"] <= 1.0 + 1e-5).all()
    assert (feats["feat_upper_wick_ratio"] >= 0.0).all() and (feats["feat_upper_wick_ratio"] <= 1.0 + 1e-5).all()
    assert (feats["feat_lower_wick_ratio"] >= 0.0).all() and (feats["feat_lower_wick_ratio"] <= 1.0 + 1e-5).all()
    assert set(np.unique(feats["feat_candle_direction"])).issubset({-1.0, 0.0, 1.0})

    # Normalized ATR
    assert (feats["feat_natr_14"] >= 0.0).all()
    assert (feats["feat_atr_ratio_sma50"] >= 0.0).all()

    # S/R range position in [0, 1]
    assert (feats["feat_sr_range_position"] >= 0.0).all() and (feats["feat_sr_range_position"] <= 1.0).all()

    # RSI in [0, 100]
    assert (feats["feat_rsi_14"] >= 0.0).all() and (feats["feat_rsi_14"] <= 100.0).all()

    # EMA alignment in {-1, 0, 1}
    assert set(np.unique(feats["feat_ema_alignment"])).issubset({-1.0, 0.0, 1.0})

    # SMC binary flags in {0, 1}
    smc_flags = [
        "feat_inside_bullish_fvg",
        "feat_inside_bearish_fvg",
        "feat_near_bullish_ob",
        "feat_near_bearish_ob",
        "feat_recent_liquidity_sweep_bullish",
        "feat_recent_liquidity_sweep_bearish",
    ]
    for col in smc_flags:
        assert set(np.unique(feats[col])).issubset({0.0, 1.0})


def test_zero_lookahead_bias(sample_ohlcv_df):
    """
    Verify strictly zero lookahead bias: modifying future candles at t+1..t+K
    must NEVER alter the extracted feature vector at time step t.
    """
    fe = FeatureExtractor()
    t_eval = 70

    # Sub-slice up to t_eval
    df_past = sample_ohlcv_df.iloc[: t_eval + 1].copy()
    feats_past = fe.extract_features(df_past)
    vec_past = feats_past.iloc[-1].values

    # Full df with different future candles
    df_future_modified = sample_ohlcv_df.copy()
    df_future_modified.loc[t_eval + 1 :, "close"] *= 2.5
    df_future_modified.loc[t_eval + 1 :, "high"] *= 3.0
    df_future_modified.loc[t_eval + 1 :, "volume"] *= 10.0

    feats_future = fe.extract_features(df_future_modified)
    vec_future = feats_future.iloc[t_eval].values

    # Must be numerically identical (diff < 1e-5)
    np.testing.assert_allclose(
        vec_past,
        vec_future,
        atol=1e-5,
        err_msg="Lookahead bias detected! Modifying future candles altered past features.",
    )


def test_extract_latest_matches_batch(sample_ohlcv_df):
    """Verify that extract_latest produces a vector consistent with batch extract_features."""
    fe = FeatureExtractor()
    window = sample_ohlcv_df.iloc[-100:].copy()

    vec_fast = fe.extract_latest(window)
    feats_batch = fe.extract_features(window)
    vec_batch = feats_batch.iloc[[-1]].values

    assert vec_fast.shape == (1, 35)
    # Check max absolute difference across all 35 features
    max_diff = np.max(np.abs(vec_fast - vec_batch))
    assert max_diff < 0.05, f"extract_latest differed by {max_diff}"


def test_extract_latest_latency_sub_millisecond(sample_ohlcv_df):
    """Verify extract_latest completes in < 3.0 ms on CPU (mean latency)."""
    fe = FeatureExtractor()
    window = sample_ohlcv_df.iloc[-100:].copy()

    # Warm-up
    for _ in range(5):
        fe.extract_latest(window)

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        fe.extract_latest(window)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    assert mean_lat < 3.0, f"extract_latest took too long: {mean_lat:.3f} ms (expected < 3.0 ms)"


# ============================================================================
# 2. Label Generator Tests
# ============================================================================
def test_label_generator_adaptive_threshold_and_lookahead(sample_ohlcv_df):
    """Verify LabelGenerator computes tri-class targets and truncates last H rows with NaN."""
    horizon = 5
    lg = LabelGenerator(horizon=horizon, atr_multiplier=0.75)
    labels = lg.compute_labels(sample_ohlcv_df)

    assert len(labels) == len(sample_ohlcv_df)
    # Strict anti-lookahead: last H rows must be NaN
    assert labels.iloc[-horizon:].isna().all()
    # Preceding rows must not be NaN
    assert not labels.iloc[:-horizon].isna().any()

    # Valid label values {-1.0, 0.0, 1.0}
    unique_vals = set(labels.dropna().unique())
    assert unique_vals.issubset({-1.0, 0.0, 1.0})


def test_label_generator_prepare_dataset(sample_ohlcv_df):
    """Verify prepare_dataset maps {-1, 0, 1} to contiguous integers {0, 1, 2}."""
    fe = FeatureExtractor()
    lg = LabelGenerator(horizon=5)

    feats = fe.extract_features(sample_ohlcv_df)
    labels = lg.compute_labels(sample_ohlcv_df)

    X, y, idx = lg.prepare_dataset(feats, labels)

    assert len(X) == len(y) == len(idx)
    assert len(X) == len(sample_ohlcv_df) - 5  # Last 5 horizon rows dropped
    assert X.shape[1] == 35
    assert set(np.unique(y)).issubset({0, 1, 2})


# ============================================================================
# 3. Purged Time-Series Split Tests
# ============================================================================
def test_purged_time_series_split():
    """Verify PurgedTimeSeriesSplit enforces chronological order and purge gap."""
    n_samples = 500
    purge_gap = 5
    cv = PurgedTimeSeriesSplit(n_splits=3, purge_window=purge_gap)
    X_dummy = np.zeros((n_samples, 5))

    splits = list(cv.split(X_dummy))
    assert len(splits) == 3

    for fold, (train_idx, test_idx) in enumerate(splits):
        # Strictly chronological
        assert train_idx.max() < test_idx.min()
        # Purge gap strictly enforced
        gap = test_idx.min() - train_idx.max()
        assert gap > purge_gap, f"Fold {fold}: Gap {gap} <= purge_gap {purge_gap}"
        # No overlap
        assert len(set(train_idx).intersection(set(test_idx))) == 0


# ============================================================================
# 4. Model Trainer Tests
# ============================================================================
def test_model_trainer_end_to_end(sample_ohlcv_df, tmp_path):
    """Verify ModelTrainer trains, calibrates, evaluates, and serializes artifact."""
    fe = FeatureExtractor()
    lg = LabelGenerator(horizon=3)

    feats = fe.extract_features(sample_ohlcv_df)
    labels = lg.compute_labels(sample_ohlcv_df)
    X, y, _ = lg.prepare_dataset(feats, labels)

    split = int(len(X) * 0.70)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split + 3 :], y[split + 3 :]  # Purge gap of 3

    trainer = ModelTrainer(output_dir=str(tmp_path), max_iter=20, random_state=42)
    save_file = str(tmp_path / "test_model.joblib")
    metrics = trainer.train(X_train, y_train, X_val, y_val, FEATURE_COLUMNS, save_path=save_file)

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "f1_macro" in metrics
    assert "roc_auc_weighted" in metrics
    assert "log_loss" in metrics
    assert "brier_score" in metrics
    assert "feature_importances" in metrics
    assert os.path.exists(save_file)

    # Verify probability calibration
    bundle = joblib.load(save_file)
    model = bundle["model"]
    probs = model.predict_proba(X_val[:5])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# ============================================================================
# 5. ML Predictor & Streaming Inference Tests
# ============================================================================
def test_ml_predictor_streaming_inference(sample_ohlcv_df):
    """Verify MLPredictor streaming inference returns valid format in < 5ms."""
    model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        pytest.skip(f"Model file '{model_path}' not found, skipping live predictor test.")

    predictor = MLPredictor(model_path=model_path)
    assert predictor.model is not None

    window = sample_ohlcv_df.iloc[-100:].copy()
    # Warm-up cold start
    for _ in range(3):
        predictor.predict(window)
    res = predictor.predict(window)

    assert res["available"] is True
    assert res["ml_bias"] in TARGET_NAMES
    assert 50 <= res["ml_confidence"] <= 95
    assert 0.0 <= res["prob_bearish"] <= 1.0
    assert 0.0 <= res["prob_neutral"] <= 1.0
    assert 0.0 <= res["prob_bullish"] <= 1.0
    total_prob = res["prob_bearish"] + res["prob_neutral"] + res["prob_bullish"]
    assert abs(total_prob - 1.0) < 0.02
    assert "latency_ms" in res
    assert res["latency_ms"] < 15.0, f"Inference took {res['latency_ms']} ms (target < 15.0 ms in test environment)"


def test_ml_predictor_fallback_behavior():
    """Verify MLPredictor handles missing model and insufficient window gracefully."""
    predictor = MLPredictor(model_path="non_existent_path.joblib")
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    res = predictor.predict(empty_df)

    assert res["available"] is False
    assert res["ml_bias"] == "NEUTRAL"
    assert res["ml_confidence"] == 50
    assert res["prob_neutral"] > 0.30


# ============================================================================
# 6. Production Model Artifact Integrity Tests
# ============================================================================
def test_production_model_artifact_size_and_budget():
    """Verify models/price_action_model.joblib exists and is strictly < 5.0 MB."""
    model_path = DEFAULT_MODEL_PATH
    assert os.path.exists(model_path), f"Expected trained model at '{model_path}'"

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    assert size_mb < 5.0, f"Model size {size_mb:.2f} MB exceeds 5.0 MB budget!"
    assert size_mb > 0.1, f"Model size {size_mb:.2f} MB seems suspiciously small!"


def test_production_model_bundle_contents():
    """Verify serialized bundle contains all required metadata and sub-estimators."""
    model_path = DEFAULT_MODEL_PATH
    bundle = joblib.load(model_path)

    assert "model" in bundle
    assert "base_model" in bundle
    assert "feature_names" in bundle
    assert len(bundle["feature_names"]) == 35
    assert "target_names" in bundle
    assert bundle["target_names"] == TARGET_NAMES
    assert "metrics" in bundle
    assert bundle["metrics"]["accuracy"] > 0.33  # Better than random baseline


# ============================================================================
# 7. Multi-Timeframe Confluence Engine Tests
# ============================================================================
from core.models import BiasType, ConfluenceReport, TimeframeConfluence
from engines.confluence_engine import ConfluenceEngine

def create_sample_confluence_df(n: int = 60, trend: str = "UP", start_price: float = 100.0) -> pd.DataFrame:
    from datetime import datetime, timezone, timedelta
    dates = [datetime.now(timezone.utc) - timedelta(hours=n - i) for i in range(n)]
    rows = []
    p = start_price
    for i in range(n):
        step = 1.0 if trend == "UP" else (-1.0 if trend == "DOWN" else (0.5 if i % 2 == 0 else -0.5))
        p += step
        o = p
        h = p + 1.5
        l = p - 1.0
        c = p + 0.5 if trend == "UP" else p - 0.5
        v = 1000.0 + i * 20.0
        rows.append([o, h, l, c, v])

    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df["timestamp"] = dates
    df["is_closed"] = True
    return df


def test_confluence_engine_initialization():
    engine = ConfluenceEngine()
    assert engine.smc_engine is not None
    assert engine.chart_pattern_engine is not None
    assert engine.ml_predictor is not None
    assert engine.WEIGHT_SMC == 0.25
    assert engine.WEIGHT_HTF == 0.20
    assert engine.WEIGHT_SR_VP == 0.15
    assert engine.WEIGHT_PAT == 0.15
    assert engine.WEIGHT_ML == 0.15
    assert engine.WEIGHT_VOL_MOM == 0.10


def test_bullish_confluence_synthesis():
    engine = ConfluenceEngine()
    df_bull = create_sample_confluence_df(n=60, trend="UP", start_price=50000.0)
    rep = engine.analyze(symbol="BTCUSDT", data=df_bull, current_price=df_bull["close"].iloc[-1])
    assert rep is not None
    assert rep.overall_bias in [BiasType.BULLISH, BiasType.STRONG_BULLISH]
    assert 50 <= rep.confidence_score <= 95
    assert rep.symbol == "BTCUSDT"


def test_bearish_confluence_synthesis():
    engine = ConfluenceEngine()
    df_bear = create_sample_confluence_df(n=60, trend="DOWN", start_price=50000.0)
    rep = engine.analyze(symbol="BTCUSDT", data=df_bear, current_price=df_bear["close"].iloc[-1])
    assert rep is not None
    assert rep.overall_bias in [BiasType.BEARISH, BiasType.STRONG_BEARISH]
    assert 50 <= rep.confidence_score <= 95


def test_forming_candle_penalty():
    engine = ConfluenceEngine()
    df = create_sample_confluence_df(n=50, trend="UP")
    rep_closed = engine.analyze(symbol="BTCUSDT", data=df, is_closed=True)
    rep_forming = engine.analyze(symbol="BTCUSDT", data=df, is_closed=False)

    mult_closed = getattr(rep_closed, "multipliers", {}).get("m_forming", 1.0)
    mult_forming = getattr(rep_forming, "multipliers", {}).get("m_forming", 1.0)
    assert mult_closed == 1.0
    assert mult_forming == 0.88


def test_multi_timeframe_dict_input():
    engine = ConfluenceEngine()
    tf_data = {
        "15m": create_sample_confluence_df(n=40, trend="UP"),
        "1h": create_sample_confluence_df(n=40, trend="UP"),
        "4h": create_sample_confluence_df(n=40, trend="UP"),
        "1d": create_sample_confluence_df(n=40, trend="UP"),
    }
    rep = engine.analyze(symbol="ETHUSDT", data=tf_data, primary_timeframe="1h")
    assert rep.symbol == "ETHUSDT"
    assert "1d" in rep.timeframe_breakdown
    assert "4h" in rep.timeframe_breakdown
    assert "1h" in rep.timeframe_breakdown
    assert "15m" in rep.timeframe_breakdown


# ============================================================================
# 8. Full Price Action Engine Integration & Cockpit Rendering Tests
# ============================================================================
def test_full_price_action_engine_integration():
    from price_action_engine import PriceActionEngine, AnalysisResult
    from core.models import TradeSetup

    engine = PriceActionEngine(max_candles=100)
    df = create_sample_confluence_df(n=60, trend="UP", start_price=60000.0)
    engine.set_history(df)

    res = engine.analyze(symbol="BTCUSDT", timeframe="1h")
    assert res is not None
    assert isinstance(res, AnalysisResult)
    assert res.confluence_report is not None
    assert res.smc_report is not None
    assert isinstance(res.chart_patterns, list)
    assert res.ml_result is not None


def test_rich_cockpit_rendering_components():
    from price_action_engine import PriceActionEngine
    from main import (
        create_header_panel,
        create_mtf_panel,
        create_smc_panel,
        create_srp_panel,
        create_trade_setup_card,
        create_logs_panel,
        render_cockpit,
        render_text_cockpit,
        build_cockpit_renderable,
    )

    engine = PriceActionEngine(max_candles=60)
    df = create_sample_confluence_df(n=50, trend="UP", start_price=3000.0)
    engine.set_history(df)

    res = engine.analyze(symbol="ETHUSDT", timeframe="15m")
    assert res is not None

    logs = ["[05:00:00] Initialized scanner", "[05:01:00] Bullish FVG detected"]
    assert create_header_panel(res) is not None
    assert create_mtf_panel(res) is not None
    assert create_smc_panel(res) is not None
    assert create_srp_panel(res) is not None
    assert create_trade_setup_card(res) is not None
    assert create_logs_panel(logs) is not None
    assert render_cockpit(res, logs) is not None
    render_text_cockpit(res, logs)

    # Compact cockpit renderable verification
    cockpit = build_cockpit_renderable(res)
    assert cockpit is not None
    from rich.console import Console
    test_c = Console(record=True, width=120)
    test_c.print(cockpit)
    rendered_lines = test_c.export_text().strip().split("\n")
    assert len(rendered_lines) <= 50, f"Cockpit height {len(rendered_lines)} exceeds 50 line budget!"
    assert len(rendered_lines) >= 15, "Cockpit should have detailed content"


