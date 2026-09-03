"""
tests/test_ml_predictor.py - Unit Tests for FeatureExtractor and MLPredictor
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from core.models import MLInferenceResult
from engines.ml_predictor import FeatureExtractor, MLPredictor


def create_ohlcv_window(n: int = 60, trend: str = "UP") -> pd.DataFrame:
    dates = [datetime.now(timezone.utc) - timedelta(hours=n - i) for i in range(n)]
    p = 100.0
    rows = []
    for i in range(n):
        p += 1.0 if trend == "UP" else -1.0
        o = p
        c = p + 0.5 if trend == "UP" else p - 0.5
        h = max(o, c) + 1.0
        l = min(o, c) - 1.0
        rows.append([o, h, l, c, 1000.0 + i * 10])
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df["timestamp"] = dates
    return df


def test_feature_extractor_35_features():
    extractor = FeatureExtractor()
    assert len(extractor.feature_columns) == 35

    df = create_ohlcv_window(n=60)
    feats = extractor.extract_features(df)

    assert len(feats) == 60
    assert len(feats.columns) == 35
    assert not feats.isna().any().any()
    assert not np.isinf(feats.values).any()


def test_ml_predictor_fallback_and_types():
    predictor = MLPredictor(model_path="non_existent_model.joblib")
    df = create_ohlcv_window(n=60, trend="UP")
    res = predictor.predict(df)

    assert isinstance(res, MLInferenceResult)
    assert 0.0 <= res.prob_bullish <= 1.0
    assert 0.0 <= res.prob_bearish <= 1.0
    assert 0.0 <= res.prob_neutral <= 1.0
    assert abs((res.prob_bullish + res.prob_bearish + res.prob_neutral) - 1.0) < 1e-3
    assert 0.0 <= res.model_confidence <= 1.0
    assert isinstance(res.feature_contributions, dict)
    assert len(res.feature_contributions) > 0


def test_ml_predictor_directional_sensitivity():
    # Test statistical inference engine directional sensitivity
    predictor = MLPredictor(model_path="non_existent_model.joblib")
    df_bull = create_ohlcv_window(n=60, trend="UP")
    df_bear = create_ohlcv_window(n=60, trend="DOWN")

    res_bull = predictor.predict(df_bull)
    res_bear = predictor.predict(df_bear)

    # Bullish data should have higher prob_bullish than prob_bearish
    assert res_bull.prob_bullish > res_bull.prob_bearish
    # Bearish data should have higher prob_bearish than prob_bullish
    assert res_bear.prob_bearish > res_bear.prob_bullish


def test_ml_predictor_with_serialized_bundle():
    predictor = MLPredictor()
    df = create_ohlcv_window(n=60, trend="UP")
    res = predictor.predict(df)

    assert isinstance(res, MLInferenceResult)
    assert 0.0 <= res.prob_bullish <= 1.0
    assert 0.0 <= res.prob_bearish <= 1.0
    assert 0.0 <= res.prob_neutral <= 1.0
    assert abs((res.prob_bullish + res.prob_bearish + res.prob_neutral) - 1.0) < 1e-3
    assert 0.0 <= res.model_confidence <= 1.0
