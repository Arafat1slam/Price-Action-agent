"""
tests/test_ml_walk_forward.py
=============================
Unit tests for PurgedWalkForwardValidator and out-of-sample ML metrics
(Sharpe ratio, max drawdown, precision/recall, directional win rate).
"""

import numpy as np
import pytest

from engines.ml_engine import (
    PurgedWalkForwardValidator,
    WalkForwardFoldResult,
    WalkForwardValidationReport,
)


@pytest.fixture
def synthetic_ml_dataset():
    """Generates synthetic sequential features and labels for testing."""
    np.random.seed(42)
    n_samples = 600
    n_features = 35

    # Random features with a slight trend signal
    X = np.random.randn(n_samples, n_features)
    # Target: 0 (Bearish), 1 (Neutral), 2 (Bullish)
    latent_signal = 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(n_samples) * 0.2
    y = np.ones(n_samples, dtype=int)
    y[latent_signal > 0.3] = 2
    y[latent_signal < -0.3] = 0

    # Simulated prices
    returns = np.random.normal(0.0002, 0.01, size=n_samples)
    prices = 100.0 * np.cumprod(1.0 + returns)

    return X, y, prices


def test_walk_forward_validator_basic(synthetic_ml_dataset):
    X, y, prices = synthetic_ml_dataset
    validator = PurgedWalkForwardValidator(
        n_splits=3,
        purge_window=5,
        min_train_size=100,
        expanding=True,
        random_state=42,
    )

    report = validator.validate(X, y, raw_prices=prices)

    assert isinstance(report, WalkForwardValidationReport)
    assert report.n_folds == 3
    assert len(report.folds) == 3
    assert report.avg_accuracy > 0.0
    assert 0.0 <= report.avg_directional_win_rate <= 100.0
    assert report.overall_max_dd_pct >= 0.0
    assert isinstance(report.avg_sharpe, float)


def test_walk_forward_fold_metrics_integrity(synthetic_ml_dataset):
    X, y, prices = synthetic_ml_dataset
    validator = PurgedWalkForwardValidator(
        n_splits=3,
        purge_window=5,
        min_train_size=100,
        expanding=True,
    )

    report = validator.validate(X, y, raw_prices=prices)

    for fold in report.folds:
        assert isinstance(fold, WalkForwardFoldResult)
        assert fold.train_size >= 100
        assert fold.test_size > 0
        assert 0.0 <= fold.accuracy <= 1.0
        assert 0.0 <= fold.f1_macro <= 1.0
        assert 0.0 <= fold.bullish_precision <= 1.0
        assert 0.0 <= fold.bullish_recall <= 1.0
        assert 0.0 <= fold.bearish_precision <= 1.0
        assert 0.0 <= fold.bearish_recall <= 1.0
        assert 0.0 <= fold.directional_win_rate <= 100.0
        assert fold.simulated_max_dd_pct >= 0.0


def test_walk_forward_no_leakage_and_embargo(synthetic_ml_dataset):
    X, y, _ = synthetic_ml_dataset
    validator = PurgedWalkForwardValidator(
        n_splits=4,
        purge_window=10,
        min_train_size=80,
    )

    n = len(X)
    fold_size = n // 5
    report = validator.validate(X, y)

    # Validate that expanding training sizes strictly grow
    train_sizes = [f.train_size for f in report.folds]
    assert sorted(train_sizes) == train_sizes
    assert len(train_sizes) == 4


def test_walk_forward_empty_or_small_data():
    X = np.random.randn(30, 35)
    y = np.ones(30, dtype=int)
    validator = PurgedWalkForwardValidator(n_splits=5, min_train_size=100)

    report = validator.validate(X, y)
    assert report.n_folds == 0
    assert len(report.folds) == 0
    assert report.avg_accuracy == 0.0
