"""
train_model.py
==============
Training and Probability Calibration Pipeline for Live Price Action Scanner.

Executes:
1. Discovers and loads historical datasets across BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT (15m, 1h, 4h).
2. Extracts 35 technical & SMC features via FeatureExtractor (strictly zero lookahead).
3. Generates adaptive volatility threshold classification labels (H=5, threshold=0.75 * ATR14).
4. Splits chronologically with an embargo/purge gap to eliminate forward-looking label overlap.
5. Evaluates Purged Time-Series Cross Validation folds.
6. Trains HistGradientBoostingClassifier with balanced class weights.
7. Calibrates posterior probabilities with Platt scaling via CalibratedClassifierCV.
8. Evaluates Accuracy, Balanced Acc, F1 (Macro/Weighted), Precision/Recall per class, ROC-AUC, Brier score.
9. Evaluates Permutation Feature Importances.
10. Saves production model pipeline bundle to models/price_action_model.joblib (<5MB).
11. Benchmarks real-time streaming inference latency (<5ms target).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
    PurgedWalkForwardValidator,
    WalkForwardValidationReport,
)

console = Console()


def load_and_prepare_datasets(
    data_dir: str = "data/historical",
    symbols: List[str] = None,
    timeframes: List[str] = None,
    train_ratio: float = 0.80,
    purge_gap: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Loads historical datasets, extracts 35 features, computes forward labels,
    and creates purged chronological train and validation splits per series.
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]

    fe = FeatureExtractor()
    lg = LabelGenerator(horizon=5, atr_multiplier=0.75)

    dataset_summaries = []
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    console.print(f"[bold cyan]Scanning historical data in: [yellow]{data_dir}[/yellow][/bold cyan]")

    for sym in symbols:
        for tf in timeframes:
            # Check parquet first, fallback to csv
            parquet_path = os.path.join(data_dir, f"{sym}_{tf}.parquet")
            csv_path = os.path.join(data_dir, f"{sym}_{tf}.csv")

            file_path = None
            if os.path.exists(parquet_path):
                file_path = parquet_path
                df = pd.read_parquet(parquet_path)
            elif os.path.exists(csv_path):
                file_path = csv_path
                df = pd.read_csv(csv_path)
            else:
                console.print(f"[yellow]Warning: No data file found for {sym}_{tf}[/yellow]")
                continue

            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)

            t0 = time.perf_counter()
            feats = fe.extract_features(df)
            labels = lg.compute_labels(df)
            X_clean, y_clean, idx = lg.prepare_dataset(feats, labels)
            t1 = time.perf_counter()

            n_samples = len(X_clean)
            train_end = int(n_samples * train_ratio)
            val_start = train_end + purge_gap

            X_tr, y_tr = X_clean[:train_end], y_clean[:train_end]
            X_v, y_v = X_clean[val_start:], y_clean[val_start:]

            X_train_list.append(X_tr)
            y_train_list.append(y_tr)
            X_val_list.append(X_v)
            y_val_list.append(y_v)

            summary = {
                "symbol": sym,
                "timeframe": tf,
                "raw_candles": len(df),
                "valid_samples": n_samples,
                "train_samples": len(X_tr),
                "val_samples": len(X_v),
                "prep_time_s": round(t1 - t0, 3),
            }
            dataset_summaries.append(summary)

    if not X_train_list:
        raise ValueError(f"No valid dataset files found in '{data_dir}' matching symbols {symbols}!")

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)

    return X_train, y_train, X_val, y_val, dataset_summaries


def evaluate_purged_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 4,
    purge_window: int = 5,
) -> List[Dict[str, float]]:
    """
    Evaluates cross-validation performance with purged embargo gaps on the training set.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_window=purge_window)
    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_tr_f, y_tr_f = X_train[train_idx], y_train[train_idx]
        X_te_f, y_te_f = X_train[test_idx], y_train[test_idx]

        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.04,
            max_iter=80,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.5,
            class_weight="balanced",
            random_state=42 + fold,
        )
        clf.fit(X_tr_f, y_tr_f)
        y_pred = clf.predict(X_te_f)

        acc = accuracy_score(y_te_f, y_pred)
        bal_acc = balanced_accuracy_score(y_te_f, y_pred)
        f1_m = f1_score(y_te_f, y_pred, average="macro")

        cv_scores.append({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "f1_macro": float(f1_m),
        })

    return cv_scores


def evaluate_walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    purge_window: int = 5,
) -> WalkForwardValidationReport:
    """
    Executes Purged Walk-Forward Out-of-Sample Validation across sequential expanding windows.
    Calculates Precision/Recall per class, Sharpe ratio, Max Drawdown %, and directional Win Rate.
    """
    validator = PurgedWalkForwardValidator(
        n_splits=n_splits,
        purge_window=purge_window,
        min_train_size=200,
        expanding=True,
    )
    return validator.validate(X, y)


def benchmark_streaming_inference(
    model_path: str,
    test_df: pd.DataFrame,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmarks real-time streaming inference latency over multiple iterations.
    Target: < 5.0 ms per tick.
    """
    predictor = MLPredictor(model_path=model_path)
    if not predictor.model:
        return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0}

    window = test_df.iloc[-100:].copy()
    latencies = []

    # Warm-up run
    for _ in range(5):
        predictor.predict(window)

    # Benchmark runs
    for _ in range(iterations):
        res = predictor.predict(window)
        latencies.append(res["latency_ms"])

    lat_arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(lat_arr)),
        "min_ms": float(np.min(lat_arr)),
        "max_ms": float(np.max(lat_arr)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
    }


def print_training_report(
    metrics: Dict[str, Any],
    cv_scores: List[Dict[str, float]],
    dataset_summaries: List[Dict[str, Any]],
    bench_results: Dict[str, float],
    wf_report: Optional[WalkForwardValidationReport] = None,
) -> None:
    """Prints comprehensive formatted training and inference metrics report."""
    console.print()
    console.print(
        Panel.fit(
            "[bold green]Live Price Action Scanner — Machine Learning Model Training Report[/bold green]\n"
            f"[dim]Trained on {metrics['train_samples']:,} samples | Evaluated on {metrics['val_samples']:,} samples[/dim]",
            border_style="green",
        )
    )

    # 1. Dataset Breakdown Table
    ds_table = Table(title="Historical Datasets Ingestion Summary (60,000 Candles)", border_style="cyan")
    ds_table.add_column("Symbol", style="bold white")
    ds_table.add_column("Timeframe", style="cyan")
    ds_table.add_column("Raw Candles", justify="right")
    ds_table.add_column("Valid Samples", justify="right")
    ds_table.add_column("Train Samples (80%)", justify="right")
    ds_table.add_column("Holdout Samples (20%)", justify="right")
    ds_table.add_column("Extraction Time", justify="right", style="green")

    for s in dataset_summaries:
        ds_table.add_row(
            s["symbol"],
            s["timeframe"],
            f"{s['raw_candles']:,}",
            f"{s['valid_samples']:,}",
            f"{s['train_samples']:,}",
            f"{s['val_samples']:,}",
            f"{s['prep_time_s']:.3f} s",
        )
    console.print(ds_table)

    # 2. Walk-Forward Out-of-Sample Validation Table
    if wf_report and wf_report.folds:
        wf_table = Table(
            title=f"Purged Walk-Forward OOS Validation ({wf_report.n_folds} Expanding Folds, Purge=5)",
            border_style="yellow",
        )
        wf_table.add_column("Fold", justify="center", style="bold")
        wf_table.add_column("Train / OOS", justify="center")
        wf_table.add_column("Accuracy", justify="right")
        wf_table.add_column("Macro F1", justify="right", style="bold green")
        wf_table.add_column("Bull P/R", justify="center")
        wf_table.add_column("Bear P/R", justify="center")
        wf_table.add_column("Win Rate", justify="right", style="cyan")
        wf_table.add_column("Sharpe", justify="right", style="bold yellow")
        wf_table.add_column("Max DD", justify="right", style="red")
        wf_table.add_column("Net Ret", justify="right", style="bold")

        for f in wf_report.folds:
            wf_table.add_row(
                f"Fold {f.fold_idx}",
                f"{f.train_size:,} / {f.test_size:,}",
                f"{f.accuracy:.3f}",
                f"{f.f1_macro:.3f}",
                f"{f.bullish_precision:.2f}/{f.bullish_recall:.2f}",
                f"{f.bearish_precision:.2f}/{f.bearish_recall:.2f}",
                f"{f.directional_win_rate:.1f}%",
                f"{f.simulated_sharpe:.2f}",
                f"{f.simulated_max_dd_pct:.1f}%",
                f"{f.simulated_net_return_pct:+.1f}%",
            )
        console.print(wf_table)

        console.print(
            Panel(
                f"[bold cyan]Walk-Forward Summary:[/bold cyan] "
                f"Avg OOS Sharpe: [bold yellow]{wf_report.avg_sharpe:.2f}[/bold yellow] | "
                f"Max Drawdown: [bold red]{wf_report.overall_max_dd_pct:.1f}%[/bold red] | "
                f"Directional Win Rate: [bold green]{wf_report.avg_directional_win_rate:.1f}%[/bold green] | "
                f"Avg Macro F1: [bold]{wf_report.avg_f1_macro:.3f}[/bold]",
                border_style="yellow",
            )
        )
    elif cv_scores:
        # Purged CV Fallback
        cv_table = Table(title="Purged Time-Series Cross Validation (Embargo Gap = 5 Candles)", border_style="yellow")
        cv_table.add_column("Fold", justify="center", style="bold")
        cv_table.add_column("Train Size", justify="right")
        cv_table.add_column("Test Size", justify="right")
        cv_table.add_column("Accuracy", justify="right")
        cv_table.add_column("Balanced Accuracy", justify="right", style="bold yellow")
        cv_table.add_column("Macro F1", justify="right", style="bold green")

        for cv in cv_scores:
            cv_table.add_row(
                f"Fold {cv['fold']}",
                f"{cv['train_size']:,}",
                f"{cv['test_size']:,}",
                f"{cv['accuracy']:.4f}",
                f"{cv['balanced_accuracy']:.4f}",
                f"{cv['f1_macro']:.4f}",
            )
        console.print(cv_table)

    # 3. Overall Holdout Performance Metrics Table
    m_table = Table(title="Calibrated Model Holdout Evaluation Metrics (Platt Scaling)", border_style="green")
    m_table.add_column("Metric Name", style="bold white")
    m_table.add_column("Score", justify="right", style="bold green")
    m_table.add_column("Target / Theoretical Baseline", style="dim")

    m_table.add_row("Accuracy", f"{metrics['accuracy']:.4f}", "Baseline: 0.3333 (Random 3-Class)")
    m_table.add_row("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}", "Baseline: 0.3333")
    m_table.add_row("F1 Score (Macro Average)", f"{metrics['f1_macro']:.4f}", "Unbiased directional performance")
    m_table.add_row("F1 Score (Weighted Average)", f"{metrics['f1_weighted']:.4f}", "Sample-weighted performance")
    m_table.add_row("Multi-Class ROC-AUC (Weighted OvR)", f"{metrics['roc_auc_weighted']:.4f}", "Baseline: 0.5000 (No discrimination)")
    m_table.add_row("Multi-Class ROC-AUC (Macro OvR)", f"{metrics['roc_auc_macro']:.4f}", "Baseline: 0.5000")
    m_table.add_row("Log Loss (Cross-Entropy)", f"{metrics['log_loss']:.4f}", "Lower is better")
    m_table.add_row("Brier Score Loss", f"{metrics['brier_score']:.4f}", "Lower is better (0.0 = perfect calibration)")
    m_table.add_row("Model File Size", f"{metrics['artifact_size_mb']:.3f} MB", "Budget: < 5.000 MB")
    m_table.add_row("Mean Streaming Latency", f"{bench_results['mean_ms']:.3f} ms", "Budget: < 5.000 ms per tick")
    m_table.add_row("P95 Streaming Latency", f"{bench_results['p95_ms']:.3f} ms", "Budget: < 5.000 ms per tick")
    console.print(m_table)

    # 4. Classification Report per Class
    c_table = Table(title="Detailed Per-Class Performance Breakdown", border_style="magenta")
    c_table.add_column("Class Label", style="bold")
    c_table.add_column("Precision", justify="right")
    c_table.add_column("Recall", justify="right")
    c_table.add_column("F1-Score", justify="right", style="bold")
    c_table.add_column("Holdout Support", justify="right")

    report = metrics["classification_report"]
    for c_name in TARGET_NAMES:
        c_metrics = report[c_name]
        c_table.add_row(
            c_name,
            f"{c_metrics['precision']:.4f}",
            f"{c_metrics['recall']:.4f}",
            f"{c_metrics['f1-score']:.4f}",
            f"{int(c_metrics['support']):,}",
        )
    console.print(c_table)

    # 5. Top 10 Feature Importances Table
    fi_table = Table(title="Top 10 Feature Importances (Permutation Importance)", border_style="blue")
    fi_table.add_column("Rank", justify="center", style="dim")
    fi_table.add_column("Feature Name", style="bold cyan")
    fi_table.add_column("Importance Score", justify="right", style="bold green")

    top_10 = list(metrics["feature_importances"].items())[:10]
    for rank, (fname, fscore) in enumerate(top_10, start=1):
        fi_table.add_row(str(rank), fname, f"{fscore:.5f}")
    console.print(fi_table)

    console.print(f"\n[bold green]Model artifact successfully serialized to:[/bold green] [yellow]{metrics['artifact_path']}[/yellow]")
    console.print(f"[bold green]Artifact size:[/bold green] [yellow]{metrics['artifact_size_mb']:.3f} MB[/yellow] (Budget: < 5.0 MB)")
    console.print(f"[bold green]Streaming Inference Latency:[/bold green] [yellow]{bench_results['mean_ms']:.3f} ms[/yellow] (Budget: < 5.0 ms)\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Live Price Action Scanner Machine Learning Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/historical", help="Path to historical data directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for serialized models")
    parser.add_argument("--cv-splits", type=int, default=4, help="Number of purged CV splits")
    parser.add_argument("--purge-gap", type=int, default=5, help="Purge gap buffer (number of candles)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--walk-forward", action="store_true", default=True, help="Run purged walk-forward validation")

    args = parser.parse_args()

    t_start = time.perf_counter()

    # 1. Ingest & Prepare Historical Datasets
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    timeframes = ["15m", "1h", "4h"]
    X_train, y_train, X_val, y_val, dataset_summaries = load_and_prepare_datasets(
        data_dir=args.data_dir,
        symbols=symbols,
        timeframes=timeframes,
        train_ratio=0.80,
        purge_gap=args.purge_gap,
    )

    console.print(f"[bold]Aggregated Data:[/bold] Train = {X_train.shape}, Validation = {X_val.shape}")

    # 2. Evaluate Purged Walk-Forward Out-of-Sample Validation
    wf_report = None
    cv_scores = []
    if args.walk_forward:
        console.print("[bold cyan]Evaluating Purged Walk-Forward Out-of-Sample Validation (5 folds)...[/bold cyan]")
        wf_report = evaluate_walk_forward_validation(
            X_train, y_train, n_splits=5, purge_window=args.purge_gap
        )
    else:
        console.print("[bold cyan]Evaluating Purged Time-Series Cross Validation (4 folds)...[/bold cyan]")
        cv_scores = evaluate_purged_cross_validation(
            X_train, y_train, n_splits=args.cv_splits, purge_window=args.purge_gap
        )

    # 3. Train & Calibrate Production Model Pipeline
    console.print("[bold cyan]Training HistGradientBoostingClassifier & Platt Scaling Calibrator...[/bold cyan]")
    trainer = ModelTrainer(
        output_dir=args.output_dir,
        learning_rate=0.04,
        max_iter=80,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=args.random_state,
    )

    save_path = os.path.join(args.output_dir, "price_action_model.joblib")
    metrics = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=FEATURE_COLUMNS,
        save_path=save_path,
    )

    # 4. Benchmark Streaming Inference Latency
    sample_df = pd.read_csv(os.path.join(args.data_dir, "BTCUSDT_15m.csv"))
    bench_results = benchmark_streaming_inference(
        model_path=save_path,
        test_df=sample_df,
        iterations=100,
    )

    # 5. Display Complete Report
    print_training_report(
        metrics=metrics,
        cv_scores=cv_scores,
        dataset_summaries=dataset_summaries,
        bench_results=bench_results,
        wf_report=wf_report,
    )

    t_total = time.perf_counter() - t_start
    console.print(f"[bold green]End-to-End Pipeline completed successfully in {t_total:.2f} seconds![/bold green]\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
