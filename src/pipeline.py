"""
pipeline.py
-----------
Orchestrates the full anomaly detection workflow:
    load → clean → feature engineering → detection → evaluation → visualization
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    clean,
    generate_synthetic_data,
    load_csv,
    validate,
)
from src.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_zscore_features,
    build_feature_matrix,
)
from src.anomaly_detection import (
    IsolationForestDetector,
    ZScoreDetector,
)
from src.visualization import (
    plot_detection_results,
    plot_method_comparison,
    plot_reconstruction_error,
    plot_sensor_overview,
)


SENSOR_COLS = ["temperature", "humidity"]


def run(
    mode: str = "synthetic",
    input_path: str | None = None,
    sensor_cols: list[str] = SENSOR_COLS,
    method: str = "all",
    output_dir: str = "reports/figures",
    save_figures: bool = True,
) -> dict:
    """
    Run the full pipeline.

    Parameters
    ----------
    mode        : "synthetic" or "file"
    input_path  : path to CSV file (used when mode="file")
    sensor_cols : list of sensor channel column names
    method      : "zscore" | "isolation_forest" | "lstm" | "all"
    output_dir  : directory to save output figures
    save_figures: if False, display plots interactively instead

    Returns
    -------
    dict of evaluation results per detector
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Sensor Anomaly Detection Pipeline")
    print("=" * 55)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    if mode == "synthetic":
        df = generate_synthetic_data()
        if 'timestamp' in df.columns:
             df = df.set_index('timestamp')
        print(f"  Generated synthetic data: {len(df):,} timesteps, {len(sensor_cols)} channels")
    elif mode == "file" and input_path:
        df = load_csv(input_path)
        # Make sure required sensor columns exist
        for col in sensor_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {input_path}. "
                                 f"Available: {list(df.columns)}")
        print(f"  Loaded from file: {input_path} ({len(df):,} rows)")
    else:
        raise ValueError("Specify mode='synthetic' or mode='file' with a valid input_path.")

    # ── 2. Clean and validate ─────────────────────────────────────
    print("\n[2/5] Cleaning and validating...")
    df = clean(df)
    validate(df)

    # ── 3. Feature engineering ────────────────────────────────────
    print("[3/5] Engineering features...")
    df = add_rolling_features(df, sensor_cols, windows=[5, 20, 60])
    df = add_lag_features(df, sensor_cols, lags=[1, 5, 10])
    df = add_zscore_features(df, sensor_cols, window=60)
    print(f"  Feature matrix shape: {df.shape}")

    # Separate labels if present
    y_true = df["anomaly_label"].values if "anomaly_label" in df.columns else None

    # Plot sensor overview
    plot_sensor_overview(
        df, sensor_cols,
        save_path=output_dir / "sensor_overview.png" if save_figures else None,
    )

    # ── 4. Detection ──────────────────────────────────────────────
    print("\n[4/5] Running anomaly detection...")
    X, scaler, feature_cols = build_feature_matrix(df, sensor_cols)

    predictions = {}
    results = []
    run_methods = ["zscore", "isolation_forest", "lstm"] if method == "all" else [method]

    # Tier 1: Z-score
    if "zscore" in run_methods:
        t0 = time.time()
        detector = ZScoreDetector(threshold=3.0, window=60)
        preds = detector.predict_from_dataframe(df, sensor_cols)
        elapsed = time.time() - t0
        predictions["zscore"] = preds
        if y_true is not None:
            score = detector.score(None, y_true) if False else {}  # manual
            from sklearn.metrics import classification_report
            rep = classification_report(y_true, preds, output_dict=True, zero_division=0)
            score = {
                "detector": "zscore",
                "precision": round(rep["1"]["precision"], 4),
                "recall":    round(rep["1"]["recall"], 4),
                "f1":        round(rep["1"]["f1-score"], 4),
            }
            results.append(score)
            print(f"  Z-score          → F1: {score['f1']:.4f}  ({elapsed:.2f}s)")

    # Tier 2: Isolation Forest
    if "isolation_forest" in run_methods:
        t0 = time.time()
        detector = IsolationForestDetector(contamination=0.03)
        detector.fit(X)
        preds = detector.predict(X)
        elapsed = time.time() - t0
        predictions["isolation_forest"] = preds
        if y_true is not None:
            score = detector.score(X, y_true)
            results.append(score)
            print(f"  Isolation Forest → F1: {score['f1']:.4f}  ({elapsed:.2f}s)")

 
    # ── 5. Visualize ──────────────────────────────────────────────
    print("\n[5/5] Generating visualizations...")

    plot_detection_results(
        df, predictions, sensor_cols[0],
        save_path=output_dir / "detection_comparison.png" if save_figures else None,
    )

    if results:
        plot_method_comparison(
            results,
            save_path=output_dir / "method_comparison.png" if save_figures else None,
        )

        print("\n── Final results ───────────────────────────────")
        for r in results:
            print(f"  {r['detector']:<22} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
        print("────────────────────────────────────────────────")

    print(f"\n✓ Done. Figures saved to: {output_dir.resolve()}\n")
    return {"results": results, "predictions": predictions}
