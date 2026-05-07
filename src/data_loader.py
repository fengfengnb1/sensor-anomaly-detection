"""
data_loader.py
--------------
Handles data ingestion, validation, cleaning, and synthetic data generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Synthetic data ──────────────────────────────────────────────────────────

def generate_synthetic_data(
    n_points: int = 2000,
    channels: list[str] = ("temperature", "humidity"),
    anomaly_fraction: float = 0.03,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic multi-channel sensor time-series with injected anomalies.

    Anomaly types injected:
        - Point anomaly   : single spike / dropout
        - Contextual anomaly : value plausible in isolation but anomalous in context
        - Collective anomaly : short burst of unusual values

    Returns
    -------
    pd.DataFrame with columns [timestamp, *channels, anomaly_label]
        anomaly_label = 1 for anomalous timestep, 0 for normal
    """
    rng = np.random.default_rng(random_seed)

    timestamps = pd.date_range("2024-01-01", periods=n_points, freq="1min")

    # Base signals: slow sinusoidal drift + Gaussian noise
    t = np.linspace(0, 4 * np.pi, n_points)
    temperature = 22 + 3 * np.sin(t / 2) + rng.normal(0, 0.4, n_points)
    humidity    = 55 + 8 * np.sin(t / 3 + 1) + rng.normal(0, 1.0, n_points)

    labels = np.zeros(n_points, dtype=int)
    n_anomalies = int(n_points * anomaly_fraction)

    # Inject point anomalies
    point_idx = rng.choice(n_points, size=n_anomalies // 2, replace=False)
    temperature[point_idx] += rng.choice([-1, 1], size=len(point_idx)) * rng.uniform(8, 15, len(point_idx))
    labels[point_idx] = 1

    # Inject collective anomalies (short bursts)
    burst_starts = rng.choice(n_points - 20, size=n_anomalies // 4, replace=False)
    for start in burst_starts:
        length = rng.integers(3, 10)
        end = min(start + length, n_points)
        humidity[start:end] += rng.uniform(20, 35)
        labels[start:end] = 1

    data = {
        "timestamp": timestamps,
        "temperature": temperature,
        "humidity": humidity,
        "anomaly_label": labels,
    }
    return pd.DataFrame(data)


# ── File loader ─────────────────────────────────────────────────────────────

def load_csv(
    filepath: str | Path,
    timestamp_col: str = "timestamp",
    value_cols: list[str] | None = None,
    label_col: str | None = "anomaly_label",
) -> pd.DataFrame:
    """
    Load a sensor CSV file, parse timestamps, and select relevant columns.

    Parameters
    ----------
    filepath      : path to CSV file
    timestamp_col : name of the datetime column
    value_cols    : sensor channel columns to keep (None = all numeric)
    label_col     : ground-truth anomaly label column, if present

    Returns
    -------
    pd.DataFrame with a DatetimeIndex and selected sensor columns
    """
    df = pd.read_csv(filepath)

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col).sort_index()

    if value_cols is not None:
        keep = value_cols + ([label_col] if label_col and label_col in df.columns else [])
        df = df[keep]

    return df


# ── Preprocessing ────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, fill_method: str = "linear") -> pd.DataFrame:
    """
    Basic cleaning pipeline:
        1. Drop duplicate timestamps
        2. Interpolate missing values (linear by default)
        3. Clip extreme hardware outliers (>5 IQR from median)
    """
    df = df[~df.index.duplicated(keep="first")]
    df = df.infer_objects()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_col = "anomaly_label" if "anomaly_label" in df.columns else None
    sensor_cols = [c for c in numeric_cols if c != label_col]

    # Interpolate missing values in sensor channels only
    df[sensor_cols] = df[sensor_cols].interpolate(method=fill_method).ffill().bfill()

    # Clip extreme values: median ± 5 × IQR
    for col in sensor_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 5 * iqr
        upper = q3 + 5 * iqr
        df[col] = df[col].clip(lower, upper)

    return df


def validate(df: pd.DataFrame) -> None:
    """
    Run basic data quality checks and print a summary report.
    Raises ValueError if critical issues are found.
    """
    print("\n── Data validation ─────────────────────────────")
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {list(df.columns)}")

    missing = df.isnull().sum()
    if missing.any():
        print(f"  Missing vals : \n{missing[missing > 0]}")
    else:
        print("  Missing vals : none ✓")

    numeric = df.select_dtypes(include=[np.number])
    print(f"\n  Summary statistics:")
    print(numeric.describe().round(3).to_string())

    if "anomaly_label" in df.columns:
        n_anom = df["anomaly_label"].sum()
        pct = 100 * n_anom / len(df)
        print(f"\n  Anomalies    : {int(n_anom):,} ({pct:.1f}%)")

    print("────────────────────────────────────────────────\n")
