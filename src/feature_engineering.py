"""
feature_engineering.py
-----------------------
Extracts time-series features used as input for anomaly detectors.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    windows: list[int] = (5, 20, 60),
) -> pd.DataFrame:
    """
    Add rolling mean, std, min, max, and range for each sensor channel.
    These capture short- and long-term statistical context — essential for
    detecting contextual and collective anomalies.

    Parameters
    ----------
    df          : input DataFrame with DatetimeIndex
    sensor_cols : list of column names to process
    windows     : rolling window sizes in number of timesteps

    Returns
    -------
    DataFrame with original columns plus rolling feature columns
    """
    result = df.copy()

    for col in sensor_cols:
        for w in windows:
            rolled = result[col].rolling(window=w, min_periods=1)
            result[f"{col}_roll_mean_{w}"] = rolled.mean()
            result[f"{col}_roll_std_{w}"]  = rolled.std().fillna(0)
            result[f"{col}_roll_min_{w}"]  = rolled.min()
            result[f"{col}_roll_max_{w}"]  = rolled.max()
            result[f"{col}_roll_range_{w}"] = (
                result[f"{col}_roll_max_{w}"] - result[f"{col}_roll_min_{w}"]
            )

    return result


def add_lag_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    lags: list[int] = (1, 5, 10),
) -> pd.DataFrame:
    """
    Add lagged values and first-order differences for each sensor channel.
    Differences expose sudden rate-of-change anomalies.
    """
    result = df.copy()

    for col in sensor_cols:
        for lag in lags:
            result[f"{col}_lag_{lag}"] = result[col].shift(lag)
        result[f"{col}_diff_1"] = result[col].diff(1)

    return result.fillna(method="bfill")


def add_zscore_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling Z-score for each sensor channel.
    Z = (x - rolling_mean) / rolling_std
    Values with |Z| > 3 are statistical outliers.
    """
    result = df.copy()

    for col in sensor_cols:
        roll_mean = result[col].rolling(window=window, min_periods=1).mean()
        roll_std  = result[col].rolling(window=window, min_periods=1).std().replace(0, np.nan)
        result[f"{col}_zscore"] = ((result[col] - roll_mean) / roll_std).fillna(0)

    return result


def build_feature_matrix(
    df: pd.DataFrame,
    sensor_cols: list[str],
    normalize: bool = True,
) -> tuple[np.ndarray, StandardScaler | None]:
    """
    Build and optionally normalize the feature matrix for ML models.

    Returns
    -------
    X         : (n_samples, n_features) numpy array
    scaler    : fitted StandardScaler, or None if normalize=False
    """
    feature_cols = [
        c for c in df.columns
        if c not in sensor_cols + ["anomaly_label"]
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    if not feature_cols:
        feature_cols = sensor_cols

    X = df[feature_cols].values

    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, scaler, feature_cols
