"""
Three-tier anomaly detection:
    Tier 1 – Statistical  : ZScoreDetector
    Tier 2 – ML           : IsolationForestDetector
    Tier 3 – Deep learning: LSTMAutoencoderDetector
#Tier 3 ignored
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score



class BaseDetector:
    """All detectors follow this interface."""

    name: str = "base"

    def fit(self, X: np.ndarray) -> "BaseDetector":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels: 1 = anomaly, 0 = normal."""
        raise NotImplementedError

    def score(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = self.predict(X)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return {
            "detector": self.name,
            "precision": round(report["1"]["precision"], 4),
            "recall":    round(report["1"]["recall"], 4),
            "f1":        round(report["1"]["f1-score"], 4),
        }


#  Tier 1: Z-score

class ZScoreDetector(BaseDetector):
    """
    Flags timesteps where ANY sensor channel exceeds `threshold` standard
    deviations from its rolling mean.

    Simple, fast, and interpretable. Works well for sudden point anomalies.
    No training required.
    """

    name = "zscore"

    def __init__(self, threshold: float = 3.0, window: int = 60):
        self.threshold = threshold
        self.window = window
        self._zscore_cols: list[str] = []

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        # Z-score method is non-parametric; nothing to fit.
        return self

    def fit_on_dataframe(self, df: pd.DataFrame, sensor_cols: list[str]) -> "ZScoreDetector":
        """Attach zscore columns from a processed DataFrame."""
        self._zscore_cols = [f"{c}_zscore" for c in sensor_cols if f"{c}_zscore" in df.columns]
        self._df = df
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("Use predict_from_dataframe() for ZScoreDetector.")

    def predict_from_dataframe(self, df: pd.DataFrame, sensor_cols: list[str]) -> np.ndarray:
        """
        Predict anomalies directly from DataFrame with zscore feature columns.
        """
        zscore_cols = [f"{c}_zscore" for c in sensor_cols if f"{c}_zscore" in df.columns]
        if not zscore_cols:
            raise ValueError("No zscore columns found. Run add_zscore_features() first.")
        max_zscore = df[zscore_cols].abs().max(axis=1)
        return (max_zscore > self.threshold).astype(int).values


#  Tier 2: Isolation Forest 

class IsolationForestDetector(BaseDetector):
    """
    Scikit-learn IsolationForest for unsupervised anomaly detection.

    Isolation Forest works by randomly partitioning the feature space.
    Anomalous points, being sparse and structurally different, require
    fewer splits to isolate — they have shorter average path lengths.

    No labels required. Works well on multi-dimensional sensor data.
    """

    name = "isolation_forest"

    def __init__(
        self,
        contamination: float = 0.03,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
# IsolationForest原理：异常点在特征空间里比较孤立，
# 随机切割时更容易被"隔离"，所以平均路径长度更短
# contamination=0.03 是因为合成数据里注入了约3%的异常
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # IsolationForest returns -1 for anomalies, +1 for normal
        raw = self._model.predict(X)
        return ((raw == -1).astype(int))

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores (lower = more anomalous).
        Useful for threshold tuning and visualization.
        """
        return -self._model.score_samples(X)
