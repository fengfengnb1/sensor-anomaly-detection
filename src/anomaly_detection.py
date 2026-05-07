"""
anomaly_detection.py
--------------------
Three-tier anomaly detection:
    Tier 1 – Statistical  : ZScoreDetector
    Tier 2 – ML           : IsolationForestDetector
    Tier 3 – Deep learning: LSTMAutoencoderDetector
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score


# ── Base class ───────────────────────────────────────────────────────────────

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


# ── Tier 1: Z-score ──────────────────────────────────────────────────────────

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


# ── Tier 2: Isolation Forest ─────────────────────────────────────────────────

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


# ── Tier 3: LSTM Autoencoder ─────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LSTMAutoencoder(nn.Module if TORCH_AVAILABLE else object):
    """
    LSTM-based autoencoder for time-series anomaly detection.

    Architecture:
        Encoder: LSTM → hidden state
        Decoder: LSTM → reconstruct input sequence

    Anomaly score = mean squared reconstruction error per timestep.
    Timesteps with reconstruction error > threshold are flagged as anomalous.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMAutoencoderDetector. Run: pip install torch")
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, input_size)
        _, (h, c) = self.encoder(x)
        # Repeat hidden state across sequence length for decoder input
        h_last = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        recon, _ = self.decoder(h_last)
        return recon


class LSTMAutoencoderDetector(BaseDetector):
    """
    Trains an LSTM Autoencoder on normal data and flags timesteps where
    reconstruction error exceeds a learned threshold.

    Particularly effective for detecting anomalies in temporal patterns
    that are invisible to point-wise methods (e.g. unusual oscillation
    frequencies, drift patterns, phase shifts).
    """

    name = "lstm_autoencoder"

    def __init__(
        self,
        seq_len: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        threshold_percentile: float = 97.0,
        random_state: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Run: pip install torch")
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self._threshold: float = 0.0
        self._model: LSTMAutoencoder | None = None
        torch.manual_seed(random_state)

    def _make_sequences(self, X: np.ndarray) -> "torch.Tensor":
        sequences = []
        for i in range(len(X) - self.seq_len + 1):
            sequences.append(X[i : i + self.seq_len])
        return torch.tensor(np.array(sequences), dtype=torch.float32)

    def fit(self, X: np.ndarray) -> "LSTMAutoencoderDetector":
        import torch.optim as optim

        n_features = X.shape[1]
        self._model = LSTMAutoencoder(n_features, self.hidden_size, self.num_layers)
        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        sequences = self._make_sequences(X)
        dataset = TensorDataset(sequences)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        print(f"\nTraining LSTM Autoencoder ({self.epochs} epochs)...")
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self._model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.epochs} | loss: {total_loss / len(loader):.6f}")

        # Compute reconstruction errors on training data to set threshold
        errors = self._reconstruction_errors(X)
        self._threshold = float(np.percentile(errors, self.threshold_percentile))
        print(f"  Anomaly threshold set at {self._threshold:.6f} "
              f"({self.threshold_percentile}th percentile)\n")
        return self

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Return per-timestep mean squared reconstruction error."""
        self._model.eval()
        sequences = self._make_sequences(X)
        errors = np.zeros(len(X))
        counts = np.zeros(len(X))

        with torch.no_grad():
            for i in range(len(sequences)):
                seq = sequences[i].unsqueeze(0)
                recon = self._model(seq).squeeze(0).numpy()
                orig  = sequences[i].numpy()
                err   = np.mean((recon - orig) ** 2, axis=1)
                for j, e in enumerate(err):
                    errors[i + j] += e
                    counts[i + j] += 1

        counts = np.where(counts == 0, 1, counts)
        return errors / counts

    def predict(self, X: np.ndarray) -> np.ndarray:
        errors = self._reconstruction_errors(X)
        return (errors > self._threshold).astype(int)

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        return self._reconstruction_errors(X)
