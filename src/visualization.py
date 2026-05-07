"""
visualization.py
----------------
Plotting functions for time-series inspection, anomaly annotation,
and method comparison.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent style across all plots
plt.rcParams.update({
    "figure.dpi":       120,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "font.size":        11,
})

ANOMALY_COLOR  = "#E24B4A"
NORMAL_COLOR   = "#378ADD"
DETECTED_COLOR = "#EF9F27"


def plot_sensor_overview(
    df: pd.DataFrame,
    sensor_cols: list[str],
    save_path: str | Path | None = None,
) -> None:
    """
    Multi-panel time-series overview with ground-truth anomaly regions shaded.
    """
    n = len(sensor_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    has_labels = "anomaly_label" in df.columns

    for ax, col in zip(axes, sensor_cols):
        ax.plot(df.index, df[col], color=NORMAL_COLOR, linewidth=0.8, alpha=0.9, label=col)

        if has_labels:
            anomaly_mask = df["anomaly_label"].astype(bool)
            ax.fill_between(
                df.index, df[col].min(), df[col].max(),
                where=anomaly_mask,
                color=ANOMALY_COLOR, alpha=0.18, label="Anomaly region",
            )
            ax.scatter(
                df.index[anomaly_mask], df[col][anomaly_mask],
                color=ANOMALY_COLOR, s=8, zorder=5, alpha=0.7,
            )

        ax.set_ylabel(col, fontsize=10)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.suptitle("Sensor data overview with anomaly regions", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_detection_results(
    df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    sensor_col: str,
    save_path: str | Path | None = None,
) -> None:
    """
    Compare multiple detectors' predictions on a single sensor channel.

    Parameters
    ----------
    predictions : {detector_name: binary_label_array}
    """
    n_detectors = len(predictions)
    fig, axes = plt.subplots(n_detectors + 1, 1, figsize=(14, 3.5 * (n_detectors + 1)), sharex=True)

    # Top panel: raw signal
    ax0 = axes[0]
    ax0.plot(df.index, df[sensor_col], color=NORMAL_COLOR, linewidth=0.8, label=sensor_col)
    if "anomaly_label" in df.columns:
        gt = df["anomaly_label"].astype(bool)
        ax0.scatter(df.index[gt], df[sensor_col][gt], color=ANOMALY_COLOR, s=8, label="Ground truth", zorder=5)
    ax0.set_ylabel(sensor_col)
    ax0.set_title("Raw signal + ground truth", fontsize=10)
    ax0.legend(fontsize=9)

    # One panel per detector
    for ax, (name, preds) in zip(axes[1:], predictions.items()):
        preds = np.asarray(preds)
        ax.plot(df.index, df[sensor_col], color=NORMAL_COLOR, linewidth=0.6, alpha=0.6)
        detected_mask = preds.astype(bool)
        ax.scatter(
            df.index[detected_mask], df[sensor_col][detected_mask],
            color=DETECTED_COLOR, s=10, label=f"{name} detections", zorder=5,
        )
        if "anomaly_label" in df.columns:
            gt_mask = df["anomaly_label"].astype(bool)
            tp = gt_mask & detected_mask
            fn = gt_mask & ~detected_mask
            ax.scatter(df.index[tp], df[sensor_col][tp], color="#3B6D11", s=18, marker="^", label="True positive", zorder=6)
            ax.scatter(df.index[fn], df[sensor_col][fn], color=ANOMALY_COLOR, s=18, marker="v", label="False negative", zorder=6)
        ax.set_ylabel(sensor_col)
        ax.set_title(f"Detector: {name}", fontsize=10)
        ax.legend(fontsize=8, ncol=3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.suptitle("Detection results by method", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_reconstruction_error(
    df: pd.DataFrame,
    errors: np.ndarray,
    threshold: float,
    detector_name: str = "LSTM Autoencoder",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot reconstruction error over time with the anomaly threshold line.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Top: reconstruction error
    sensor_cols = [c for c in df.columns if c not in ("anomaly_label",)]
    axes[0].plot(df.index, errors, color=NORMAL_COLOR, linewidth=0.7, label="Reconstruction error")
    axes[0].axhline(threshold, color=ANOMALY_COLOR, linewidth=1.2, linestyle="--", label=f"Threshold ({threshold:.4f})")
    axes[0].fill_between(df.index, errors, threshold, where=(errors > threshold), color=ANOMALY_COLOR, alpha=0.3)
    axes[0].set_ylabel("MSE")
    axes[0].set_title(f"{detector_name} — reconstruction error", fontsize=10)
    axes[0].legend(fontsize=9)

    # Bottom: predicted vs. ground truth
    pred = (errors > threshold).astype(int)
    axes[1].plot(df.index, pred, color=DETECTED_COLOR, linewidth=0.8, label="Predicted anomaly")
    if "anomaly_label" in df.columns:
        axes[1].fill_between(df.index, 0, 1, where=df["anomaly_label"].astype(bool),
                             color=ANOMALY_COLOR, alpha=0.25, label="Ground truth anomaly")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Normal", "Anomaly"])
    axes[1].legend(fontsize=9)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_method_comparison(
    results: list[dict],
    save_path: str | Path | None = None,
) -> None:
    """
    Bar chart comparing Precision, Recall, F1 across detectors.

    Parameters
    ----------
    results : list of dicts with keys [detector, precision, recall, f1]
    """
    df = pd.DataFrame(results).set_index("detector")
    metrics = ["precision", "recall", "f1"]
    colors = [NORMAL_COLOR, "#1D9E75", DETECTED_COLOR]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.25

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df[metric], width, label=metric.capitalize(), color=color, alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df.index, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Detector performance comparison", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Figure saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()
