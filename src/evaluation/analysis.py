"""Error distribution plots, residual diagnostics, and segment analysis."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                            save_dir: Path | None = None) -> None:
    """Histogram of prediction errors."""
    save_dir = save_dir or REPORTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    errors = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw error histogram
    axes[0].hist(errors, bins=100, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Prediction Error (pred - actual)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)

    # Clipped for visibility
    clip_val = np.percentile(np.abs(errors), 99)
    clipped = errors[(errors > -clip_val) & (errors < clip_val)]
    axes[1].hist(clipped, bins=100, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Prediction Error (clipped 99th pct)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Error Distribution (Clipped)")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(save_dir / "error_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved error distribution plot")


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                             save_dir: Path | None = None) -> None:
    """Scatter plot of predicted vs actual values."""
    save_dir = save_dir or REPORTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for plotting performance
    n = len(y_true)
    if n > 50_000:
        idx = np.random.RandomState(42).choice(n, 50_000, replace=False)
        y_true_s, y_pred_s = y_true[idx], y_pred[idx]
    else:
        y_true_s, y_pred_s = y_true, y_pred

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_s, y_pred_s, alpha=0.1, s=2)
    max_val = max(y_true_s.max(), y_pred_s.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1)
    ax.set_xlabel("Actual Demand")
    ax.set_ylabel("Predicted Demand")
    ax.set_title("Predicted vs Actual")
    ax.set_xlim(0, np.percentile(y_true_s, 99))
    ax.set_ylim(0, np.percentile(y_pred_s, 99))

    plt.tight_layout()
    plt.savefig(save_dir / "predicted_vs_actual.png", dpi=150)
    plt.close()
    logger.info("Saved predicted vs actual plot")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   save_dir: Path | None = None) -> None:
    """Residual diagnostics: residual vs predicted, and Q-Q-like plot."""
    save_dir = save_dir or REPORTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred
    n = len(y_true)
    if n > 50_000:
        idx = np.random.RandomState(42).choice(n, 50_000, replace=False)
        residuals_s, y_pred_s = residuals[idx], y_pred[idx]
    else:
        residuals_s, y_pred_s = residuals, y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual vs predicted
    axes[0].scatter(y_pred_s, residuals_s, alpha=0.1, s=2)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Demand")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs Predicted")

    # Residual histogram
    axes[1].hist(residuals_s, bins=100, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")
    axes[1].axvline(0, color="red", linestyle="--")

    plt.tight_layout()
    plt.savefig(save_dir / "residual_diagnostics.png", dpi=150)
    plt.close()
    logger.info("Saved residual diagnostics plot")


def segment_error_analysis(df: pd.DataFrame,
                           y_true_col: str,
                           y_pred_col: str,
                           segment_col: str,
                           top_n: int = 20,
                           save_dir: Path | None = None) -> None:
    """Bar plot of RMSE by segment for the worst-performing segments."""
    save_dir = save_dir or REPORTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    from src.evaluation.metrics import evaluate_by_segment
    seg_metrics = evaluate_by_segment(df, y_true_col, y_pred_col, segment_col)
    top = seg_metrics.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top["rmse"], color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top[segment_col].astype(str))
    ax.set_xlabel("RMSE")
    ax.set_title(f"Top {top_n} Worst Segments by RMSE ({segment_col})")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_dir / f"segment_rmse_{segment_col}.png", dpi=150)
    plt.close()
    logger.info("Saved segment RMSE plot for %s", segment_col)


def generate_all_plots(y_true: np.ndarray, y_pred: np.ndarray,
                       df: pd.DataFrame | None = None,
                       save_dir: Path | None = None) -> None:
    """Generate all diagnostic plots."""
    plot_error_distribution(y_true, y_pred, save_dir)
    plot_predicted_vs_actual(y_true, y_pred, save_dir)
    plot_residuals(y_true, y_pred, save_dir)

    if df is not None:
        for col in ["Producto_ID", "Agencia_ID"]:
            if col in df.columns:
                segment_error_analysis(df, "actual", "predicted", col, save_dir=save_dir)
