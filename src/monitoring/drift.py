"""Prediction distribution drift, data drift, and bias monitoring."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def prediction_drift(reference_preds: np.ndarray,
                     current_preds: np.ndarray,
                     threshold: float = 0.05) -> dict:
    """Detect drift in prediction distribution using KS test.

    Parameters
    ----------
    reference_preds : predictions from training/validation period.
    current_preds : recent production predictions.
    threshold : p-value threshold for drift alert.

    Returns
    -------
    Dict with statistic, p_value, and drift_detected flag.
    """
    ks_stat, p_value = stats.ks_2samp(reference_preds, current_preds)
    drift_detected = p_value < threshold

    result = {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "drift_detected": drift_detected,
        "reference_mean": float(np.mean(reference_preds)),
        "current_mean": float(np.mean(current_preds)),
        "reference_std": float(np.std(reference_preds)),
        "current_std": float(np.std(current_preds)),
    }

    if drift_detected:
        logger.warning("PREDICTION DRIFT DETECTED: KS=%.4f, p=%.6f", ks_stat, p_value)
    else:
        logger.info("No prediction drift: KS=%.4f, p=%.6f", ks_stat, p_value)

    return result


def feature_drift(reference_df: pd.DataFrame,
                  current_df: pd.DataFrame,
                  feature_cols: list[str],
                  threshold: float = 0.05) -> pd.DataFrame:
    """Detect drift in individual feature distributions.

    Returns DataFrame with one row per feature and drift status.
    """
    rows = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref = reference_df[col].dropna().values
        cur = current_df[col].dropna().values
        if len(ref) == 0 or len(cur) == 0:
            continue

        ks_stat, p_value = stats.ks_2samp(ref, cur)
        rows.append({
            "feature": col,
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drift_detected": p_value < threshold,
            "ref_mean": np.mean(ref),
            "cur_mean": np.mean(cur),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values("p_value").reset_index(drop=True)
        n_drift = result["drift_detected"].sum()
        logger.info("Feature drift: %d / %d features show drift (p < %.3f)",
                     n_drift, len(result), threshold)

    return result


def bias_monitor(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 segments: pd.Series | None = None) -> dict:
    """Monitor prediction bias overall and by segment.

    Returns dict with overall bias and per-segment bias if segments provided.
    """
    overall_bias = float(np.mean(y_pred - y_true))
    result = {"overall_bias": overall_bias}

    if segments is not None:
        seg_bias = {}
        for seg in segments.unique():
            mask = segments == seg
            seg_bias[str(seg)] = float(np.mean(y_pred[mask] - y_true[mask]))

        result["segment_bias"] = seg_bias
        # Flag segments with high bias
        high_bias = {k: v for k, v in seg_bias.items() if abs(v) > abs(overall_bias) * 2}
        if high_bias:
            logger.warning("High bias segments: %s", high_bias)

    return result


def plot_drift_report(reference_preds: np.ndarray,
                      current_preds: np.ndarray,
                      save_dir: Path | None = None) -> None:
    """Visual drift report comparing prediction distributions."""
    save_dir = save_dir or REPORTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overlapping histograms
    axes[0].hist(reference_preds, bins=80, alpha=0.5, label="Reference", density=True)
    axes[0].hist(current_preds, bins=80, alpha=0.5, label="Current", density=True)
    axes[0].set_xlabel("Predicted Demand")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Prediction Distribution: Reference vs Current")
    axes[0].legend()

    # ECDFs
    ref_sorted = np.sort(reference_preds)
    cur_sorted = np.sort(current_preds)
    axes[1].plot(ref_sorted, np.linspace(0, 1, len(ref_sorted)), label="Reference")
    axes[1].plot(cur_sorted, np.linspace(0, 1, len(cur_sorted)), label="Current")
    axes[1].set_xlabel("Predicted Demand")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Empirical CDF Comparison")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "drift_report.png", dpi=150)
    plt.close()
    logger.info("Saved drift report plot")
