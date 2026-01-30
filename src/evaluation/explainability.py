import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def compute_shap_values(model,
                        X: np.ndarray,
                        feature_names: list[str],
                        sample_size: int = 10_000) -> shap.Explanation:
    if len(X) > sample_size:
        idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    logger.info("Computing SHAP values on %d samples …", len(X_sample))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame(X_sample, columns=feature_names))

    return shap_values


def plot_shap_importance(shap_values: shap.Explanation,
                         max_display: int = 20,
                         save_dir: Path | None = None) -> None:
    save_dir = save_dir or REPORTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False, ax=ax)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP importance plot")


def plot_shap_summary(shap_values: shap.Explanation,
                      max_display: int = 20,
                      save_dir: Path | None = None) -> None:
    save_dir = save_dir or REPORTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot")


def shap_feature_importance_table(shap_values: shap.Explanation) -> pd.DataFrame:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    names = shap_values.feature_names
    df = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df


def generate_shap_report(model,
                         X: np.ndarray,
                         feature_names: list[str],
                         save_dir: Path | None = None) -> pd.DataFrame:
    shap_values = compute_shap_values(model, X, feature_names)
    plot_shap_importance(shap_values, save_dir=save_dir)
    plot_shap_summary(shap_values, save_dir=save_dir)
    table = shap_feature_importance_table(shap_values)
    logger.info("SHAP report complete. Top 5:\n%s", table.head().to_string())
    return table
