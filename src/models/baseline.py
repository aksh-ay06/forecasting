"""Naive baselines for demand forecasting."""

import logging

import pandas as pd
import numpy as np

from src.config import TARGET_COL

logger = logging.getLogger(__name__)


def naive_lag1_baseline(history: pd.DataFrame,
                        target_df: pd.DataFrame,
                        target_semana: int) -> np.ndarray:
    """Predict demand as last week's demand for the same (client, product).

    Falls back to product mean, then global mean for unseen pairs.
    """
    prev = history[history["Semana"] == target_semana - 1]
    last_demand = (
        prev.groupby(["Cliente_ID", "Producto_ID"])[TARGET_COL]
        .sum()
        .reset_index()
        .rename(columns={TARGET_COL: "pred"})
    )

    merged = target_df[["Cliente_ID", "Producto_ID"]].merge(
        last_demand, on=["Cliente_ID", "Producto_ID"], how="left"
    )

    # Fallback: product mean
    product_mean = (
        history[history["Semana"] < target_semana]
        .groupby("Producto_ID")[TARGET_COL].mean()
        .reset_index()
        .rename(columns={TARGET_COL: "prod_mean"})
    )
    merged = merged.merge(product_mean, on="Producto_ID", how="left")

    global_mean = history[history["Semana"] < target_semana][TARGET_COL].mean()

    preds = merged["pred"].fillna(merged["prod_mean"]).fillna(global_mean).values
    preds = np.clip(preds, 0, None)

    logger.info("Lag-1 baseline: %d predictions, mean=%.2f", len(preds), preds.mean())
    return preds


def moving_average_baseline(history: pd.DataFrame,
                            target_df: pd.DataFrame,
                            target_semana: int,
                            window: int = 3) -> np.ndarray:
    """Predict demand as the mean of the last `window` weeks."""
    recent_semanas = sorted(history[history["Semana"] < target_semana]["Semana"].unique())[-window:]
    recent = history[history["Semana"].isin(recent_semanas)]

    ma = (
        recent.groupby(["Cliente_ID", "Producto_ID"])[TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COL: "pred"})
    )

    merged = target_df[["Cliente_ID", "Producto_ID"]].merge(
        ma, on=["Cliente_ID", "Producto_ID"], how="left"
    )

    global_mean = history[history["Semana"] < target_semana][TARGET_COL].mean()
    preds = merged["pred"].fillna(global_mean).values
    preds = np.clip(preds, 0, None)

    logger.info("MA(%d) baseline: %d predictions, mean=%.2f", window, len(preds), preds.mean())
    return preds
