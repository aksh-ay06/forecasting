"""Lag demand and rolling statistics at the (Cliente_ID, Producto_ID) level."""

import logging

import pandas as pd
import numpy as np

from src.config import LAG_WEEKS, ROLLING_WINDOWS, TARGET_COL

logger = logging.getLogger(__name__)

GROUP_KEYS = ["Cliente_ID", "Producto_ID"]


def build_lag_features(history: pd.DataFrame, target_semana: int) -> pd.DataFrame:
    """Compute lag and rolling features using strictly prior data.

    Parameters
    ----------
    history : DataFrame
        Historical transactions with Semana < target_semana.
    target_semana : int
        The week to forecast. Only data with Semana < target_semana is used.

    Returns
    -------
    DataFrame indexed by (Cliente_ID, Producto_ID) with lag/rolling columns.
    """
    # Caller guarantees history contains only Semana < target_semana
    if history.empty:
        return pd.DataFrame()
    df = history

    # Pivot: one row per (client, product, semana)
    demand = (
        df.groupby(GROUP_KEYS + ["Semana"])[TARGET_COL]
        .sum()
        .reset_index()
        .sort_values("Semana")
    )

    features = demand[GROUP_KEYS].drop_duplicates().copy()

    # Lag features
    for lag in LAG_WEEKS:
        lag_semana = target_semana - lag
        lag_df = demand[demand["Semana"] == lag_semana][GROUP_KEYS + [TARGET_COL]].rename(
            columns={TARGET_COL: f"lag_{lag}"}
        )
        features = features.merge(lag_df, on=GROUP_KEYS, how="left")

    # Rolling statistics over the most recent `w` weeks
    valid_semanas = sorted(s for s in demand["Semana"].unique() if s < target_semana)
    for w in ROLLING_WINDOWS:
        recent = valid_semanas[-w:]
        sub = demand[demand["Semana"].isin(recent)]
        agg = sub.groupby(GROUP_KEYS)[TARGET_COL].agg(
            **{
                f"rolling_mean_{w}": "mean",
                f"rolling_std_{w}": "std",
                f"rolling_median_{w}": "median",
            }
        ).reset_index()
        features = features.merge(agg, on=GROUP_KEYS, how="left")

    # Fill NaN rolling_std with 0
    std_cols = [c for c in features.columns if "rolling_std" in c]
    features[std_cols] = features[std_cols].fillna(0)

    # Downcast to float32 to reduce memory (~50% savings on feature columns)
    float_cols = [c for c in features.columns if c not in GROUP_KEYS]
    features[float_cols] = features[float_cols].astype(np.float32)

    logger.info("Built %d lag/rolling features for target_semana=%d on %d groups",
                len([c for c in features.columns if c not in GROUP_KEYS]),
                target_semana, len(features))
    return features
