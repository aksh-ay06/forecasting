import logging

import pandas as pd
import numpy as np

from src.config import LAG_WEEKS, ROLLING_WINDOWS, TARGET_COL

logger = logging.getLogger(__name__)

GROUP_KEYS = ["Cliente_ID", "Producto_ID"]


def build_lag_features(history: pd.DataFrame, target_semana: int) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    df = history

    demand = (
        df.groupby(GROUP_KEYS + ["Semana"])[TARGET_COL]
        .sum()
        .reset_index()
        .sort_values("Semana")
    )

    features = demand[GROUP_KEYS].drop_duplicates().copy()

    for lag in LAG_WEEKS:
        lag_semana = target_semana - lag
        lag_df = demand[demand["Semana"] == lag_semana][GROUP_KEYS + [TARGET_COL]].rename(
            columns={TARGET_COL: f"lag_{lag}"}
        )
        features = features.merge(lag_df, on=GROUP_KEYS, how="left")

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

    std_cols = [c for c in features.columns if "rolling_std" in c]
    features[std_cols] = features[std_cols].fillna(0)

    float_cols = [c for c in features.columns if c not in GROUP_KEYS]
    features[float_cols] = features[float_cols].astype(np.float32)

    logger.info("Built %d lag/rolling features for target_semana=%d on %d groups",
                len([c for c in features.columns if c not in GROUP_KEYS]),
                target_semana, len(features))
    return features
