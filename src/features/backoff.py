import logging

import pandas as pd
import numpy as np

from src.config import TARGET_COL, FEATURES_DIR

logger = logging.getLogger(__name__)


def _compute_level_means(history: pd.DataFrame, brand_map: pd.DataFrame) -> list[pd.DataFrame]:
    if "brand" not in history.columns:
        h = history.merge(brand_map, on="Producto_ID", how="left")
        h["brand"] = h["brand"].fillna("UNK")
    else:
        h = history

    global_mean = h[TARGET_COL].mean()

    levels = []

    levels.append(
        h.groupby(["Cliente_ID", "Producto_ID"])[TARGET_COL]
        .mean().reset_index().rename(columns={TARGET_COL: "backoff_est_0"})
    )
    levels.append(
        h.groupby(["Ruta_SAK", "Producto_ID"])[TARGET_COL]
        .mean().reset_index().rename(columns={TARGET_COL: "backoff_est_1"})
    )
    levels.append(
        h.groupby(["Agencia_ID", "Producto_ID"])[TARGET_COL]
        .mean().reset_index().rename(columns={TARGET_COL: "backoff_est_2"})
    )
    levels.append(
        h.groupby(["Producto_ID"])[TARGET_COL]
        .mean().reset_index().rename(columns={TARGET_COL: "backoff_est_3"})
    )
    levels.append(
        h.groupby(["brand"])[TARGET_COL]
        .mean().reset_index().rename(columns={TARGET_COL: "backoff_est_4"})
    )
    levels.append(pd.DataFrame({"backoff_est_5": [global_mean]}))

    return levels


def build_backoff_features(target_df: pd.DataFrame,
                           history: pd.DataFrame,
                           target_semana: int) -> pd.DataFrame:
    hist = history

    brand_path = FEATURES_DIR / "product_brand_map.parquet"
    if brand_path.exists():
        brand_map = pd.read_parquet(brand_path)
    else:
        brand_map = pd.DataFrame({"Producto_ID": [], "brand": []})

    levels = _compute_level_means(hist, brand_map)
    df = target_df.copy()

    if "brand" not in df.columns:
        df = df.merge(brand_map, on="Producto_ID", how="left")
        df["brand"] = df["brand"].fillna("UNK")

    df = df.merge(levels[0], on=["Cliente_ID", "Producto_ID"], how="left")
    df = df.merge(levels[1], on=["Ruta_SAK", "Producto_ID"], how="left")
    df = df.merge(levels[2], on=["Agencia_ID", "Producto_ID"], how="left")
    df = df.merge(levels[3], on=["Producto_ID"], how="left")
    df = df.merge(levels[4], on=["brand"], how="left")
    global_mean = levels[5]["backoff_est_5"].iloc[0]

    est_cols = [f"backoff_est_{i}" for i in range(5)]
    df["backoff_estimate"] = np.nan
    df["backoff_level"] = 5

    for i in range(5, -1, -1):
        col = f"backoff_est_{i}"
        if i == 5:
            mask = df["backoff_estimate"].isna()
            df.loc[mask, "backoff_estimate"] = global_mean
            df.loc[mask, "backoff_level"] = 5
        else:
            mask = df[col].notna()
            df.loc[mask, "backoff_estimate"] = df.loc[mask, col]
            df.loc[mask, "backoff_level"] = i

    df.drop(columns=est_cols, inplace=True, errors="ignore")
    if "brand" in df.columns and "brand" not in target_df.columns:
        df.drop(columns=["brand"], inplace=True)

    df["backoff_level"] = df["backoff_level"].astype("uint8")
    df["backoff_estimate"] = df["backoff_estimate"].astype("float32")

    logger.info("Backoff levels distribution:\n%s",
                df["backoff_level"].value_counts().sort_index().to_string())
    return df
