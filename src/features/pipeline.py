import logging

import pandas as pd

from src.config import TARGET_COL, AGG_GROUP_KEYS
from src.features.lag_features import build_lag_features
from src.features.agg_features import build_agg_features
from src.features.product_features import load_product_features
from src.features.backoff import build_backoff_features

logger = logging.getLogger(__name__)

ID_COLS = ["Semana", "Cliente_ID", "Producto_ID", "Agencia_ID",
           "Ruta_SAK", "Canal_ID"]


def build_features(target_df: pd.DataFrame,
                   history: pd.DataFrame,
                   target_semana: int,
                   include_target: bool = True) -> pd.DataFrame:
    logger.info("Building features for Semana %d (%d target rows) …", target_semana, len(target_df))
    df = target_df.copy()

    lag_feats = build_lag_features(history, target_semana)
    if not lag_feats.empty:
        df = df.merge(lag_feats, on=["Cliente_ID", "Producto_ID"], how="left")

    agg_dict = build_agg_features(history, target_semana)
    for name, agg_df in agg_dict.items():
        keys = AGG_GROUP_KEYS[name]
        df = df.merge(agg_df, on=keys, how="left")

    product_feats = load_product_features()
    df = df.merge(product_feats, on="Producto_ID", how="left")

    df = build_backoff_features(df, history, target_semana)

    df["target_semana"] = target_semana

    feature_cols = [c for c in df.columns if c not in ID_COLS and c != TARGET_COL]
    df[feature_cols] = df[feature_cols].fillna(0)

    if not include_target and TARGET_COL in df.columns:
        df.drop(columns=[TARGET_COL], inplace=True)

    logger.info("Feature matrix: %d rows × %d columns", len(df), len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in ID_COLS and c != TARGET_COL
            and df[c].dtype.kind in ("f", "i", "u", "b")]
