import logging

import pandas as pd
import numpy as np

from src.config import AGG_GROUP_KEYS, TARGET_COL

logger = logging.getLogger(__name__)


def build_agg_features(history: pd.DataFrame, target_semana: int) -> dict[str, pd.DataFrame]:
    df = history
    result = {}

    for name, keys in AGG_GROUP_KEYS.items():
        missing = [k for k in keys if k not in df.columns]
        if missing:
            logger.warning("Skipping agg group '%s': missing columns %s", name, missing)
            continue

        agg = (
            df.groupby(keys)[TARGET_COL]
            .agg(
                **{
                    f"{name}_mean": "mean",
                    f"{name}_std": "std",
                    f"{name}_count": "count",
                }
            )
            .reset_index()
        )
        agg[f"{name}_std"] = agg[f"{name}_std"].fillna(0)

        for col in [f"{name}_mean", f"{name}_std"]:
            agg[col] = agg[col].astype("float32")
        agg[f"{name}_count"] = agg[f"{name}_count"].astype("int32")

        result[name] = agg
        logger.info("Agg '%s': %d groups, cols=%s", name, len(agg),
                     [c for c in agg.columns if c not in keys])

    return result
