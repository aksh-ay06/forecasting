import logging

import pandas as pd

from src.config import TRAIN_SEMANAS, VAL_SEMANA

logger = logging.getLogger(__name__)


def temporal_train_val_split(df: pd.DataFrame,
                             val_semana: int = VAL_SEMANA
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["Semana"] < val_semana].copy()
    val = df[df["Semana"] == val_semana].copy()
    logger.info("Temporal split: train Semana < %d (%d rows), val Semana == %d (%d rows)",
                val_semana, len(train), val_semana, len(val))
    return train, val


def rolling_window_cv(df: pd.DataFrame,
                      min_train_semanas: int = 3,
                      ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    semanas = sorted(df["Semana"].unique())
    folds = []

    for val_s in semanas:
        train_semanas = [s for s in semanas if s < val_s]
        if len(train_semanas) < min_train_semanas:
            continue
        train = df[df["Semana"].isin(train_semanas)]
        val = df[df["Semana"] == val_s]
        folds.append((train, val))
        logger.info("CV fold %d: train Semana %s → val Semana %d (%d / %d rows)",
                     len(folds), train_semanas, val_s, len(train), len(val))

    logger.info("Generated %d rolling CV folds", len(folds))
    return folds
