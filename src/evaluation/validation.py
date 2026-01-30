"""Temporal split and rolling window cross-validation."""

import logging

import pandas as pd

from src.config import TRAIN_SEMANAS, VAL_SEMANA

logger = logging.getLogger(__name__)


def temporal_train_val_split(df: pd.DataFrame,
                             val_semana: int = VAL_SEMANA
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by Semana into train and validation sets.

    Parameters
    ----------
    df : DataFrame with Semana column.
    val_semana : int, the week used for validation.

    Returns
    -------
    (train_df, val_df)
    """
    train = df[df["Semana"] < val_semana].copy()
    val = df[df["Semana"] == val_semana].copy()
    logger.info("Temporal split: train Semana < %d (%d rows), val Semana == %d (%d rows)",
                val_semana, len(train), val_semana, len(val))
    return train, val


def rolling_window_cv(df: pd.DataFrame,
                      min_train_semanas: int = 3,
                      ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate rolling-window cross-validation folds.

    Each fold uses all weeks before the validation week as training data.
    The validation week advances from the earliest feasible week to the last
    available week.

    Parameters
    ----------
    df : DataFrame with Semana column.
    min_train_semanas : int
        Minimum number of training weeks required before the first fold.

    Returns
    -------
    List of (train_df, val_df) tuples.
    """
    semanas = sorted(df["Semana"].unique())
    folds = []

    for i, val_s in enumerate(semanas):
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
