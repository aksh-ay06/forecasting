import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_c = np.clip(y_true, 0, None)
    y_pred_c = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred_c) - np.log1p(y_true_c)) ** 2)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "rmsle": rmsle(y_true, y_pred),
    }


def evaluate_by_segment(df: pd.DataFrame,
                        y_true_col: str,
                        y_pred_col: str,
                        segment_col: str) -> pd.DataFrame:
    rows = []
    for seg, group in df.groupby(segment_col):
        y_t = group[y_true_col].values
        y_p = group[y_pred_col].values
        m = evaluate(y_t, y_p)
        m[segment_col] = seg
        m["n"] = len(group)
        rows.append(m)

    result = pd.DataFrame(rows)
    cols = [segment_col, "n"] + [c for c in result.columns if c not in [segment_col, "n"]]
    result = result[cols].sort_values("rmse", ascending=False).reset_index(drop=True)
    logger.info("Segment evaluation on '%s': %d segments", segment_col, len(result))
    return result
