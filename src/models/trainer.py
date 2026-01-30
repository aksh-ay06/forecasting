"""End-to-end training orchestration with MLflow tracking."""

import ctypes
import gc
import logging

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.config import (
    TARGET_COL, VAL_SEMANA, PROCESSED_DIR, REPORTS_DIR,
    FEATURES_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, LGB_PARAMS,
)
from src.data.loader import load_parquet_semanas
from src.features.pipeline import build_features, get_feature_columns
from src.models.gbm_model import GBMModel
from src.models.baseline import naive_lag1_baseline, moving_average_baseline
from src.evaluation.metrics import evaluate

logger = logging.getLogger(__name__)


def _release_memory():
    """Collect garbage and force glibc to return freed pages to the OS.

    Without malloc_trim, Python/glibc hold freed heap pages in the
    process address space indefinitely, bloating RSS and causing OOM
    when large arrays are allocated later.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def run_training(val_semana: int = VAL_SEMANA,
                 with_quantiles: bool = True,
                 log_mlflow: bool = True) -> dict:
    """Full training pipeline.

    1. Build features for train and validation sets.
    2. Compute baseline metrics.
    3. Train LightGBM.
    4. Evaluate and log to MLflow.
    5. Save model.

    Returns dict with metrics.
    """
    # 1. Build training features semana by semana, spilling to disk.
    #    Load data on-demand per iteration instead of holding all semanas
    #    in memory at once, to keep peak RSS low.
    logger.info("Building training features for Semanas 4-%d …", val_semana - 1)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    train_feat_paths = []
    train_row_counts = []
    for s in range(4, val_semana):
        s_history = load_parquet_semanas(list(range(3, s)))
        s_data = load_parquet_semanas([s])
        if len(s_data) == 0 or len(s_history) == 0:
            del s_history, s_data
            _release_memory()
            continue
        s_feats = build_features(s_data, s_history, target_semana=s, include_target=True)
        path = FEATURES_DIR / f"_train_feats_s{s}.parquet"
        s_feats.to_parquet(path, index=False)
        train_feat_paths.append(path)
        train_row_counts.append(len(s_feats))
        del s_data, s_history, s_feats
        _release_memory()

    # 2. Build validation features + baselines
    logger.info("Building features for validation Semana %d …", val_semana)
    history = load_parquet_semanas(list(range(3, val_semana)))
    val_raw = load_parquet_semanas([val_semana])
    val_feats = build_features(val_raw, history, target_semana=val_semana, include_target=True)

    baseline_lag1_preds = naive_lag1_baseline(history, val_raw, val_semana)
    baseline_ma_preds = moving_average_baseline(history, val_raw, val_semana)
    y_val = val_feats[TARGET_COL].values
    baseline_lag1_metrics = evaluate(y_val, baseline_lag1_preds)
    baseline_ma_metrics = evaluate(y_val, baseline_ma_preds)
    logger.info("Baseline lag-1:  %s", baseline_lag1_metrics)
    logger.info("Baseline MA(3):  %s", baseline_ma_metrics)

    del history, val_raw, baseline_lag1_preds, baseline_ma_preds
    _release_memory()

    # 3. Extract validation arrays, then free the DataFrame
    feature_cols = get_feature_columns(val_feats)
    X_val = val_feats[feature_cols].values.astype(np.float32)
    n_val_rows = len(val_feats)
    del val_feats
    _release_memory()

    # 4. Reload training features from disk into pre-allocated arrays.
    #    Pre-allocation avoids the 2× peak memory of np.vstack.
    n_train_rows = sum(train_row_counts)
    n_features = len(feature_cols)
    logger.info("Training features: %d rows × %d features", n_train_rows, n_features)

    X_train = np.empty((n_train_rows, n_features), dtype=np.float32)
    y_train = np.empty(n_train_rows, dtype=np.float32)
    offset = 0
    for p, n_rows in zip(train_feat_paths, train_row_counts):
        part = pd.read_parquet(p)
        X_train[offset:offset + n_rows] = part[feature_cols].values.astype(np.float32)
        y_train[offset:offset + n_rows] = part[TARGET_COL].values.astype(np.float32)
        offset += n_rows
        del part
        gc.collect()
        p.unlink(missing_ok=True)

    # 5. Build LightGBM Datasets, then free raw numpy arrays before training.
    #    This avoids holding ~10 GB of raw floats alongside LightGBM's internal
    #    binned representation during tree construction.
    dtrain = lgb.Dataset(X_train, label=y_train,
                         feature_name=feature_cols,
                         categorical_feature="auto",
                         free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val,
                       feature_name=feature_cols,
                       categorical_feature="auto",
                       free_raw_data=False)
    dtrain.construct()
    del X_train, y_train
    _release_memory()
    logger.info("LightGBM Datasets constructed; raw training arrays freed.")

    # 6. Train GBM
    model = GBMModel(with_quantiles=with_quantiles)
    train_metrics = model.train(dtrain, dval, feature_names=feature_cols)

    del dtrain
    _release_memory()

    # 7. Evaluate
    preds = model.predict(X_val)
    gbm_metrics = evaluate(y_val, preds)
    logger.info("GBM metrics:     %s", gbm_metrics)

    # Feature importance
    importance = model.feature_importance()
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:20]
    logger.info("Top features: %s", [(f, f"{v:.0f}") for f, v in top_features])

    # 8. MLflow logging
    if log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.log_params({k: str(v) for k, v in LGB_PARAMS.items()})
            mlflow.log_param("val_semana", val_semana)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("n_train_rows", n_train_rows)
            mlflow.log_param("n_val_rows", n_val_rows)

            for name, val in gbm_metrics.items():
                mlflow.log_metric(f"gbm_{name}", val)
            for name, val in baseline_lag1_metrics.items():
                mlflow.log_metric(f"baseline_lag1_{name}", val)
            for name, val in baseline_ma_metrics.items():
                mlflow.log_metric(f"baseline_ma_{name}", val)
            mlflow.log_metric("best_iteration", train_metrics.get("best_iteration", 0))

            # Log feature importance as artifact
            imp_df = pd.DataFrame(top_features, columns=["feature", "importance"])
            imp_path = REPORTS_DIR / "feature_importance.csv"
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            imp_df.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path))

    # 9. Save model
    model.save()

    return {
        "gbm": gbm_metrics,
        "baseline_lag1": baseline_lag1_metrics,
        "baseline_ma": baseline_ma_metrics,
        "best_iteration": train_metrics.get("best_iteration", 0),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }
