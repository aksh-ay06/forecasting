"""End-to-end training orchestration with MLflow tracking."""

import logging

import mlflow
import numpy as np
import pandas as pd

from src.config import (
    TARGET_COL, VAL_SEMANA, PROCESSED_DIR,
    MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, LGB_PARAMS,
)
from src.data.loader import load_parquet_semanas
from src.features.pipeline import build_features, get_feature_columns
from src.models.gbm_model import GBMModel
from src.models.baseline import naive_lag1_baseline, moving_average_baseline
from src.evaluation.metrics import evaluate

logger = logging.getLogger(__name__)


def run_training(val_semana: int = VAL_SEMANA,
                 with_quantiles: bool = True,
                 log_mlflow: bool = True) -> dict:
    """Full training pipeline.

    1. Load preprocessed parquet files.
    2. Build features for train and validation sets.
    3. Compute baseline metrics.
    4. Train LightGBM.
    5. Evaluate and log to MLflow.
    6. Save model.

    Returns dict with metrics.
    """
    # 1. Load data
    train_semanas = [s for s in range(3, val_semana)]
    logger.info("Loading Semanas %s for training, %d for validation …",
                train_semanas, val_semana)

    all_data = load_parquet_semanas(list(range(3, val_semana + 1)))
    history = all_data[all_data["Semana"] < val_semana]
    val_raw = all_data[all_data["Semana"] == val_semana]

    # 2. Build features
    logger.info("Building features for validation Semana %d …", val_semana)
    val_feats = build_features(val_raw, history, target_semana=val_semana, include_target=True)

    # Also build training features for the previous week as training data
    # We train on Semana 4..val_semana-1, each using all prior data as history
    train_dfs = []
    for s in range(4, val_semana):
        s_data = all_data[all_data["Semana"] == s]
        s_history = all_data[all_data["Semana"] < s]
        if len(s_data) == 0 or len(s_history) == 0:
            continue
        s_feats = build_features(s_data, s_history, target_semana=s, include_target=True)
        train_dfs.append(s_feats)

    train_feats = pd.concat(train_dfs, ignore_index=True)
    feature_cols = get_feature_columns(train_feats)
    logger.info("Training features: %d rows × %d features", len(train_feats), len(feature_cols))

    X_train = train_feats[feature_cols].values
    y_train = train_feats[TARGET_COL].values
    X_val = val_feats[feature_cols].values
    y_val = val_feats[TARGET_COL].values

    # 3. Baselines
    baseline_lag1_preds = naive_lag1_baseline(history, val_raw, val_semana)
    baseline_ma_preds = moving_average_baseline(history, val_raw, val_semana)
    baseline_lag1_metrics = evaluate(y_val, baseline_lag1_preds)
    baseline_ma_metrics = evaluate(y_val, baseline_ma_preds)
    logger.info("Baseline lag-1:  %s", baseline_lag1_metrics)
    logger.info("Baseline MA(3):  %s", baseline_ma_metrics)

    # 4. Train GBM
    model = GBMModel(with_quantiles=with_quantiles)
    train_metrics = model.train(
        X_train, y_train, X_val, y_val,
        feature_names=feature_cols,
    )

    # 5. Evaluate
    preds = model.predict(X_val)
    gbm_metrics = evaluate(y_val, preds)
    logger.info("GBM metrics:     %s", gbm_metrics)

    # Feature importance
    importance = model.feature_importance()
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:20]
    logger.info("Top features: %s", [(f, f"{v:.0f}") for f, v in top_features])

    # 6. MLflow logging
    if log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.log_params({k: str(v) for k, v in LGB_PARAMS.items()})
            mlflow.log_param("val_semana", val_semana)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("n_train_rows", len(train_feats))
            mlflow.log_param("n_val_rows", len(val_feats))

            for name, val in gbm_metrics.items():
                mlflow.log_metric(f"gbm_{name}", val)
            for name, val in baseline_lag1_metrics.items():
                mlflow.log_metric(f"baseline_lag1_{name}", val)
            for name, val in baseline_ma_metrics.items():
                mlflow.log_metric(f"baseline_ma_{name}", val)
            mlflow.log_metric("best_iteration", train_metrics.get("best_iteration", 0))

            # Log feature importance as artifact
            imp_df = pd.DataFrame(top_features, columns=["feature", "importance"])
            imp_path = "feature_importance.csv"
            imp_df.to_csv(imp_path, index=False)
            mlflow.log_artifact(imp_path)

    # 7. Save model
    model.save()

    return {
        "gbm": gbm_metrics,
        "baseline_lag1": baseline_lag1_metrics,
        "baseline_ma": baseline_ma_metrics,
        "best_iteration": train_metrics.get("best_iteration", 0),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }
