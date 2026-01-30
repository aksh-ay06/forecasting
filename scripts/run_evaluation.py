#!/usr/bin/env python

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VAL_SEMANA, TARGET_COL, REPORTS_DIR
from src.data.loader import load_parquet_semanas
from src.features.pipeline import build_features, get_feature_columns
from src.models.gbm_model import GBMModel
from src.evaluation.metrics import evaluate, evaluate_by_segment
from src.evaluation.analysis import generate_all_plots
from src.evaluation.explainability import generate_shap_report


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_data = load_parquet_semanas(list(range(3, VAL_SEMANA + 1)))
    history = all_data[all_data["Semana"] < VAL_SEMANA]
    val_raw = all_data[all_data["Semana"] == VAL_SEMANA]

    val_feats = build_features(val_raw, history, target_semana=VAL_SEMANA, include_target=True)
    feature_cols = get_feature_columns(val_feats)

    X_val = val_feats[feature_cols].values
    y_val = val_feats[TARGET_COL].values

    model = GBMModel.load()
    preds = model.predict(X_val)

    metrics = evaluate(y_val, preds)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    val_feats["predicted"] = preds
    val_feats["actual"] = y_val
    for col in ["Producto_ID", "Agencia_ID"]:
        seg_metrics = evaluate_by_segment(val_feats, "actual", "predicted", col)
        seg_metrics.to_csv(REPORTS_DIR / f"segment_metrics_{col}.csv", index=False)
        print(f"\nWorst 5 segments by {col} (RMSE):")
        print(seg_metrics.head().to_string())

    generate_all_plots(y_val, preds, val_feats, REPORTS_DIR)

    shap_table = generate_shap_report(model.model, X_val, feature_cols, REPORTS_DIR)
    shap_table.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)

    print(f"\nReports saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
