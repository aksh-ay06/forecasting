#!/usr/bin/env python
"""Run batch inference on test.csv and generate submission file."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TEST_CSV, PROCESSED_DIR, ARTIFACTS_DIR, TARGET_COL
from src.data.loader import load_test, load_parquet_semanas
from src.features.pipeline import build_features, get_feature_columns
from src.models.gbm_model import GBMModel


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Load test data
    test = load_test()
    print(f"Test data: {len(test)} rows")

    # Load all available history
    available_semanas = []
    for s in range(3, 12):
        path = PROCESSED_DIR / f"semana_{s}.parquet"
        if path.exists():
            available_semanas.append(s)

    if not available_semanas:
        print("ERROR: No preprocessed data found. Run preprocessing first.")
        sys.exit(1)

    history = load_parquet_semanas(available_semanas)
    print(f"History: {len(history)} rows (Semanas {available_semanas})")

    # Load model
    model = GBMModel.load()
    feature_cols = model.feature_names

    # Process each test Semana separately
    all_preds = []
    for semana in sorted(test["Semana"].unique()):
        test_s = test[test["Semana"] == semana]
        print(f"\nProcessing Semana {semana}: {len(test_s)} rows")

        feats = build_features(test_s, history, target_semana=semana, include_target=False)

        # Ensure columns match
        for col in feature_cols:
            if col not in feats.columns:
                feats[col] = 0
        X = feats[feature_cols].values

        preds = model.predict(X)
        result = test_s[["id"]].copy()
        result[TARGET_COL] = preds.round().astype(int)
        result[TARGET_COL] = result[TARGET_COL].clip(lower=0)
        all_preds.append(result)

    submission = pd.concat(all_preds, ignore_index=True)
    submission = submission.sort_values("id").reset_index(drop=True)

    output_path = ARTIFACTS_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved: {output_path} ({len(submission)} rows)")
    print(f"Prediction stats: mean={submission[TARGET_COL].mean():.2f}, "
          f"median={submission[TARGET_COL].median():.0f}, "
          f"max={submission[TARGET_COL].max()}")


if __name__ == "__main__":
    main()
