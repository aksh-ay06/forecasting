#!/usr/bin/env python

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.trainer import run_training


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    results = run_training()

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"GBM metrics:          {results['gbm']}")
    print(f"Baseline lag-1:       {results['baseline_lag1']}")
    print(f"Baseline MA(3):       {results['baseline_ma']}")
    print(f"Best iteration:       {results['best_iteration']}")
    print(f"Number of features:   {results['n_features']}")
    print("=" * 60)

    if results["gbm"]["rmse"] < results["baseline_lag1"]["rmse"]:
        print("GBM RMSE < Lag-1 baseline RMSE")
    else:
        print("WARNING: GBM did not beat lag-1 baseline")


if __name__ == "__main__":
    main()
