import logging

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR, MODELS_DIR
from src.models.gbm_model import GBMModel
from src.features.pipeline import build_features, get_feature_columns
from src.data.loader import load_parquet_semanas

logger = logging.getLogger(__name__)


class Predictor:

    def __init__(self):
        self.model: GBMModel | None = None
        self.history: pd.DataFrame | None = None
        self.feature_cols: list[str] = []

    def load(self) -> None:
        logger.info("Loading model …")
        self.model = GBMModel.load(MODELS_DIR)
        self.feature_cols = self.model.feature_names

        logger.info("Loading historical data for features …")
        available = []
        for s in range(3, 12):
            path = PROCESSED_DIR / f"semana_{s}.parquet"
            if path.exists():
                available.append(s)

        if available:
            self.history = load_parquet_semanas(available)
            logger.info("Loaded %d rows of history (Semanas %s)",
                        len(self.history), available)
        else:
            logger.warning("No historical parquet files found. Predictions will use defaults.")
            self.history = pd.DataFrame()

    def predict_single(self, request: dict) -> tuple[float, float, float]:
        target_df = pd.DataFrame([request])
        return self._predict(target_df)

    def predict_batch(self, requests: list[dict]) -> list[tuple[float, float, float]]:
        target_df = pd.DataFrame(requests)
        target_df["_batch_idx"] = range(len(target_df))

        all_results = pd.DataFrame(index=range(len(target_df)),
                                   columns=["pred", "lower", "upper"])

        for semana, group in target_df.groupby("Semana"):
            feats = build_features(
                group.drop(columns=["_batch_idx"]), self.history,
                target_semana=int(semana),
                include_target=False,
            )

            for col in self.feature_cols:
                if col not in feats.columns:
                    feats[col] = 0
            X = feats[self.feature_cols].values

            pred, lower, upper = self.model.predict_interval(X)
            idx = group["_batch_idx"].values
            all_results.loc[idx, "pred"] = pred
            all_results.loc[idx, "lower"] = lower
            all_results.loc[idx, "upper"] = upper

        return [
            (float(row["pred"]), float(row["lower"]), float(row["upper"]))
            for _, row in all_results.iterrows()
        ]

    def _predict(self, target_df: pd.DataFrame) -> tuple[float, float, float]:
        target_semana = int(target_df["Semana"].iloc[0])
        feats = build_features(
            target_df, self.history,
            target_semana=target_semana,
            include_target=False,
        )

        for col in self.feature_cols:
            if col not in feats.columns:
                feats[col] = 0
        X = feats[self.feature_cols].values

        pred, lower, upper = self.model.predict_interval(X)
        return float(pred[0]), float(lower[0]), float(upper[0])

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
