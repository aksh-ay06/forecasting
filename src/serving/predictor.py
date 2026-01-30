"""Inference logic with feature lookup for real-time predictions."""

import logging

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR, MODELS_DIR
from src.models.gbm_model import GBMModel
from src.features.pipeline import build_features, get_feature_columns
from src.data.loader import load_parquet_semanas

logger = logging.getLogger(__name__)


class Predictor:
    """Loads model and feature stores at startup for real-time inference."""

    def __init__(self):
        self.model: GBMModel | None = None
        self.history: pd.DataFrame | None = None
        self.feature_cols: list[str] = []

    def load(self) -> None:
        """Load model and historical data for feature computation."""
        logger.info("Loading model …")
        self.model = GBMModel.load(MODELS_DIR)
        self.feature_cols = self.model.feature_names

        # Load historical data for feature computation
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
        """Predict demand for a single request.

        Parameters
        ----------
        request : dict with keys Semana, Agencia_ID, Canal_ID, Ruta_SAK, Cliente_ID, Producto_ID.

        Returns
        -------
        (predicted_demand, confidence_lower, confidence_upper)
        """
        target_df = pd.DataFrame([request])
        return self._predict(target_df)

    def predict_batch(self, requests: list[dict]) -> list[tuple[float, float, float]]:
        """Predict demand for a batch of requests."""
        target_df = pd.DataFrame(requests)
        results = []
        # Process all at once for efficiency
        feats = build_features(
            target_df, self.history,
            target_semana=target_df["Semana"].iloc[0],
            include_target=False,
        )

        # Ensure columns match model's expected features
        for col in self.feature_cols:
            if col not in feats.columns:
                feats[col] = 0
        X = feats[self.feature_cols].values

        pred, lower, upper = self.model.predict_interval(X)
        for i in range(len(pred)):
            results.append((float(pred[i]), float(lower[i]), float(upper[i])))
        return results

    def _predict(self, target_df: pd.DataFrame) -> tuple[float, float, float]:
        """Internal prediction for a single-row DataFrame."""
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
