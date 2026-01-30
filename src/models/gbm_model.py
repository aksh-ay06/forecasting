import logging
from pathlib import Path

import numpy as np
import lightgbm as lgb

from src.config import (
    LGB_PARAMS, LGB_NUM_BOOST_ROUND, LGB_EARLY_STOPPING_ROUNDS,
    LGB_QUANTILE_PARAMS_LO, LGB_QUANTILE_PARAMS_HI,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class GBMModel:

    def __init__(self, params: dict | None = None, with_quantiles: bool = True):
        self.params = params or LGB_PARAMS.copy()
        self.with_quantiles = with_quantiles
        self.model: lgb.Booster | None = None
        self.model_lo: lgb.Booster | None = None
        self.model_hi: lgb.Booster | None = None
        self.feature_names: list[str] = []

    def train(self,
              dtrain: lgb.Dataset,
              dval: lgb.Dataset | None = None,
              feature_names: list[str] | None = None,
              ) -> dict:
        self.feature_names = feature_names or dtrain.feature_name

        callbacks = [lgb.log_evaluation(100)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if dval is not None:
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS))

        logger.info("Training main LightGBM model …")
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=LGB_NUM_BOOST_ROUND,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        metrics = {"best_iteration": self.model.best_iteration}

        if self.with_quantiles:
            logger.info("Training quantile (α=0.1) model …")
            self.model_lo = lgb.train(
                LGB_QUANTILE_PARAMS_LO,
                dtrain,
                num_boost_round=self.model.best_iteration or LGB_NUM_BOOST_ROUND,
            )
            logger.info("Training quantile (α=0.9) model …")
            self.model_hi = lgb.train(
                LGB_QUANTILE_PARAMS_HI,
                dtrain,
                num_boost_round=self.model.best_iteration or LGB_NUM_BOOST_ROUND,
            )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        return np.clip(preds, 0, None)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred = self.predict(X)
        if self.model_lo is not None and self.model_hi is not None:
            lower = np.clip(self.model_lo.predict(X), 0, None)
            upper = np.clip(self.model_hi.predict(X), 0, None)
        else:
            lower = pred
            upper = pred
        return pred, lower, upper

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        imp = self.model.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names, imp))

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path / "model_main.txt"))
        if self.model_lo is not None:
            self.model_lo.save_model(str(path / "model_lo.txt"))
        if self.model_hi is not None:
            self.model_hi.save_model(str(path / "model_hi.txt"))

        logger.info("Models saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "GBMModel":
        path = path or MODELS_DIR
        model = cls(with_quantiles=False)

        model.model = lgb.Booster(model_file=str(path / "model_main.txt"))
        model.feature_names = model.model.feature_name()

        lo_path = path / "model_lo.txt"
        hi_path = path / "model_hi.txt"
        if lo_path.exists() and hi_path.exists():
            model.model_lo = lgb.Booster(model_file=str(lo_path))
            model.model_hi = lgb.Booster(model_file=str(hi_path))
            model.with_quantiles = True

        logger.info("Loaded models from %s", path)
        return model
