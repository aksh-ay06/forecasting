"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import rmse, mae, bias, rmsle, evaluate, evaluate_by_segment


class TestScalarMetrics:
    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert rmse(y_true, y_pred) == 1.0

    def test_mae_perfect(self):
        y = np.array([5.0, 10.0])
        assert mae(y, y) == 0.0

    def test_mae_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 4.0, 6.0])
        assert mae(y_true, y_pred) == 2.0

    def test_bias_positive(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert bias(y_true, y_pred) == 1.0  # over-predicting

    def test_bias_negative(self):
        y_true = np.array([2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert bias(y_true, y_pred) == -1.0  # under-predicting

    def test_rmsle_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmsle(y, y) == 0.0

    def test_rmsle_clips_negative(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([-1.0, 1.0])
        # Should not raise, negative clipped to 0
        result = rmsle(y_true, y_pred)
        assert result >= 0


class TestEvaluate:
    def test_returns_all_metrics(self):
        y = np.array([1.0, 2.0, 3.0])
        result = evaluate(y, y)
        assert "rmse" in result
        assert "mae" in result
        assert "bias" in result
        assert "rmsle" in result


class TestSegmentEvaluation:
    def test_segment_breakdown(self):
        df = pd.DataFrame({
            "actual": [1, 2, 3, 4, 5, 6],
            "predicted": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "segment": ["A", "A", "A", "B", "B", "B"],
        })
        result = evaluate_by_segment(df, "actual", "predicted", "segment")
        assert len(result) == 2
        assert "rmse" in result.columns
        assert "segment" in result.columns
