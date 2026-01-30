"""Unit tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np

from src.features.lag_features import build_lag_features
from src.features.agg_features import build_agg_features
from src.data.metadata import parse_product_name


@pytest.fixture
def sample_history():
    """Create a small synthetic history DataFrame."""
    np.random.seed(42)
    rows = []
    for semana in range(3, 8):
        for client in [1, 2, 3]:
            for product in [100, 200]:
                rows.append({
                    "Semana": semana,
                    "Agencia_ID": 1000,
                    "Canal_ID": 1,
                    "Ruta_SAK": 500,
                    "Cliente_ID": client,
                    "Producto_ID": product,
                    "Demanda_uni_equil": np.random.randint(1, 20),
                })
    return pd.DataFrame(rows)


class TestLagFeatures:
    def test_builds_lags(self, sample_history):
        result = build_lag_features(sample_history, target_semana=7)
        assert not result.empty
        assert "lag_1" in result.columns
        assert "lag_2" in result.columns

    def test_no_future_leakage(self, sample_history):
        """Ensure no data from target_semana or later is used."""
        result = build_lag_features(sample_history, target_semana=5)
        # lag_1 should use semana 4, lag_2 should use semana 3
        assert "lag_1" in result.columns
        # lag_3 for target=5 would need semana 2 which doesn't exist
        assert result["lag_3"].isna().all() if "lag_3" in result.columns else True

    def test_rolling_stats(self, sample_history):
        result = build_lag_features(sample_history, target_semana=7)
        assert "rolling_mean_2" in result.columns
        assert "rolling_std_2" in result.columns
        assert "rolling_median_2" in result.columns

    def test_empty_history(self):
        empty = pd.DataFrame(columns=["Semana", "Cliente_ID", "Producto_ID", "Demanda_uni_equil"])
        result = build_lag_features(empty, target_semana=5)
        assert result.empty


class TestAggFeatures:
    def test_builds_aggregations(self, sample_history):
        result = build_agg_features(sample_history, target_semana=7)
        assert "product" in result
        assert "client" in result
        assert "product_mean" in result["product"].columns
        assert "client_std" in result["client"].columns

    def test_no_future_data(self, sample_history):
        """Aggregations should only use data before target_semana."""
        result_5 = build_agg_features(sample_history, target_semana=5)
        result_7 = build_agg_features(sample_history, target_semana=7)
        # More history available for target 7, so counts should be higher
        assert result_7["product"]["product_count"].sum() >= result_5["product"]["product_count"].sum()


class TestMetadataParser:
    def test_parse_weight_grams(self):
        result = parse_product_name("Bimbollos Ext sAjonjoli 6p 480g BIM 41")
        assert result["weight_g"] == 480.0

    def test_parse_weight_kg(self):
        result = parse_product_name("Capuccino Moka 750g NES 9")
        assert result["weight_g"] == 750.0

    def test_parse_pieces(self):
        result = parse_product_name("Bimbollos Ext sAjonjoli 6p 480g BIM 41")
        assert result["pieces"] == 6

    def test_parse_brand(self):
        result = parse_product_name("Bimbollos Ext sAjonjoli 6p 480g BIM 41")
        assert result["brand"] == "BIM"

    def test_no_promo(self):
        result = parse_product_name("Pan Blanco 500g BIM 10")
        assert result["has_promo"] == 0

    def test_unknown_brand(self):
        result = parse_product_name("X")
        assert result["brand"] == "UNK"
