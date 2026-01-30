import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.serving.app import app, predictor


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_predictor():
    predictor.model = MagicMock()
    predictor.feature_cols = ["f0", "f1", "f2"]
    predictor.history = MagicMock()

    with patch.object(predictor, "predict_single", return_value=(7.5, 3.0, 12.0)):
        with patch.object(predictor, "predict_batch", return_value=[(7.5, 3.0, 12.0), (5.0, 2.0, 8.0)]):
            yield


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_predict_single(self, client):
        payload = {
            "Semana": 10,
            "Agencia_ID": 1110,
            "Canal_ID": 7,
            "Ruta_SAK": 3301,
            "Cliente_ID": 15766,
            "Producto_ID": 1212,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_demand" in data
        assert "confidence_lower" in data
        assert "confidence_upper" in data
        assert data["confidence_lower"] <= data["predicted_demand"] <= data["confidence_upper"]

    def test_predict_invalid_semana(self, client):
        payload = {
            "Semana": 0,
            "Agencia_ID": 1110,
            "Canal_ID": 7,
            "Ruta_SAK": 3301,
            "Cliente_ID": 15766,
            "Producto_ID": 1212,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    def test_predict_batch(self, client):
        payload = {
            "items": [
                {
                    "Semana": 10,
                    "Agencia_ID": 1110,
                    "Canal_ID": 7,
                    "Ruta_SAK": 3301,
                    "Cliente_ID": 15766,
                    "Producto_ID": 1212,
                },
                {
                    "Semana": 10,
                    "Agencia_ID": 1110,
                    "Canal_ID": 7,
                    "Ruta_SAK": 3301,
                    "Cliente_ID": 15766,
                    "Producto_ID": 1216,
                },
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2
