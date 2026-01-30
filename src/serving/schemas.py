"""Pydantic request/response models for the prediction API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Single prediction request."""
    Semana: int = Field(..., ge=3, le=15, description="Week number")
    Agencia_ID: int = Field(..., ge=0, description="Agency/warehouse ID")
    Canal_ID: int = Field(..., ge=0, description="Sales channel ID")
    Ruta_SAK: int = Field(..., ge=0, description="Route ID")
    Cliente_ID: int = Field(..., ge=0, description="Client ID")
    Producto_ID: int = Field(..., ge=0, description="Product ID")


class PredictionResponse(BaseModel):
    """Single prediction response."""
    predicted_demand: float = Field(..., description="Predicted demand units")
    confidence_lower: float = Field(..., description="Lower bound (10th percentile)")
    confidence_upper: float = Field(..., description="Upper bound (90th percentile)")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    items: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    n_features: int = 0
