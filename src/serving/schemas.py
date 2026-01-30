from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    Semana: int = Field(..., ge=3, le=15)
    Agencia_ID: int = Field(..., ge=0)
    Canal_ID: int = Field(..., ge=0)
    Ruta_SAK: int = Field(..., ge=0)
    Cliente_ID: int = Field(..., ge=0)
    Producto_ID: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    predicted_demand: float
    confidence_lower: float
    confidence_upper: float


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False
    n_features: int = 0
