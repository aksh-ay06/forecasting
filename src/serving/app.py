import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.serving.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse,
)
from src.serving.predictor import Predictor

logger = logging.getLogger(__name__)

predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Loading model and feature stores …")
    try:
        predictor.load()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
    yield


app = FastAPI(
    title="Grupo Bimbo Demand Forecasting API",
    description="Predicts weekly demand for bakery products across Mexican stores.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        pred, lower, upper = predictor.predict_single(request.model_dump())
        return PredictionResponse(
            predicted_demand=round(pred, 2),
            confidence_lower=round(lower, 2),
            confidence_upper=round(upper, 2),
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        items = [item.model_dump() for item in request.items]
        results = predictor.predict_batch(items)
        predictions = [
            PredictionResponse(
                predicted_demand=round(p, 2),
                confidence_lower=round(lo, 2),
                confidence_upper=round(hi, 2),
            )
            for p, lo, hi in results
        ]
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        n_features=len(predictor.feature_cols),
    )
