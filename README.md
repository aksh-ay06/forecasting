# Grupo Bimbo Demand Forecasting

Demand forecasting system for Grupo Bimbo's bakery product distribution across Mexico. Predicts weekly demand (`Demanda_uni_equil`) at the (client, product, route, week) level using 74M+ historical transactions.

## Business Context

Grupo Bimbo delivers products to over 1 million stores weekly via 45,000+ routes. Accurate demand forecasting reduces unsold product returns (waste) and prevents stockouts. The system forecasts demand for ~2,600 products across ~900K clients.

## Architecture

```
data/ (raw CSVs) → preprocessing → parquet (partitioned by Semana)
                                       ↓
                              feature engineering
                          (lags, aggregations, backoff)
                                       ↓
                           LightGBM training + MLflow
                                       ↓
                        FastAPI serving ← model artifacts
```

Key design decisions:
- **Optimized dtypes** reduce 74M rows from ~6GB to ~2.4GB in memory
- **Parquet partitioned by Semana** for efficient temporal access
- **Strict temporal isolation**: features for week S use only data from weeks < S
- **Hierarchical backoff** handles new/unseen client-product combinations
- **Confidence intervals** via quantile regression (10th/90th percentile models)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Place raw CSVs in data/
# train.csv, test.csv, producto_tabla.csv, town_state.csv, cliente_tabla.csv
```

## Pipeline

```bash
# 1. Preprocess: load CSVs, merge metadata, save parquet
make preprocess

# 2. Train: feature engineering, LightGBM, MLflow logging
make train

# 3. Evaluate: metrics, plots, SHAP analysis
make evaluate

# 4. Inference: batch predictions on test.csv
make inference

# 5. Serve: start API
make serve

# 6. Tests
make test
```

## Features (~44 total)

| Category | Features | Description |
|----------|----------|-------------|
| Lag | lag_1..lag_5 | Past week demand at (client, product) |
| Rolling | mean/std/median × windows 2-4 | Demand trends and volatility |
| Aggregation | mean/std/count × 7 entity groups | Product, client, route, agency, and cross-level |
| Product | brand, weight, pieces, promo | Parsed from product names |
| Backoff | level + estimate | Hierarchical fallback for unseen entities |

## API Usage

```bash
# Start server
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Semana": 10, "Agencia_ID": 1110, "Canal_ID": 7, "Ruta_SAK": 3301, "Cliente_ID": 15766, "Producto_ID": 1212}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"Semana": 10, "Agencia_ID": 1110, "Canal_ID": 7, "Ruta_SAK": 3301, "Cliente_ID": 15766, "Producto_ID": 1212}]}'

# Health check
curl http://localhost:8000/health
```

## Docker

```bash
# Build and run (requires trained model in artifacts/)
make docker-build
make docker-up
# API available at http://localhost:8000
```

## Project Structure

```
src/
├── config.py              # Paths, dtypes, hyperparameters
├── data/                  # Loading, parsing, preprocessing
├── features/              # Lag, aggregation, product, backoff features
├── models/                # Baselines, LightGBM, training orchestration
├── evaluation/            # Metrics, validation, analysis, SHAP
├── serving/               # FastAPI application
└── monitoring/            # Drift detection
```
