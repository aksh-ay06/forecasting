from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
PRODUCT_CSV = DATA_DIR / "producto_tabla.csv"
TOWN_CSV = DATA_DIR / "town_state.csv"
CLIENT_CSV = DATA_DIR / "cliente_tabla.csv"

TRAIN_DTYPES = {
    "Semana": "uint8",
    "Agencia_ID": "uint16",
    "Canal_ID": "uint8",
    "Ruta_SAK": "uint16",
    "Cliente_ID": "int64",
    "Producto_ID": "uint16",
    "Venta_uni_hoy": "int32",
    "Venta_hoy": "float32",
    "Dev_uni_proxima": "int32",
    "Dev_proxima": "float32",
    "Demanda_uni_equil": "int32",
}

TEST_DTYPES = {
    "id": "int32",
    "Semana": "uint8",
    "Agencia_ID": "uint16",
    "Canal_ID": "uint8",
    "Ruta_SAK": "uint16",
    "Cliente_ID": "int64",
    "Producto_ID": "uint16",
}

CHUNK_SIZE = 5_000_000

TRAIN_SEMANAS = list(range(3, 10))
VAL_SEMANA = 9
TEST_SEMANAS = [10, 11]
TARGET_COL = "Demanda_uni_equil"

LAG_WEEKS = [1, 2, 3, 4, 5]
ROLLING_WINDOWS = [2, 3, 4]
AGG_GROUP_KEYS = {
    "product": ["Producto_ID"],
    "client": ["Cliente_ID"],
    "route": ["Ruta_SAK"],
    "agency": ["Agencia_ID"],
    "client_product": ["Cliente_ID", "Producto_ID"],
    "route_product": ["Ruta_SAK", "Producto_ID"],
    "agency_product": ["Agencia_ID", "Producto_ID"],
}

LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}
LGB_NUM_BOOST_ROUND = 1500
LGB_EARLY_STOPPING_ROUNDS = 50

LGB_QUANTILE_PARAMS_LO = {**LGB_PARAMS, "objective": "quantile", "alpha": 0.1}
LGB_QUANTILE_PARAMS_HI = {**LGB_PARAMS, "objective": "quantile", "alpha": 0.9}
LGB_QUANTILE_PARAMS_LO["metric"] = "quantile"
LGB_QUANTILE_PARAMS_HI["metric"] = "quantile"

BACKOFF_LEVELS = [
    ("Cliente_ID", "Producto_ID"),
    ("Ruta_SAK", "Producto_ID"),
    ("Agencia_ID", "Producto_ID"),
    ("Producto_ID",),
    ("brand",),
    (),
]

API_HOST = "0.0.0.0"
API_PORT = 8000

MLFLOW_EXPERIMENT = "grupo_bimbo_demand"
MLFLOW_TRACKING_URI = "mlruns"
