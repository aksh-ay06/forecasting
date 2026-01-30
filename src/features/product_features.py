"""Product metadata features from parsed product names."""

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import FEATURES_DIR

logger = logging.getLogger(__name__)


def load_product_features() -> pd.DataFrame:
    """Load pre-computed product metadata features.

    Returns DataFrame with columns:
        Producto_ID, brand_encoded, weight_g, pieces, has_promo
    """
    path = FEATURES_DIR / "product_metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Product metadata not found at {path}. Run preprocessing first."
        )

    products = pd.read_parquet(path)

    # Label-encode brand
    le = LabelEncoder()
    products["brand_encoded"] = le.fit_transform(
        products["brand"].fillna("UNK")
    ).astype("uint16")

    result = products[["Producto_ID", "brand_encoded", "weight_g", "pieces", "has_promo"]].copy()
    result["weight_g"] = result["weight_g"].astype("float32")
    result["pieces"] = result["pieces"].astype("uint8")
    result["has_promo"] = result["has_promo"].astype("uint8")

    # Also save brand mapping for backoff
    brand_map = products[["Producto_ID", "brand"]].copy()
    brand_map.to_parquet(FEATURES_DIR / "product_brand_map.parquet", index=False)

    logger.info("Product features: %d products, %d unique brands",
                len(result), result["brand_encoded"].nunique())
    return result
