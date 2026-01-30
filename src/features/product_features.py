import logging

import pandas as pd

from src.config import FEATURES_DIR

logger = logging.getLogger(__name__)


def load_product_features() -> pd.DataFrame:
    path = FEATURES_DIR / "product_metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Product metadata not found at {path}. Run preprocessing first."
        )

    products = pd.read_parquet(path)

    encoding_path = FEATURES_DIR / "brand_encoding.parquet"
    if encoding_path.exists():
        brand_enc = pd.read_parquet(encoding_path)
        products = products.merge(brand_enc, on="brand", how="left")
        products["brand_encoded"] = products["brand_encoded"].fillna(
            brand_enc["brand_encoded"].max() + 1
        ).astype("uint16")
    else:
        raise FileNotFoundError(
            f"Brand encoding not found at {encoding_path}. Run preprocessing first."
        )

    result = products[["Producto_ID", "brand_encoded", "weight_g", "pieces", "has_promo"]].copy()
    result["weight_g"] = result["weight_g"].astype("float32")
    result["pieces"] = result["pieces"].astype("uint8")
    result["has_promo"] = result["has_promo"].astype("uint8")

    logger.info("Product features: %d products, %d unique brands",
                len(result), result["brand_encoded"].nunique())
    return result
