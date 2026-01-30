"""Merge metadata, label-encode, and save per-Semana parquet files."""

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import PROCESSED_DIR, FEATURES_DIR
from src.data.loader import load_train_chunked, load_metadata, save_parquet_by_semana
from src.data.metadata import enrich_product_table

logger = logging.getLogger(__name__)


def preprocess_and_save() -> None:
    """Full preprocessing pipeline.

    1. Load raw train CSV.
    2. Load + parse product metadata, load town metadata.
    3. Merge metadata onto transactions.
    4. Label-encode State.
    5. Save per-Semana parquet files.
    6. Save enriched product table for feature reuse.
    """
    # 1. Load raw data
    train = load_train_chunked()

    # 2. Load metadata
    products, towns, _clients = load_metadata()
    products = enrich_product_table(products)

    # 3. Merge
    logger.info("Merging product metadata …")
    train = train.merge(
        products[["Producto_ID", "brand", "weight_g", "pieces", "has_promo"]],
        on="Producto_ID",
        how="left",
    )

    logger.info("Merging town/state metadata …")
    train = train.merge(towns, on="Agencia_ID", how="left")

    # 4. Label-encode State
    le = LabelEncoder()
    train["State"] = train["State"].fillna("UNKNOWN")
    train["State_encoded"] = le.fit_transform(train["State"]).astype("uint8")
    train.drop(columns=["State", "Town"], inplace=True)

    # Drop the product name column (not needed downstream)
    if "NombreProducto" in train.columns:
        train.drop(columns=["NombreProducto"], inplace=True)

    # 5. Save parquet partitioned by Semana
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet_by_semana(train, PROCESSED_DIR)

    # 6. Save enriched product table
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    products.to_parquet(FEATURES_DIR / "product_metadata.parquet", index=False)

    # Save label encoder mapping for State
    state_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    pd.DataFrame(list(state_mapping.items()), columns=["State", "State_encoded"]).to_parquet(
        FEATURES_DIR / "state_mapping.parquet", index=False
    )

    logger.info("Preprocessing complete. Parquet files saved to %s", PROCESSED_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    preprocess_and_save()
