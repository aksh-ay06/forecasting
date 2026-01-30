import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import PROCESSED_DIR, FEATURES_DIR
from src.data.loader import load_train_chunked, load_metadata, save_parquet_by_semana
from src.data.metadata import enrich_product_table

logger = logging.getLogger(__name__)


def preprocess_and_save() -> None:
    train = load_train_chunked()

    products, towns, _clients = load_metadata()
    products = enrich_product_table(products)

    logger.info("Merging product metadata …")
    train = train.merge(
        products[["Producto_ID", "brand", "weight_g", "pieces", "has_promo"]],
        on="Producto_ID",
        how="left",
    )

    logger.info("Merging town/state metadata …")
    train = train.merge(towns, on="Agencia_ID", how="left")

    le = LabelEncoder()
    train["State"] = train["State"].fillna("UNKNOWN")
    n_states = train["State"].nunique()
    state_dtype = "uint8" if n_states <= 255 else "uint16"
    train["State_encoded"] = le.fit_transform(train["State"]).astype(state_dtype)
    train.drop(columns=["State", "Town"], inplace=True)

    if "NombreProducto" in train.columns:
        train.drop(columns=["NombreProducto"], inplace=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_parquet_by_semana(train, PROCESSED_DIR)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    products.to_parquet(FEATURES_DIR / "product_metadata.parquet", index=False)

    products[["Producto_ID", "brand"]].to_parquet(
        FEATURES_DIR / "product_brand_map.parquet", index=False
    )

    brand_le = LabelEncoder()
    brand_le.fit(products["brand"].fillna("UNK"))
    brand_enc_df = pd.DataFrame({
        "brand": brand_le.classes_,
        "brand_encoded": range(len(brand_le.classes_)),
    })
    brand_enc_df.to_parquet(FEATURES_DIR / "brand_encoding.parquet", index=False)

    state_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    pd.DataFrame(list(state_mapping.items()), columns=["State", "State_encoded"]).to_parquet(
        FEATURES_DIR / "state_mapping.parquet", index=False
    )

    logger.info("Preprocessing complete. Parquet files saved to %s", PROCESSED_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    preprocess_and_save()
