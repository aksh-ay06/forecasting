import logging
from pathlib import Path

import pandas as pd

from src.config import CHUNK_SIZE, PROCESSED_DIR, TRAIN_CSV, TRAIN_DTYPES, TEST_CSV, TEST_DTYPES

logger = logging.getLogger(__name__)


def load_train_chunked(csv_path: Path = TRAIN_CSV,
                       chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    logger.info("Loading %s in %d-row chunks …", csv_path, chunk_size)
    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(csv_path, dtype=TRAIN_DTYPES, chunksize=chunk_size)
    ):
        chunks.append(chunk)
        if (i + 1) % 5 == 0:
            logger.info("  … read %d chunks (%d rows)", i + 1, (i + 1) * chunk_size)
    df = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %d rows, %.2f GB", len(df), df.memory_usage(deep=True).sum() / 1e9)
    return df


def iter_train_chunks(csv_path: Path = TRAIN_CSV,
                      chunk_size: int = CHUNK_SIZE):
    for chunk in pd.read_csv(csv_path, dtype=TRAIN_DTYPES, chunksize=chunk_size):
        yield chunk


def load_test(csv_path: Path = TEST_CSV) -> pd.DataFrame:
    logger.info("Loading %s …", csv_path)
    return pd.read_csv(csv_path, dtype=TEST_DTYPES)


def save_parquet_by_semana(df: pd.DataFrame, output_dir: Path = PROCESSED_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for semana, group in df.groupby("Semana"):
        path = output_dir / f"semana_{semana}.parquet"
        group.to_parquet(path, engine="pyarrow", index=False)
        logger.info("Saved %s (%d rows)", path, len(group))


def load_parquet_semanas(semanas: list[int],
                         input_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    dfs = []
    for s in semanas:
        path = input_dir / f"semana_{s}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found: {path}. Run preprocessing first.")
        dfs.append(pd.read_parquet(path, engine="pyarrow"))
    return pd.concat(dfs, ignore_index=True)


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from src.config import PRODUCT_CSV, TOWN_CSV, CLIENT_CSV
    products = pd.read_csv(PRODUCT_CSV)
    towns = pd.read_csv(TOWN_CSV)
    clients = pd.read_csv(CLIENT_CSV)
    logger.info("Loaded metadata: %d products, %d towns, %d clients",
                len(products), len(towns), len(clients))
    return products, towns, clients
