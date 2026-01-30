import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(g|kg|ml|l)\b", re.IGNORECASE)
_PIECES_RE = re.compile(r"(\d+)\s*p\b", re.IGNORECASE)


def parse_product_name(name: str) -> dict:
    tokens = name.strip().split()
    brand = tokens[-2] if len(tokens) >= 2 else "UNK"

    weight_g = 0.0
    m = _WEIGHT_RE.search(name)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit in ("kg", "l"):
            val *= 1000
        weight_g = val

    pieces = 1
    m = _PIECES_RE.search(name)
    if m:
        pieces = int(m.group(1))

    has_promo = int(bool(re.search(r"promo|oferta|desc|gratis", name, re.IGNORECASE)))

    return {
        "brand": brand,
        "weight_g": weight_g,
        "pieces": pieces,
        "has_promo": has_promo,
    }


def enrich_product_table(products: pd.DataFrame) -> pd.DataFrame:
    parsed = products["NombreProducto"].apply(parse_product_name)
    parsed_df = pd.DataFrame(parsed.tolist())
    result = pd.concat([products, parsed_df], axis=1)
    logger.info("Parsed %d product names → %d brands, mean weight %.0f g",
                len(result), result["brand"].nunique(), result["weight_g"].mean())
    return result
