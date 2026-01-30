"""Parse product names to extract brand, weight, pieces, and promo flag."""

import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Pattern: "Product Name <weight><unit> <BRAND> <ID>"
# Examples:
#   "Bimbollos Ext sAjonjoli 6p 480g BIM 41"
#   "Capuccino Moka 750g NES 9"
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(g|kg|ml|l)\b", re.IGNORECASE)
_PIECES_RE = re.compile(r"(\d+)\s*p\b", re.IGNORECASE)


def parse_product_name(name: str) -> dict:
    """Extract structured fields from a product name string.

    Returns dict with keys: brand, weight_g, pieces, has_promo.
    """
    tokens = name.strip().split()
    # Brand is the second-to-last token (last is Producto_ID)
    brand = tokens[-2] if len(tokens) >= 2 else "UNK"

    # Weight
    weight_g = 0.0
    m = _WEIGHT_RE.search(name)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit in ("kg", "l"):
            val *= 1000
        weight_g = val

    # Pieces
    pieces = 1
    m = _PIECES_RE.search(name)
    if m:
        pieces = int(m.group(1))

    # Promo flag (heuristic: name contains promo-related keywords)
    has_promo = int(bool(re.search(r"promo|oferta|desc|gratis", name, re.IGNORECASE)))

    return {
        "brand": brand,
        "weight_g": weight_g,
        "pieces": pieces,
        "has_promo": has_promo,
    }


def enrich_product_table(products: pd.DataFrame) -> pd.DataFrame:
    """Add parsed columns to the producto_tabla DataFrame.

    Parameters
    ----------
    products : DataFrame with columns [Producto_ID, NombreProducto]

    Returns
    -------
    DataFrame with added columns: brand, weight_g, pieces, has_promo
    """
    parsed = products["NombreProducto"].apply(parse_product_name)
    parsed_df = pd.DataFrame(parsed.tolist())
    result = pd.concat([products, parsed_df], axis=1)
    logger.info("Parsed %d product names → %d brands, mean weight %.0f g",
                len(result), result["brand"].nunique(), result["weight_g"].mean())
    return result
