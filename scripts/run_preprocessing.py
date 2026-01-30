#!/usr/bin/env python
"""Run the data preprocessing pipeline."""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import preprocess_and_save


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    preprocess_and_save()


if __name__ == "__main__":
    main()
