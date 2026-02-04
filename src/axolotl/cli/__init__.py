"""Axolotl CLI module initialization."""

import os

from axolotl.logging_config import configure_logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

configure_logging()
