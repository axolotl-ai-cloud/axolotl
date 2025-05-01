"""Axolotl CLI module initialization."""

import os

from axolotl.logging_config import configure_logging

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
configure_logging()
