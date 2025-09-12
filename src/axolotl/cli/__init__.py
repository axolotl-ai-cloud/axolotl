"""Axolotl CLI module initialization."""

import os

from axolotl.logging_config import configure_logging
from axolotl.utils.tee import start_output_buffering

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Start teeing terminal output early; finalize once cfg.output_dir is known
start_output_buffering()

configure_logging()
