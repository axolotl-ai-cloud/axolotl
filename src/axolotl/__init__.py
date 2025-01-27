"""Axolotl - Train and fine-tune large language models"""

import pkgutil

from .cli.config import choose_config, load_cfg, validate_config
from .datasets import ConstantLengthDataset, TokenizedPromptDataset
from .evaluate import evaluate
from .train import train

__path__ = pkgutil.extend_path(__path__, __name__)  # Make this a namespace package
__version__ = "0.6.0"

__all__ = [
    "train",
    "evaluate",
    "TokenizedPromptDataset",
    "ConstantLengthDataset",
    "load_cfg",
    "choose_config",
    "validate_config",
]
