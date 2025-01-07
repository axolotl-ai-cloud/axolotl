"""Shared module for CLI specific utilities."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

import axolotl.monkeypatch.data.batch_dataset_fetcher  # pylint: disable=unused-import  # noqa: F401
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer

configure_logging()
LOG = logging.getLogger(__name__)


@dataclass
class PreprocessCliArgs:
    """Dataclass with CLI arguments for `axolotl preprocess` command."""

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=1)
    prompter: Optional[str] = field(default=None)
    download: Optional[bool] = field(default=True)


@dataclass
class TrainerCliArgs:
    """Dataclass with CLI arguments for `axolotl train` command."""

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=0)
    merge_lora: bool = field(default=False)
    prompter: Optional[str] = field(default=None)
    shard: bool = field(default=False)


@dataclass
class EvaluateCliArgs:
    """Dataclass with CLI arguments for `axolotl evaluate` command."""

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=0)


@dataclass
class InferenceCliArgs:
    """Dataclass with CLI arguments for `axolotl inference` command."""

    prompter: Optional[str] = field(default=None)


def load_model_and_tokenizer(
    *,
    cfg: DictDefault,
    inference: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer | PreTrainedTokenizerFast | Any]:
    """
    Helper function for loading a model and tokenizer specified in the given `axolotl`
    config.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        inference: Boolean denoting inference mode.

    Returns:
        `transformers` model and tokenizer.
    """
    LOG.info(f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}")
    tokenizer = load_tokenizer(cfg)

    LOG.info("loading model...")
    model, _ = load_model(cfg, tokenizer, inference=inference)

    return model, tokenizer
