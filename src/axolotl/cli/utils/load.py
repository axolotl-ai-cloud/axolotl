"""Utilities for model, tokenizer, etc. loading."""

from typing import Any

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    ProcessorMixin,
)

from axolotl.loaders import load_processor, load_tokenizer
from axolotl.loaders.model import ModelLoader
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def load_model_and_tokenizer(
    *,
    cfg: DictDefault,
    inference: bool = False,
) -> tuple[
    PreTrainedModel,
    PreTrainedTokenizer | PreTrainedTokenizerFast | Any,
    ProcessorMixin | None,
]:
    """
    Helper function for loading a model, tokenizer, and processor specified in the
    given `axolotl` config.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        inference: Boolean denoting inference mode.

    Returns:
        Tuple of (PreTrainedModel, PreTrainedTokenizer, ProcessorMixin).
    """
    LOG.info(f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}")
    tokenizer = load_tokenizer(cfg)

    LOG.info("loading model...")
    model_loader = ModelLoader(cfg, tokenizer, inference=inference)
    model, _ = model_loader.load()

    processor = None
    if cfg.is_multimodal:
        LOG.info("loading processor...")
        processor = load_processor(cfg, tokenizer)

    return model, tokenizer, processor
