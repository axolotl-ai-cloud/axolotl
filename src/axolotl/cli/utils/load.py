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
from axolotl.utils.chat_templates import get_chat_template_from_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def resolve_chat_template_str(
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | Any,
) -> str | None:
    """
    Resolves the chat template string for inference from the `axolotl` config,
    mirroring how it would be resolved at training time: an explicit
    `chat_template` config takes precedence, then the first dataset's
    `chat_template` if that dataset is of type `chat_template`.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        tokenizer: Tokenizer to fall back to for tokenizer-default templates.

    Returns:
        Chat template string, or None if the config does not specify one.
    """
    if cfg.chat_template:
        return get_chat_template_from_config(cfg, ds_cfg=None, tokenizer=tokenizer)
    if cfg.datasets and cfg.datasets[0].type == "chat_template":
        return get_chat_template_from_config(
            cfg=cfg, ds_cfg=cfg.datasets[0], tokenizer=tokenizer
        )
    return None


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

    processor = None
    if cfg.is_multimodal:
        LOG.info("loading processor...")
        processor = load_processor(cfg, tokenizer)

    LOG.info("loading model...")
    model_loader = ModelLoader(
        cfg,
        tokenizer,
        processor=processor,
        inference=inference,
    )
    model, _ = model_loader.load()

    return model, tokenizer, processor
