"""
shared module for cli specific things
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer

configure_logging()
LOG = logging.getLogger("axolotl.common.cli")


@dataclass
class TrainerCliArgs:
    """
    dataclass representing the various non-training arguments
    """

    debug: bool = field(default=False)
    inference: bool = field(default=False)
    merge_lora: bool = field(default=False)
    prepare_ds_only: bool = field(default=False)
    prompter: Optional[str] = field(default=None)
    shard: bool = field(default=False)


def load_model_and_tokenizer(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
):
    LOG.info(f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}")
    tokenizer = load_tokenizer(cfg)
    LOG.info("loading model and (optionally) peft_config...")
    model, _ = load_model(cfg, tokenizer, inference=cli_args.inference)

    return model, tokenizer
