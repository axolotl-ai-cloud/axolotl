"""
shared module for cli specific things
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import axolotl.monkeypatch.data.batch_dataset_fetcher  # pylint: disable=unused-import  # noqa: F401
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_tokenizer

configure_logging()
LOG = logging.getLogger("axolotl.common.cli")


@dataclass
class PreprocessCliArgs:
    """
    dataclass with arguments for preprocessing only
    """

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=1)
    prompter: Optional[str] = field(default=None)
    download: Optional[bool] = field(default=True)


@dataclass
class TrainerCliArgs:
    """
    dataclass with various non-training arguments
    """

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=0)
    inference: bool = field(default=False)
    merge_lora: bool = field(default=False)
    prompter: Optional[str] = field(default=None)
    shard: bool = field(default=False)


@dataclass
class EvaluateCliArgs:
    """
    dataclass with various evaluation arguments
    """

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=0)


@dataclass
class ConvertDiffTransformerCliArgs:
    """
    dataclass with arguments for convert-diff-transformer CLI
    """

    debug: bool = field(default=False)
    zero_init: bool = field(default=False)
    sublayer_norm: bool = field(default=True)


def load_model_and_tokenizer(
    *,
    cfg: DictDefault,
    cli_args: Union[TrainerCliArgs, EvaluateCliArgs, ConvertDiffTransformerCliArgs],
):
    LOG.info(f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}")
    tokenizer = load_tokenizer(cfg)

    LOG.info("loading model and (optionally) peft_config...")
    inference = getattr(cli_args, "inference", False)
    model, _ = load_model(cfg, tokenizer, inference=inference)

    return model, tokenizer
