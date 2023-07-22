"""
This module defines the BatchEval class, which handles batch evaluation of a model.
"""
import logging
from typing import Any, Dict

import click

from axolotl import cfg
from axolotl.cli.option_groups import model_option_group
from axolotl.cli.options import (
    dataset_option,
    dataset_prepared_path_option,
    max_packed_sequence_len_option,
    micro_batch_size_option,
    pretraining_dataset_option,
    seed_option,
    sequence_len_option,
    split_name_option,
    train_on_inputs_option,
)
from axolotl.eval import BatchEval
from axolotl.utils.config import startup_load_dataset, update_config
from axolotl.utils.models import load_model, load_tokenizer

LOG = logging.getLogger(__name__)


@click.group(name="eval")
def eval_group():
    """Axolotl evaluation tools"""


@eval_group.command(name="batch")
@seed_option()
@dataset_option()
@model_option_group()
@train_on_inputs_option()
@micro_batch_size_option()
@pretraining_dataset_option()
@dataset_prepared_path_option()
@max_packed_sequence_len_option()
@sequence_len_option()
@split_name_option()
def batch(**kwargs: Dict[str, Any]):
    """Executes a batch evaluation operation"""

    # Override default configuration
    update_config(overrides=kwargs)

    # Load the tokenizer
    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    LOG.info("Loading tokenizer: %s", tokenizer_config)
    tokenizer = load_tokenizer(tokenizer_config, cfg.tokenizer_type, cfg)

    # Load dataset
    dataset, _ = startup_load_dataset(cfg, tokenizer)

    # Load the model
    LOG.info("Loading model and peft_config: %s", cfg.base_model)
    model, _ = load_model(
        base_model=cfg.base_model,
        base_model_config=cfg.base_model_config,
        model_type=cfg.model_type,
        tokenizer=tokenizer,
        cfg=cfg,
        adapter=cfg.adapter,
    )

    cli_handler = BatchEval(cfg=cfg, model=model, tokenizer=tokenizer, dataset=dataset)
    cli_handler.validate_and_warn()
    cli_handler.run()
