"""
This module defines the BatchEval class, which handles batch evaluation of a model.
"""
import logging
from typing import Dict, Any
import json

import click

from axolotl import cfg
from axolotl.cli.options import (
    dataset_prepared_path_option,
    max_packed_sequence_len_option,
    micro_batch_size_option,
    pretraining_dataset_option,
    seed_option,
    dataset_option,
    sequence_len_option,
    split_name_option,
    train_on_inputs_option,
)
from axolotl.cli.option_groups import model_option_group
from axolotl.utils.config import update_config

LOG = logging.getLogger("axolotl")


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

    # TODO: writeme
