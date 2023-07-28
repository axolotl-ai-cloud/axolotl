"""
This module is a CLI adapter to the Axolotl inferencing capabilities
"""
import json
import logging
from typing import Any, Dict

import click

from axolotl import cfg
from axolotl.cli import CTX_ACCELERATOR
from axolotl.cli.option_groups import model_option_group
from axolotl.cli.options import (
    dataset_option,
    dataset_prepared_path_option,
    generation_config_option,
    landmark_attention_option,
    max_packed_sequence_len_option,
    micro_batch_size_option,
    output_dir_option,
    pretraining_dataset_option,
    seed_option,
    sequence_len_option,
    split_name_option,
    train_on_inputs_option,
)
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.storage import gen_timestamp_file

LOG = logging.getLogger(__name__)


@click.group(name="inference")
def inference_group():
    """Axolotl inferencing tools"""


@inference_group.command(name="batch")
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
@generation_config_option()
@output_dir_option()
@landmark_attention_option()
def batch(**kwargs: Dict[str, Any]):
    """Executes a batch evaluation operation"""

    from axolotl.inference import BatchInference, JsonFilePostProcessor
    from axolotl.utils.config import update_config
    from axolotl.utils.models import load_model, load_tokenizer

    # Override default configuration
    update_config(overrides=kwargs)

    # pylint: disable=R0801
    # Load the tokenizer
    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    LOG.info("Loading tokenizer: %s", tokenizer_config)
    tokenizer = load_tokenizer(tokenizer_config, cfg.tokenizer_type, cfg)

    # Load dataset
    dataset = load_tokenized_prepared_datasets(
        tokenizer, cfg, cfg.dataset_prepared_path
    )

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

    if cfg.landmark_attention:
        from axolotl.monkeypatch.llama_landmark_attn import set_model_mem_id

        LOG.info("Enabling landmark attention")

        set_model_mem_id(model, tokenizer)
        model.set_mem_cache_args(
            max_seq_len=255, mem_freq=50, top_k=5, max_cache_size=None
        )

    cli_handler = BatchInference(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        accelerator=click.get_current_context().meta[CTX_ACCELERATOR],
        post_processors=[
            JsonFilePostProcessor(
                cfg=cfg, filename=gen_timestamp_file(cfg.output_dir, suffix=".json")
            )
        ],
    )
    cli_handler.validate_and_warn()
    response = cli_handler.run()

    # Output a single line of json as the response
    click.echo(json.dumps(response))
