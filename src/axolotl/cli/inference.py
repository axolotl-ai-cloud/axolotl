"""
This module is a CLI adapter to the Axolotl inferencing capabilities
"""
import json
from typing import Any, Dict

import click
from accelerate.logging import get_logger

from axolotl import cfg
from axolotl.cli import CTX_ACCELERATOR
from axolotl.cli.option_groups import model_option_group
from axolotl.cli.options import (
    dataset_option,
    dataset_prepared_path_option,
    generation_config_option,
    landmark_attention_option,
    max_packed_sequence_len_option,
    output_dir_option,
    pretraining_dataset_option,
    seed_option,
    sequence_len_option,
    split_name_option,
    train_on_inputs_option,
    truncate_features_option,
)
from axolotl.utils.data import load_tokenized_prepared_datasets

LOG = get_logger(__name__)


@click.group(name="inference")
def inference_group():
    """Axolotl inferencing tools"""


@inference_group.command(name="batch")
@seed_option()
@dataset_option()
@model_option_group()
@train_on_inputs_option()
@pretraining_dataset_option()
@dataset_prepared_path_option()
@max_packed_sequence_len_option()
@sequence_len_option()
@split_name_option()
@generation_config_option()
@output_dir_option()
@landmark_attention_option()
@truncate_features_option()
def batch(**kwargs: Dict[str, Any]):
    """Executes a batch evaluation operation"""

    from axolotl.inference import BatchInference, JsonFilePostProcessor
    from axolotl.utils.config import update_config
    from axolotl.utils.models import load_model, load_tokenizer

    # Override default configuration
    update_config(overrides=kwargs)

    BatchInference.validate_and_warn(cfg=cfg)

    # pylint: disable=R0801
    # Load the tokenizer
    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    LOG.info("Loading tokenizer: %s", tokenizer_config)
    tokenizer = load_tokenizer(tokenizer_config, cfg.tokenizer_type, cfg)

    accelerator = click.get_current_context().meta[CTX_ACCELERATOR]

    # Load dataset
    if accelerator.main_process_first():
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
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        accelerator=accelerator,
        seed=cfg.seed,
        output_dir=cfg.output_dir,
        generation_config=cfg.generation_config,
        post_processors=[JsonFilePostProcessor(output_dir=cfg.output_dir)],
    )
    response = cli_handler.run()

    # Output a single line of json as the response
    click.echo(json.dumps(response))
