"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""
import logging
from pathlib import Path

import fire
import transformers

from axolotl.cli import (
    check_accelerate_default_config,
    do_inference,
    do_merge_lora,
    load_cfg,
    load_datasets,
    print_axolotl_text_art,
)
from axolotl.cli.shard import shard
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train

LOG = logging.getLogger("axolotl.scripts.finetune")


def do_cli(config: Path = Path("examples/"), **kwargs):
    print_axolotl_text_art()
    LOG.warning(
        str(
            PendingDeprecationWarning(
                "scripts/finetune.py will be replaced with calling axolotl.cli.train"
            )
        )
    )
    parsed_cfg = load_cfg(config, **kwargs)
    check_accelerate_default_config()
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    if parsed_cli_args.inference:
        do_inference(cfg=parsed_cfg, cli_args=parsed_cli_args)
    elif parsed_cli_args.merge_lora:
        do_merge_lora(cfg=parsed_cfg, cli_args=parsed_cli_args)
    elif parsed_cli_args.shard:
        shard(cfg=parsed_cfg, cli_args=parsed_cli_args)
    else:
        dataset_meta = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
        if parsed_cli_args.prepare_ds_only:
            return
        train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)


if __name__ == "__main__":
    fire.Fire(do_cli)
