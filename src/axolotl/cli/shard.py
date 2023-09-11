"""
CLI to shard a trained model into 10GiB chunks
"""
import logging
from pathlib import Path

import fire
import transformers

from axolotl.cli import load_cfg, print_axolotl_text_art
from axolotl.common.cli import TrainerCliArgs, load_model_and_tokenizer
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.scripts")


def shard(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
):
    model, _ = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
    safe_serialization = cfg.save_safetensors is True
    LOG.debug("Re-saving model w/ sharding")
    model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)


def do_cli(config: Path = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parsed_cli_args.shard = True

    shard(cfg=parsed_cfg, cli_args=parsed_cli_args)


fire.Fire(do_cli)
