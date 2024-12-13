"""
CLI to run training on a model
"""
import logging
from pathlib import Path
from typing import Union

import fire
from dotenv import load_dotenv
from transformers.hf_argparser import HfArgumentParser

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    load_rl_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.evaluate import evaluate

LOG = logging.getLogger("axolotl.cli.train")


def do_evaluate(cfg, cli_args) -> None:
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()

    if cfg.rl:  # and cfg.rl != "orpo":
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)
    else:
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    evaluate(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs) -> None:
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser(TrainerCliArgs)
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    do_evaluate(parsed_cfg, parsed_cli_args)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
