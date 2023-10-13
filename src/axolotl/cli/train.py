"""
CLI to run training on a model
"""
import logging
from pathlib import Path

import fire
import transformers
from colorama import Fore

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.train import train

LOG = logging.getLogger("axolotl.cli.train")


def do_cli(config: Path = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    check_accelerate_default_config()
    check_user_token()
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    if parsed_cli_args.prepare_ds_only and not parsed_cfg.dataset_prepared_path:
        msg = (
            Fore.RED
            + "--prepare_ds_only called without dataset_prepared_path set."
            + Fore.RESET
        )
        LOG.warning(msg)
        parsed_cfg.dataset_prepared_path = DEFAULT_DATASET_PREPARED_PATH

    dataset_meta = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
    if parsed_cli_args.prepare_ds_only:
        return
    train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)


if __name__ == "__main__":
    fire.Fire(do_cli)
