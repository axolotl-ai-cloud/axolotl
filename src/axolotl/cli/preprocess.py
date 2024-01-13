"""
CLI to run training on a model
"""
import logging
from pathlib import Path

import fire
import transformers
from colorama import Fore
from datasets import disable_caching

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import PreprocessCliArgs
from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH

LOG = logging.getLogger("axolotl.cli.preprocess")


def do_cli(config: Path = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    check_accelerate_default_config()
    check_user_token()
    parser = transformers.HfArgumentParser((PreprocessCliArgs))
    parsed_cli_args, remaining_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    if (
        remaining_args.get("disable_caching") is not None
        and remaining_args["disable_caching"]
    ):
        disable_caching()
    if not parsed_cfg.dataset_prepared_path:
        msg = (
            Fore.RED
            + "preprocess CLI called without dataset_prepared_path set, "
            + f"using default path: {DEFAULT_DATASET_PREPARED_PATH}"
            + Fore.RESET
        )
        LOG.warning(msg)
        parsed_cfg.dataset_prepared_path = DEFAULT_DATASET_PREPARED_PATH

    _ = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
    LOG.info(
        Fore.GREEN
        + f"Success! Preprocessed data path: `dataset_prepared_path: {parsed_cfg.dataset_prepared_path}`"
        + Fore.RESET
    )


if __name__ == "__main__":
    fire.Fire(do_cli)
