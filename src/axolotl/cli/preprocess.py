"""CLI to run preprocessing of a dataset."""

import os
import warnings
from pathlib import Path
from typing import Union

import fire
import transformers
from accelerate import init_empty_weights
from colorama import Fore
from transformers import AutoModelForCausalLM

from axolotl.cli.args import PreprocessCliArgs
from axolotl.cli.checks import check_accelerate_default_config, check_user_token
from axolotl.cli.config import load_cfg
from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.common.datasets import load_datasets, load_preference_datasets
from axolotl.integrations.base import PluginManager
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger
from axolotl.utils.trainer import disable_datasets_caching

LOG = get_logger(__name__)


def do_preprocess(cfg: DictDefault, cli_args: PreprocessCliArgs) -> None:
    """
    Preprocesses dataset specified in axolotl config.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Preprocessing-specific CLI arguments.
    """
    check_accelerate_default_config()
    check_user_token()

    for key in ["skip_prepare_dataset", "pretraining_dataset"]:
        if cfg.get(key):
            LOG.error(
                f"You have set `{key}:`. `preprocess` is not needed. Run the `axolotl train` CLI directly instead."
            )
            return

    if not cfg.dataset_prepared_path:
        msg = (
            Fore.RED
            + "preprocess CLI called without dataset_prepared_path set, "
            + f"using default path: {DEFAULT_DATASET_PREPARED_PATH}"
            + Fore.RESET
        )
        LOG.warning(msg)
        cfg.dataset_prepared_path = DEFAULT_DATASET_PREPARED_PATH

    with disable_datasets_caching():
        plugin_manager = PluginManager.get_instance()
        if plugin_manager.load_datasets(cfg, preprocess=True):
            pass
        elif cfg.rl:
            load_preference_datasets(cfg=cfg, cli_args=cli_args)
        else:
            load_datasets(cfg=cfg, cli_args=cli_args)

    if cli_args.download:
        model_name = cfg.base_model
        with warnings.catch_warnings():
            # there are a bunch of useless UserWarnings about
            # "copying from a non-meta parameter in the checkpoint to a meta parameter in the current model"
            warnings.simplefilter("ignore")
            with init_empty_weights(include_buffers=True):
                # fmt: off
                try:
                    AutoModelForCausalLM.from_pretrained(
                        model_name, trust_remote_code=True
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught,unused-variable  # nosec B110  # noqa F841
                    pass
                # fmt: on

    LOG.info(
        Fore.GREEN
        + f"Success! Preprocessed data path: `dataset_prepared_path: {cfg.dataset_prepared_path}`"
        + Fore.RESET
    )


def do_cli(
    config: Union[Path, str] = Path("examples/"),
    **kwargs,
) -> None:
    """
    Parses `axolotl` config, CLI args, and calls `do_preprocess`.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """
    # pylint: disable=duplicate-code
    os.environ["AXOLOTL_IS_PREPROCESS"] = "1"
    parsed_cfg = load_cfg(config, **kwargs)
    parsed_cfg.is_preprocess = True
    parser = transformers.HfArgumentParser(PreprocessCliArgs)
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    do_preprocess(parsed_cfg, parsed_cli_args)


if __name__ == "__main__":
    fire.Fire(do_cli)
