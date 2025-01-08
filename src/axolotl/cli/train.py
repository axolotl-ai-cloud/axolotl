"""CLI to run training on a model."""

import logging
from pathlib import Path
from typing import Union

import fire
from dotenv import load_dotenv
from transformers.hf_argparser import HfArgumentParser

from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.checks import check_accelerate_default_config, check_user_token
from axolotl.cli.config import load_cfg
from axolotl.common.cli import TrainerCliArgs
from axolotl.common.datasets import load_datasets, load_rl_datasets
from axolotl.integrations.base import PluginManager
from axolotl.train import train
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger(__name__)


def do_train(cfg: DictDefault, cli_args: TrainerCliArgs) -> None:
    """
    Trains a `transformers` model by first loading the dataset(s) specified in the
    `axolotl` config, and then calling `axolotl.train.train`. Also runs the plugin
    manager's `post_train_unload` once training completes.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Training-specific CLI arguments.
    """
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()

    if cfg.rl:  # and cfg.rl != "orpo":
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)
    else:
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    model, tokenizer = train(cfg=cfg, dataset_meta=dataset_meta)
    plugin_manager = PluginManager.get_instance()

    del model
    del tokenizer

    plugin_manager.post_train_unload(cfg)


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs) -> None:
    """
    Parses `axolotl` config, CLI args, and calls `do_train`.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser(TrainerCliArgs)
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    do_train(parsed_cfg, parsed_cli_args)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
