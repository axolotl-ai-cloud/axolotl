"""
CLI to run training on a model
"""
import logging
from pathlib import Path
from typing import Tuple, Union

import fire
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    load_rl_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.prompt_strategies.sharegpt import register_chatml_template
from axolotl.train import train

LOG = logging.getLogger("axolotl.cli.train")


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    parser = HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    return do_train(parsed_cfg, parsed_cli_args)


def do_train(cfg, cli_args) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print_axolotl_text_art()
    check_accelerate_default_config()
    check_user_token()
    if cfg.chat_template == "chatml" and cfg.default_system_message:
        LOG.info(
            f"ChatML set. Adding default system message: {cfg.default_system_message}"
        )
        register_chatml_template(cfg.default_system_message)
    else:
        register_chatml_template()

    if cfg.rl:
        dataset_meta = load_rl_datasets(cfg=cfg, cli_args=cli_args)
    else:
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

    return train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)


if __name__ == "__main__":
    fire.Fire(do_cli)
