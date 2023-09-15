"""
CLI to run inference on a trained model
"""
from pathlib import Path

import fire
import transformers

from axolotl.cli import do_inference, load_cfg, print_axolotl_text_art
from axolotl.common.cli import TrainerCliArgs


def do_cli(config: Path = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parsed_cli_args.inference = True

    do_inference(cfg=parsed_cfg, cli_args=parsed_cli_args)


fire.Fire(do_cli)
