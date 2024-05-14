"""
CLI to run inference on a trained model
"""
from pathlib import Path

import fire
import transformers

from axolotl.cli import (
    do_inference,
    do_inference_gradio,
    load_cfg,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs


def do_cli(config: Path = Path("examples/"), gradio=False, **kwargs):
    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parsed_cfg = load_cfg(config, **kwargs)
    parsed_cfg.sample_packing = False
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parsed_cli_args.inference = True

    if gradio:
        do_inference_gradio(cfg=parsed_cfg, cli_args=parsed_cli_args)
    else:
        do_inference(cfg=parsed_cfg, cli_args=parsed_cli_args)


if __name__ == "__main__":
    fire.Fire(do_cli)
