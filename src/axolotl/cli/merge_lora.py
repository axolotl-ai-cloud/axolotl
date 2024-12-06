"""
CLI to run merge a trained LoRA into a base model
"""
from pathlib import Path
from typing import Union

import fire
import transformers
from dotenv import load_dotenv

from axolotl.cli import do_merge_lora, load_cfg, print_axolotl_text_art
from axolotl.common.cli import TrainerCliArgs


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs):
    # pylint: disable=duplicate-code
    print_axolotl_text_art()
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parsed_cli_args.merge_lora = True

    parsed_cfg = load_cfg(
        config,
        merge_lora=True,
        load_in_8bit=False,
        load_in_4bit=False,
        flash_attention=False,
        deepspeed=None,
        fsdp=None,
        **kwargs,
    )

    if not parsed_cfg.lora_model_dir and parsed_cfg.output_dir:
        parsed_cfg.lora_model_dir = parsed_cfg.output_dir
    if not Path(parsed_cfg.lora_model_dir).exists():
        raise ValueError(
            f"Target directory for merge: `{parsed_cfg.lora_model_dir}` does not exist."
        )

    parsed_cfg.load_in_4bit = False
    parsed_cfg.load_in_8bit = False
    parsed_cfg.flash_attention = False
    parsed_cfg.deepspeed = None
    parsed_cfg.fsdp = None
    parsed_cfg.fsdp_config = None

    do_merge_lora(cfg=parsed_cfg, cli_args=parsed_cli_args)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
