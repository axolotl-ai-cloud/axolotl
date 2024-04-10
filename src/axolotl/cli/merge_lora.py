"""
CLI to run merge a trained LoRA into a base model
"""
from pathlib import Path

import fire
import transformers

from axolotl.cli import do_merge_lora, load_cfg, print_axolotl_text_art
from axolotl.common.cli import TrainerCliArgs
from axolotl.utils.dict import DictDefault


def do_cli(config: Path = Path("examples/"), **kwargs):
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
        **kwargs,
    )
    cfg = modify_cfg_for_merge(parsed_cfg)

    do_merge_lora(cfg=cfg, cli_args=parsed_cli_args)


def modify_cfg_for_merge(cfg: DictDefault) -> DictDefault:
    if not cfg.lora_model_dir and cfg.output_dir:
        cfg.lora_model_dir = cfg.output_dir
    if not Path(cfg.lora_model_dir).exists():
        raise ValueError(
            f"Target directory for merge: `{cfg.lora_model_dir}` does not exist."
        )

    cfg.load_in_4bit = False
    cfg.load_in_8bit = False
    cfg.flash_attention = False
    cfg.deepspeed = None
    cfg.fsdp = None

    return cfg


if __name__ == "__main__":
    fire.Fire(do_cli)
