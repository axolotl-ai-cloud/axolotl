"""
CLI to run merge a trained LoRA into a base model
"""
import logging
from pathlib import Path
from typing import Union

import fire
import transformers
from dotenv import load_dotenv

from axolotl.cli import print_axolotl_text_art
from axolotl.cli.config import load_cfg
from axolotl.common.cli import TrainerCliArgs, load_model_and_tokenizer
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger("axolotl.cli.merge_lora")


def do_merge_lora(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
):
    model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
    safe_serialization = cfg.save_safetensors is True

    LOG.info("running merge of LoRA with base model")
    model = model.merge_and_unload(progressbar=True)
    try:
        model.to(dtype=cfg.torch_dtype)
    except RuntimeError:
        pass
    model.generation_config.do_sample = True

    if cfg.local_rank == 0:
        LOG.info(f"saving merged model to: {str(Path(cfg.output_dir) / 'merged')}")
        model.save_pretrained(
            str(Path(cfg.output_dir) / "merged"),
            safe_serialization=safe_serialization,
            progressbar=True,
        )
        tokenizer.save_pretrained(str(Path(cfg.output_dir) / "merged"))


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
