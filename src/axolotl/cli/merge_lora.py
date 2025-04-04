"""CLI to merge a trained LoRA into a base model."""

import logging
from pathlib import Path
from typing import Union

import fire
import transformers
from dotenv import load_dotenv

from axolotl.cli.args import TrainerCliArgs
from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.config import load_cfg
from axolotl.cli.utils import load_model_and_tokenizer
from axolotl.utils.dict import DictDefault

LOG = logging.getLogger(__name__)


def do_merge_lora(*, cfg: DictDefault) -> None:
    """
    Calls `transformers`' `merge_and_unload` on the model given in the `axolotl` config
    along with the LoRA adapters to combine them into a single base model.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
    """
    print_axolotl_text_art()

    model, tokenizer, processor = load_model_and_tokenizer(cfg=cfg)
    safe_serialization = cfg.save_safetensors is True

    LOG.info("Running merge of LoRA with base model...")
    model = model.merge_and_unload(progressbar=True)
    model.to(dtype=cfg.torch_dtype)
    model.generation_config.do_sample = True

    if cfg.local_rank == 0:
        LOG.info(f"Saving merged model to: {str(Path(cfg.output_dir) / 'merged')}...")
        model.save_pretrained(
            str(Path(cfg.output_dir) / "merged"),
            safe_serialization=safe_serialization,
            progressbar=True,
        )
        tokenizer.save_pretrained(str(Path(cfg.output_dir) / "merged"))

        if processor:
            processor.save_pretrained(str(Path(cfg.output_dir) / "merged"))


def do_cli(config: Union[Path, str] = Path("examples/"), **kwargs) -> None:
    """
    Parses `axolotl` config, CLI args, and calls `do_merge_lora`. Note that various
    config values will be overwritten to allow the LoRA merge logic to work as expected
    (`load_in_8bit=False`, `load_in4bit=False`, `flash_attention=False`, etc.).

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments to override config file values.

    Raises:
        ValueError: If target directory for LoRA merged model does not exist.
    """
    # pylint: disable=duplicate-code
    parser = transformers.HfArgumentParser(TrainerCliArgs)
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
        sequence_parallel_degree=None,
        deepspeed=None,
        fsdp=None,
        fsdp_config=None,
        **kwargs,
    )

    if not parsed_cfg.lora_model_dir and parsed_cfg.output_dir:
        parsed_cfg.lora_model_dir = parsed_cfg.output_dir
    if not Path(parsed_cfg.lora_model_dir).exists():
        raise ValueError(
            f"Target directory for merge: `{parsed_cfg.lora_model_dir}` does not exist."
        )

    do_merge_lora(cfg=parsed_cfg)


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
