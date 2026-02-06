"""CLI to merge a trained LoRA into a base model."""

from pathlib import Path
from typing import Union

import fire

from axolotl.cli.config import load_cfg
from axolotl.cli.utils import load_model_and_tokenizer
from axolotl.telemetry.errors import send_errors
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@send_errors
def do_merge_lora(*, cfg: DictDefault) -> None:
    """
    Calls `transformers`' `merge_and_unload` on the model given in the `axolotl` config
    along with the LoRA adapters to combine them into a single base model.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
    """
    model, tokenizer, processor = load_model_and_tokenizer(cfg=cfg)

    LOG.info("Running merge of LoRA with base model...")
    model = model.merge_and_unload(progressbar=True)
    try:
        model.to(dtype=cfg.torch_dtype)
    except ValueError as e:
        LOG.warning("Failed to convert model to dtype %s", cfg.torch_dtype)
        LOG.warning("Ignore this if the base_model is pre-quantized.")
        LOG.warning("Error raised: %s", e)

    model.generation_config.do_sample = True
    model.config.use_cache = True

    if cfg.local_rank == 0:
        LOG.info(f"Saving merged model to: {str(Path(cfg.output_dir) / 'merged')}...")
        model.save_pretrained(
            str(Path(cfg.output_dir) / "merged"),
            progressbar=True,
        )
        tokenizer.save_pretrained(
            str(Path(cfg.output_dir) / "merged"),
            save_jinja_files=cfg.tokenizer_save_jinja_files,
        )

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

    parsed_cfg = load_cfg(
        config,
        merge_lora=True,
        load_in_8bit=False,
        load_in_4bit=False,
        flash_attention=False,
        context_parallel_size=None,
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
    fire.Fire(do_cli)
