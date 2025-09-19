"""CLI to merge a trained LoRA into a base model."""

from pathlib import Path
from typing import Union

import fire
import torch

from axolotl.cli.config import load_cfg
from axolotl.cli.utils import load_model_and_tokenizer
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger
from axolotl.utils.lora_merge_efficient import merge_lora_sharded_efficient

LOG = get_logger(__name__)


def do_merge_lora(*, cfg: DictDefault) -> None:
    """
    Merges LoRA adapters with base model using either memory-efficient or legacy approach.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
    """
    merge_method = (
        str(getattr(cfg, "merge_method", "")).strip().lower().replace("-", "_")
    )
    if merge_method in {"legacy", "standard"}:
        LOG.debug("Using legacy LoRA merging method...")
        _do_merge_lora_legacy(cfg=cfg)
    else:
        LOG.debug("Using memory-efficient LoRA merging method...")
        try:
            _do_merge_lora_efficient(cfg=cfg)
        except Exception:  # pylint: disable=broad-exception-caught
            LOG.exception("Memory-efficient merge failed; falling back to legacy.")
            _do_merge_lora_legacy(cfg=cfg)


def _do_merge_lora_legacy(*, cfg: DictDefault) -> None:
    """
    Legacy LoRA merging using merge_and_unload.
    Loads the full model into memory before merging.
    """
    LOG.debug("Using legacy LoRA merging method...")
    model, tokenizer, processor = load_model_and_tokenizer(cfg=cfg)
    safe_serialization = cfg.save_safetensors is True

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
            safe_serialization=safe_serialization,
            progressbar=True,
        )
        tokenizer.save_pretrained(
            str(Path(cfg.output_dir) / "merged"),
            save_jinja_files=cfg.tokenizer_save_jinja_files,
        )

        if processor:
            processor.save_pretrained(str(Path(cfg.output_dir) / "merged"))


def _do_merge_lora_efficient(*, cfg: DictDefault) -> None:
    """
    Memory-efficient LoRA merging using shard-by-shard processing.
    Does not load the full model into memory.

    Note: Currently only supports standard LoRA, not advanced methods like DoRA or RSLoRA.
    Will automatically fall back to legacy method for unsupported configurations.
    """
    LOG.debug("Using memory-efficient LoRA merging method...")

    output_path = Path(cfg.output_dir) / "merged"
    safe_tensors = getattr(cfg, "save_safetensors", True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Perform memory-efficient merge
    merge_lora_sharded_efficient(
        base_model_path=cfg.base_model,
        lora_adapter_path=cfg.lora_model_dir,
        output_path=output_path,
        safe_tensors=safe_tensors,
        device=device,
    )

    LOG.debug("Memory-efficient LoRA merge completed successfully!")


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
            f"Target directory for LoRA adapter weights does not exist: `{parsed_cfg.lora_model_dir}`"
        )

    do_merge_lora(cfg=parsed_cfg)


if __name__ == "__main__":
    fire.Fire(do_cli)
