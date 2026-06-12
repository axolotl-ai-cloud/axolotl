"""CLI to export a trained TorchSpec draft checkpoint to HuggingFace format."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from axolotl.cli.train_speculator import _load_speculator_cfg
from axolotl.integrations.torchspec.translate import _get_spec_args
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def do_export_speculator(
    config: Union[Path, str],
    input_dir: str | None = None,
    output_dir: str | None = None,
    dtype: str | None = None,
    prune_vocab: bool = False,
    force: bool = False,
    **kwargs,
) -> None:
    """Convert a TorchSpec FSDP draft checkpoint into an HF-loadable EAGLE-3 model.

    Args:
        config: the axolotl config used for training (supplies base_model,
            chat_template, output_dir, and the speculator block).
        input_dir: FSDP checkpoint dir (default ``<output_dir>/checkpoints``).
        output_dir: HF output dir (default ``<input_dir>_hf``).
        dtype: output dtype (float16/bfloat16/float32); float16 for FP8 targets.
        prune_vocab: prune the draft vocab using the training dataset.
        force: overwrite the output dir if it exists.
    """
    cfg: DictDefault = _load_speculator_cfg(config, **kwargs)
    spec = _get_spec_args(cfg)
    trust_remote_code = bool(cfg.get("trust_remote_code"))

    train_output_dir = (
        spec.output_dir or cfg.get("output_dir") or "./outputs/speculator"
    )
    resolved_input = (input_dir or str(Path(train_output_dir) / "checkpoints")).rstrip(
        "/"
    )
    if not Path(resolved_input).exists():
        raise FileNotFoundError(
            f"Checkpoint dir not found: {resolved_input}. Pass --input-dir "
            "explicitly (the FSDP checkpoint directory written during training)."
        )

    import torch
    from torchspec.tools.convert_to_hf import (
        _convert_fsdp_to_hf,
        _detect_model_dir,
        _resolve_config_path,
    )

    resolved_output = output_dir or f"{resolved_input}_hf"
    if os.path.exists(resolved_output) and not force:
        raise FileExistsError(
            f"Output dir {resolved_output} already exists; pass --force to overwrite."
        )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    output_dtype = dtype_map[dtype] if dtype else None

    chat_template = spec.chat_template
    if chat_template is None and cfg.get("chat_template"):
        from axolotl.integrations.torchspec.translate import _CHAT_TEMPLATE_MAP

        chat_template = _CHAT_TEMPLATE_MAP.get(cfg.get("chat_template"))

    config_path = _resolve_config_path(
        resolved_input,
        spec.draft_model_config,
        cfg.get("base_model"),
        trust_remote_code,
    )

    LOG.info("Exporting draft checkpoint %s -> %s", resolved_input, resolved_output)
    _convert_fsdp_to_hf(
        config_path=config_path,
        input_dir=_detect_model_dir(resolved_input),
        output_dir=resolved_output,
        target_model_path=cfg.get("base_model"),
        output_dtype=output_dtype,
        prune_vocab=prune_vocab,
        dataset_path=(
            str(Path(train_output_dir) / "torchspec_data" / "train.jsonl")
            if prune_vocab
            else None
        ),
        draft_vocab_size=spec.draft_vocab_size if prune_vocab else None,
        tokenizer=cfg.get("base_model") if prune_vocab else None,
        chat_template=chat_template if prune_vocab else None,
        prompt_key=spec.prompt_key,
        max_seq_length=int(cfg.get("sequence_len") or 32768),
        cache_dir=cfg.get("model_download_dir"),
    )
    LOG.info("Exported HF EAGLE-3 draft model to %s", resolved_output)
