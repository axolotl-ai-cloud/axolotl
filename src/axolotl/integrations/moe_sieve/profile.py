"""Calibrate with ``python -m axolotl.integrations.moe_sieve.profile config.yml``."""

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from .args import MoeSieveConfig
from .plugin import validate_runtime
from .selection import profile_routing


def profile(config: str, output: str | None = None):
    """Profile a reproducible sample of an Axolotl SFT training dataset."""
    from axolotl.cli.config import load_cfg
    from axolotl.cli.utils import load_model_and_tokenizer
    from axolotl.common.datasets import load_datasets

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run MoE-Sieve calibration in one process")
    cfg = load_cfg(
        config,
        fsdp=None,
        fsdp_config=None,
        dp_shard_size=1,
        dp_replicate_size=1,
        expert_parallel_size=1,
        context_parallel_size=1,
        context_parallel=None,
    )
    validate_runtime(cfg)
    if cfg.rl or cfg.is_multimodal or cfg.streaming or cfg.pretraining_dataset:
        raise ValueError(
            "MoE-Sieve calibration currently requires a map-style text SFT dataset"
        )
    settings = MoeSieveConfig.model_validate(dict(cfg.get("moe_sieve") or {}))
    destination = output or settings.selection_file
    if not destination:
        raise ValueError("Set moe_sieve.selection_file or pass --output")
    cfg.adapter = None
    cfg.lora_model_dir = None
    dataset = load_datasets(cfg=cfg).train_dataset
    seed = cfg.seed if cfg.seed is not None else 42
    dataset = dataset.shuffle(seed=seed).select(
        range(min(settings.calibration_samples, len(dataset)))
    )
    model, _, _ = load_model_and_tokenizer(cfg=cfg, inference=True)
    device = model.get_input_embeddings().weight.device
    digest = hashlib.sha256()

    def batches():
        for row in dataset:
            inputs = {
                key: row[key]
                for key in ("input_ids", "attention_mask", "position_ids")
                if key in row
            }
            digest.update(json.dumps(inputs, sort_keys=True).encode())
            yield {
                key: torch.tensor(value, dtype=torch.long, device=device).unsqueeze(0)
                for key, value in inputs.items()
            }

    autocast = (
        torch.autocast(device_type=device.type, dtype=cfg.torch_dtype)
        if cfg.torch_dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )
    with autocast:
        selection = profile_routing(model, batches(), settings.fraction)
    result = {
        "version": 1,
        "base_model": cfg.base_model,
        "revision": cfg.revision_of_model,
        "fraction": settings.fraction,
        "seed": seed,
        "calibration_samples": len(dataset),
        "calibration_sha256": digest.hexdigest(),
        "selection": selection,
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return str(path)


if __name__ == "__main__":
    import fire

    fire.Fire(profile)
