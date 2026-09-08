"""Launch-only probe for ``fsdp_config.cpu_ram_efficient_loading``: build the model the way
``axolotl train`` does but skip dataset preparation, whose rank checks initialize the process
group as a side effect and would mask a loader that forgets to. Then run the trainer's
``accelerator.prepare(model, optimizer)`` and record what every rank ends up holding. Dumps
per-rank facts as JSON to ``$CPU_RAM_LOAD_PROBE_DIR/rank<LOCAL_RANK>.json`` for the outer test.

Usage: ``accelerate launch --num_processes 2 <this file> <config.yaml>``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def _weight_checksums(model: torch.nn.Module) -> list[float]:
    """Per-param sums of the gathered weights, in ``named_parameters`` order (collective)."""
    from torch.distributed.tensor import DTensor

    sums = []
    for name, param in model.named_parameters():
        # EP-sharded base experts legitimately differ per rank; everything else must match rank 0
        if ".experts." in name and "lora_" not in name:
            continue
        full = param.full_tensor() if isinstance(param, DTensor) else param
        sums.append(float(full.detach().float().sum()))
    return sums


def main() -> None:
    from accelerate import Accelerator

    from axolotl.cli.config import load_cfg
    from axolotl.train import setup_model_and_tokenizer

    cfg = load_cfg(sys.argv[1])
    dist_initialized_before_load = dist.is_initialized()
    model, _tokenizer, _peft_config, _processor = setup_model_and_tokenizer(cfg)

    params = list(model.named_parameters())
    facts = {
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "dist_initialized_before_load": dist_initialized_before_load,
        "meta_buffers": [n for n, b in model.named_buffers() if b.is_meta],
        "trainable_meta_params": [
            n for n, p in params if p.requires_grad and p.is_meta
        ],
        "num_meta_params": sum(p.is_meta for _, p in params),
        "num_params": len(params),
    }

    # Same order as transformers' Trainer under FSDP2: optimizer first, then prepare both.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    model, optimizer = Accelerator().prepare(model, optimizer)

    slots = [p for group in optimizer.param_groups for p in group["params"]]
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    facts.update(
        {
            "optimizer_slots": len(slots),
            "optimizer_distinct_params": len({id(p) for p in slots}),
            "optimizer_params_are_model_trainable": {id(p) for p in slots}
            == trainable_ids,
            "meta_params_after_prepare": [
                n for n, p in model.named_parameters() if p.is_meta
            ],
            "weight_checksums": _weight_checksums(model),
        }
    )

    out_dir = Path(os.environ["CPU_RAM_LOAD_PROBE_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rank{facts['local_rank']}.json").write_text(
        json.dumps(facts, indent=2)
    )
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
