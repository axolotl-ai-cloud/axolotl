#!/usr/bin/env python
"""Inspect Qwen2 MoE expert implementations for grouped-mm debugging."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT / "transformers" / "src"),
        str(ROOT / "src"),
    ]
)

from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

from axolotl.kernels.moe.torch_grouped import _iter_expert_impls


def main() -> None:
    cfg = Qwen2MoeConfig(
        hidden_size=4096,
        moe_intermediate_size=14336,
        shared_expert_intermediate_size=14336,
        num_experts=32,
        num_experts_per_tok=4,
    )

    block = Qwen2MoeSparseMoeBlock(cfg).to("cuda", dtype=torch.bfloat16)
    experts = block.experts
    experts._ax_parent_block = block

    impls = _iter_expert_impls(experts)
    print(f"impl count: {len(impls)}")
    for idx, impl in enumerate(impls[:8]):
        has_gate = hasattr(impl, "gate_proj")
        has_up = hasattr(impl, "up_proj")
        print(
            f"impl[{idx}] type={impl.__class__.__name__} has_gate={has_gate} has_up={has_up}"
        )
        if has_gate:
            print(f"  gate shape {tuple(impl.gate_proj.weight.shape)}")
            print(f"  up shape   {tuple(impl.up_proj.weight.shape)}")
            print(f"  down shape {tuple(impl.down_proj.weight.shape)}")


if __name__ == "__main__":
    main()
