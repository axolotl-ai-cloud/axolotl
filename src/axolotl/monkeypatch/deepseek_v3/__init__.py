from __future__ import annotations

from typing import Iterable

import torch

from axolotl.kernels.moe.deepseek_v3 import moe_forward_kernel
from axolotl.utils.logging import get_logger


LOG = get_logger(__name__)


def _looks_like_deepseek_v3_moe(module: torch.nn.Module) -> bool:
    # Heuristic: module has gate (Linear), experts (iterable of modules with gate_proj/up_proj/down_proj)
    if not hasattr(module, "gate") or not hasattr(module, "experts"):
        return False
    gate = module.gate
    experts = module.experts
    if not isinstance(gate, torch.nn.Linear):
        return False
    if not isinstance(experts, Iterable):
        return False
    # Verify at least one expert has expected projections
    try:
        exp0 = next(iter(experts))
    except StopIteration:
        return False
    needed = all(hasattr(exp0, name) for name in ("gate_proj", "up_proj", "down_proj"))
    return needed


def _mk_patched_forward(module: torch.nn.Module):
    # Extract config-like attributes if present
    top_k = getattr(module, "top_k", None)
    if top_k is None:
        # HF often stores as num_experts_per_tok
        top_k = getattr(module, "num_experts_per_tok", 2)

    # DeepSeek-V3 commonly uses sigmoid routing with normalization
    score_func = getattr(module, "router_score_fn", "sigmoid")
    route_norm = getattr(module, "route_norm", True)
    route_scale = getattr(module, "route_scale", 1.0)

    def patched_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        return moe_forward_kernel(
            hidden_states=hidden_states,
            gate=module.gate,
            experts=list(module.experts),
            shared_expert=getattr(module, "shared_experts", None),
            top_k=top_k,
            score_func=score_func,
            route_norm=route_norm,
            route_scale=route_scale,
        )

    return patched_forward


def apply_titan_moe_to_deepseek_v3(model: torch.nn.Module) -> None:
    """Replace MoE MLP forward in DeepSeek-V3-like HF models with torchtitan-style kernels.

    This scans submodules heuristically and patches the ones matching DeepSeek-V3 MoE MLP.
    """
    patched = 0
    for name, module in model.named_modules():
        if _looks_like_deepseek_v3_moe(module):
            try:
                module.forward = _mk_patched_forward(module)
                patched += 1
            except Exception as e:
                LOG.warning(f"Failed to patch MoE at {name}: {e}")

    if patched:
        LOG.info(f"Applied torchtitan MoE kernels to {patched} DeepSeek-V3 MoE layers")
    else:
        LOG.info("No DeepSeek-V3 MoE layers found to patch")
