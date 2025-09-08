from __future__ import annotations

from typing import Iterable

import torch

from axolotl.kernels.moe.deepseek_v3 import moe_forward_kernel
from axolotl.utils.logging import get_logger


LOG = get_logger(__name__)


def _is_router(mod: torch.nn.Module) -> bool:
    """Return True if module looks like a router (Linear or DeepseekV3TopkRouter)."""
    if not isinstance(mod, torch.nn.Module):
        return False
    if isinstance(mod, torch.nn.Linear):
        return True
    cls_name = mod.__class__.__name__
    return cls_name in {"DeepseekV3TopkRouter", "TopKRouter", "DeepseekV3Router"}


def _is_deepseek_v3_mlp(mod: torch.nn.Module) -> bool:
    """Check that a module exposes expected DeepSeek-V3 MLP projections."""
    return all(hasattr(mod, n) for n in ("gate_proj", "up_proj", "down_proj"))


def _looks_like_deepseek_v3_moe(module: torch.nn.Module) -> bool:
    """Detect DeepSeek-V3 MoE blocks robustly.

    Supports both:
    - Official DeepseekV3MoE (with `gate` router module and `experts: ModuleList[DeepseekV3MLP]`)
    - Axolotl mini variant (with `gate: Linear` and `experts` exposing *_proj)
    """
    # Quick positive match by class name
    cls_name = module.__class__.__name__
    if cls_name in {"DeepseekV3MoE", "DeepseekV3MiniMoEMLP"}:
        return True

    # Structural checks
    if not hasattr(module, "gate") or not hasattr(module, "experts"):
        return False
    gate = getattr(module, "gate")
    experts = getattr(module, "experts")
    if not _is_router(gate):
        return False
    if not isinstance(experts, Iterable):
        return False
    try:
        exp0 = next(iter(experts))
    except StopIteration:
        return False
    return _is_deepseek_v3_mlp(exp0)


def _mk_patched_forward(module: torch.nn.Module):
    # Extract config-like attributes if present
    top_k = getattr(module, "top_k", None)
    if top_k is None:
        # HF often stores as num_experts_per_tok on MoE module
        top_k = getattr(module, "num_experts_per_tok", None)
    if top_k is None and hasattr(module, "gate"):
        # Some implementations keep it on the router
        top_k = getattr(module.gate, "top_k", getattr(module.gate, "k", None))
    if top_k is None:
        top_k = 2

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
