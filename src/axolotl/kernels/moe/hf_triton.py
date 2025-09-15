"""
Adapter for Hugging Face kernels hub (kernels-community/triton_kernels).
This file provides light probes and placeholders for future integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class HFTritonHandles:
    routing: Any
    matmul_ogs: Any
    swiglu: Any


def available() -> bool:
    try:
        import kernels  # noqa: F401

        return True
    except Exception:
        return False


# Cache loaded handles so we don't trigger repeated hub fetches
_CACHED_HANDLES: Optional[HFTritonHandles] = None
_LOAD_ATTEMPTED: bool = False


def load() -> Optional[HFTritonHandles]:
    global _CACHED_HANDLES, _LOAD_ATTEMPTED
    if _CACHED_HANDLES is not None:
        return _CACHED_HANDLES
    if _LOAD_ATTEMPTED:
        # Previously failed; avoid spamming retries per call
        return None
    _LOAD_ATTEMPTED = True
    try:
        from kernels import get_kernel

        tk = get_kernel("kernels-community/triton_kernels")
        _CACHED_HANDLES = HFTritonHandles(
            routing=tk.routing, matmul_ogs=tk.matmul_ogs, swiglu=tk.swiglu
        )
        return _CACHED_HANDLES
    except Exception:
        # Keep None in cache state to prevent repeated fetch attempts
        return None


def route_topk(logits, top_k: int):
    handles = load()
    if handles is None:
        return None
    return handles.routing.routing_torch(logits, n_expts_act=top_k)


def swiglu(x, alpha, limit=1.0, routing_data=None):
    handles = load()
    if handles is None:
        return None
    pc = handles.swiglu.PrecisionConfig(limit=limit)
    return handles.swiglu.swiglu(x, alpha, pc, routing_data)


def moe_ffn_forward_stub(
    hidden_states, gate_linear, experts_module, top_k: int
) -> Tuple[object, object]:
    """
    Temporary stub that uses kernels hub routing, but falls back to per-expert compute.
    Returns (final_hidden_states, router_logits).
    """
    import torch
    import torch.nn.functional as F

    bsz, seqlen, hdim = hidden_states.shape
    flat = hidden_states.view(-1, hdim)
    router_logits = gate_linear(flat)
    # For now, do not call routing to avoid extra overhead until
    # grouped GEMM integration is complete. Use the naive compute path
    # for correctness and baseline performance.
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, top_k, dim=-1, sorted=False)
    topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
    topk_weight = topk_weight.to(flat.dtype)
    x_rep = flat.repeat_interleave(top_k, dim=0)
    y = torch.empty_like(x_rep)
    flat_idx = topk_idx.view(-1)
    for i in range(experts_module.num_experts):
        expert = experts_module[i]
        sel = flat_idx == i
        if sel.any():
            y[sel] = expert(x_rep[sel])
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    return y.reshape(bsz, seqlen, hdim), router_logits
