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


def load() -> Optional[HFTritonHandles]:
    try:
        from kernels import get_kernel

        tk = get_kernel("kernels-community/triton_kernels")
        return HFTritonHandles(
            routing=tk.routing, matmul_ogs=tk.matmul_ogs, swiglu=tk.swiglu
        )
    except Exception:
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
    # use hub routing if available; otherwise fallback to softmax+topk
    routed = None
    if available():
        try:
            routed = route_topk(router_logits, top_k)
        except Exception:
            routed = None
    if routed is None:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weight, topk_idx = torch.topk(routing_weights, top_k, dim=-1, sorted=False)
        topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight.to(flat.dtype)
        x_rep = flat.repeat_interleave(top_k, dim=0)
        y = torch.empty_like(x_rep)
        flat_idx = topk_idx.view(-1)
        for i in range(experts_module.num_experts):
            expert = experts_module[i]
            y[flat_idx == i] = expert(x_rep[flat_idx == i])
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        return y.reshape(bsz, seqlen, hdim), router_logits
    # If routed via hub, still fallback to per-expert compute until grouped GEMM path is wired.
    ex_routing_data, gather_idx, scatter_idx = routed
    # Convert to naive per-expert compute on packed tokens (future: call matmul_ogs + swiglu)
    # For now, reconstruct the same result as naive path (no speedup but validates routing).
    # We map the selected experts from gather_idx back to expert ids via router_logits argmax among top-k.
    # Simpler: reuse naive computation for correctness; detailed integration will follow.
    routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, top_k, dim=-1, sorted=False)
    topk_weight = (topk_weight / topk_weight.sum(dim=-1, keepdim=True)).to(flat.dtype)
    x_rep = flat.repeat_interleave(top_k, dim=0)
    y = torch.empty_like(x_rep)
    flat_idx = topk_idx.view(-1)
    for i in range(experts_module.num_experts):
        expert = experts_module[i]
        y[flat_idx == i] = expert(x_rep[flat_idx == i])
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    return y.reshape(bsz, seqlen, hdim), router_logits
