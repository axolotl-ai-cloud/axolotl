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
    # Fast path via kernels hub: route tokens, do grouped GEMMs for up, gate, and down.
    handles = load()
    if handles is not None:
        try:
            routing_data, gather_idx, scatter_idx = handles.routing.routing_torch(
                router_logits, n_expts_act=top_k
            )
            # Prepare and cache expert weights: shapes [E, K, N]
            import torch

            E = experts_module.num_experts
            dev = flat.device
            dt = flat.dtype
            if (
                not hasattr(experts_module, "_stacked_w1")
                or experts_module._stacked_w1.device != dev
                or experts_module._stacked_w1.dtype != dt
            ):
                W1 = []
                W3 = []
                W2 = []
                for i in range(E):
                    exp = experts_module[i]
                    W1.append(exp.w1.weight.t())
                    W3.append(exp.w3.weight.t())
                    W2.append(exp.w2.weight.t())
                experts_module._stacked_w1 = (
                    torch.stack(W1, dim=0)
                    .to(device=dev, dtype=dt, non_blocking=True)
                    .contiguous()
                )
                experts_module._stacked_w3 = (
                    torch.stack(W3, dim=0)
                    .to(device=dev, dtype=dt, non_blocking=True)
                    .contiguous()
                )
                experts_module._stacked_w2 = (
                    torch.stack(W2, dim=0)
                    .to(device=dev, dtype=dt, non_blocking=True)
                    .contiguous()
                )
            W1 = experts_module._stacked_w1
            W3 = experts_module._stacked_w3
            # Fused up+gate: single matmul on concatenated weights [E, H, 2I]
            W13 = getattr(experts_module, "_stacked_w13", None)
            if (
                W13 is None
                or W13.device != dev
                or W13.dtype != dt
                or W13.shape[-1] != (W1.shape[-1] + W3.shape[-1])
            ):
                W13 = torch.cat([W1, W3], dim=-1).contiguous()
                experts_module._stacked_w13 = W13
            Y13 = handles.matmul_ogs.matmul_ogs(
                flat,
                W13,
                None,
                routing_data=routing_data,
                gather_indx=gather_idx,
                scatter_indx=None,
                precision_config=handles.matmul_ogs.PrecisionConfig(),
            )
            # Use kernels hub SwiGLU for optimal MoE launch
            sw_pc = handles.swiglu.PrecisionConfig(limit=1.0)
            Hidden = handles.swiglu.swiglu(Y13, 1.0, sw_pc, routing_data)
            # Down projection weights [E, inter, hidden]
            W2 = experts_module._stacked_w2
            # Down matmul with fused scatter back using scatter_indx
            Out = handles.matmul_ogs.matmul_ogs(
                Hidden,
                W2,
                None,
                routing_data=routing_data,
                gather_indx=None,
                scatter_indx=scatter_idx,
                precision_config=handles.matmul_ogs.PrecisionConfig(),
                gammas=routing_data.gate_scal,
            )
            return Out.view(bsz, seqlen, hdim), router_logits
        except Exception:
            pass
    # Fallback naive path for correctness
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
