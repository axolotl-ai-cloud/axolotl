"""Minimal grouped GEMM fast path for MoE experts using PyTorch _grouped_mm."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

_LOGGER = logging.getLogger("axolotl.moe.grouped")


def available() -> bool:
    try:
        major, minor = map(int, torch.__version__.split("+")[0].split(".")[:2])
        if (major, minor) < (2, 8):
            return False
        if not torch.cuda.is_available():
            return False
        sm, _ = torch.cuda.get_device_capability()
        if sm < 9:
            return False
        return hasattr(torch.ops, "_grouped_mm")
    except Exception:
        return False


def _iter_expert_impls(experts_module) -> List[torch.nn.Module]:
    impls: List[torch.nn.Module] = []
    for exp in experts_module:
        impls.append(getattr(exp, "mlp", getattr(exp, "ffn", exp)))
    return impls


def _stack_weights(
    experts_module,
    names: Tuple[str, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for mod in _iter_expert_impls(experts_module):
        parts = [getattr(mod, name).weight.t() for name in names]
        tensors.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1))

    return (
        torch.stack(tensors, dim=0)
        .to(device=device, dtype=dtype, non_blocking=True)
        .contiguous()
    )


def moe_ffn_forward_grouped(
    hidden_states: torch.Tensor,
    gate_linear: torch.nn.Linear,
    experts_module,
    top_k: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not available():
        return None, None

    bsz, seqlen, hdim = hidden_states.shape
    tokens = bsz * seqlen
    device = hidden_states.device

    routing_dtype = gate_linear.weight.dtype
    expert_dtype = hidden_states.dtype

    if expert_dtype not in (torch.bfloat16, torch.float16):
        _LOGGER.debug(
            "torch_grouped: unsupported expert dtype %s; falling back to naive",
            expert_dtype,
        )
        return None, None

    for suffix in ("w13", "w2"):
        attr = f"_ax_grouped_{suffix}"
        if hasattr(experts_module, attr):
            delattr(experts_module, attr)

    expert_impls = _iter_expert_impls(experts_module)
    sample_mod = expert_impls[0]
    if (
        hasattr(sample_mod, "w1")
        and hasattr(sample_mod, "w3")
        and hasattr(sample_mod, "w2")
    ):
        w13 = _stack_weights(
            experts_module, ("w1", "w3"), dtype=expert_dtype, device=device
        )
        w2 = _stack_weights(experts_module, ("w2",), dtype=expert_dtype, device=device)
    else:
        if hasattr(sample_mod, "gate_up_proj"):
            names13: Tuple[str, ...] = ("gate_up_proj",)
        else:
            names13 = ("up_proj", "gate_proj")
        w13 = _stack_weights(experts_module, names13, dtype=expert_dtype, device=device)
        w2 = _stack_weights(
            experts_module, ("down_proj",), dtype=expert_dtype, device=device
        )

    x_flat = hidden_states.view(tokens, hdim).to(expert_dtype)
    router_logits = gate_linear(x_flat.to(routing_dtype))

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, top_k, dim=-1, sorted=False)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    flat_idx = topk_idx.view(-1)
    num_experts = len(expert_impls)
    if flat_idx.numel() == 0:
        zero = torch.zeros_like(x_flat)
        return zero.view(bsz, seqlen, hdim), router_logits

    assignments = torch.bincount(flat_idx, minlength=num_experts)
    if assignments.sum() == 0:
        zero = torch.zeros_like(x_flat)
        return zero.view(bsz, seqlen, hdim), router_logits

    perm = torch.argsort(flat_idx, stable=True)
    token_indices_sorted = perm // top_k
    scores_sorted = topk_weight.view(-1)[perm]

    gather_index = token_indices_sorted.unsqueeze(-1).expand(-1, hdim)
    routed_input = torch.gather(x_flat, 0, gather_index).contiguous()

    offsets = torch.cumsum(assignments.to(device=device, dtype=torch.int32), dim=0)
    if offsets[-1].item() == 0:
        zero = torch.zeros_like(x_flat)
        return zero.view(bsz, seqlen, hdim), router_logits

    mid = w13.shape[-1] // 2
    w_gate = w13[..., :mid]
    w_up = w13[..., mid:]

    w_gate_t = w_gate.transpose(-2, -1).contiguous()
    w_up_t = w_up.transpose(-2, -1).contiguous()
    w2_t = w2.transpose(-2, -1).contiguous()

    routed_in = routed_input.to(expert_dtype)
    gate_out = torch._grouped_mm(routed_in, w_gate_t, offs=offsets)
    up_out = torch._grouped_mm(routed_in, w_up_t, offs=offsets)
    activated = F.silu(gate_out) * up_out
    down_out = torch._grouped_mm(activated, w2_t, offs=offsets)

    weights_fp32 = scores_sorted.unsqueeze(-1).to(torch.float32)
    weighted = (down_out.to(torch.float32) * weights_fp32).to(expert_dtype)

    combined = torch.zeros_like(x_flat)
    combined.scatter_add_(0, gather_index, weighted)
    return combined.view(bsz, seqlen, hdim), router_logits
