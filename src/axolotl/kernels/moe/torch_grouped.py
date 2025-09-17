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


def _stack_weights(
    experts_module,
    names: Tuple[str, ...],
    *,
    key: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    attr = f"_ax_grouped_{key}"
    cached = getattr(experts_module, attr, None)
    if cached is not None and cached.dtype == dtype and cached.device == device:
        return cached

    tensors: List[torch.Tensor] = []
    for exp in experts_module:
        mod = getattr(exp, "mlp", getattr(exp, "ffn", exp))
        parts = [getattr(mod, name).weight.t() for name in names]
        tensors.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1))

    stacked = (
        torch.stack(tensors, dim=0)
        .to(device=device, dtype=dtype, non_blocking=True)
        .contiguous()
    )
    setattr(experts_module, attr, stacked)
    return stacked


def _call_grouped_mm(
    As: List[torch.Tensor], Bs: List[torch.Tensor], dtype: torch.dtype
) -> Optional[List[torch.Tensor]]:
    if not As:
        return []

    try:
        As2 = [a.to(dtype).contiguous().view(a.shape[0], a.shape[1]) for a in As]
        Bs2 = [b.to(dtype).contiguous().view(b.shape[0], b.shape[1]) for b in Bs]
        device = As2[0].device
        offs = torch.tensor([a.shape[0] for a in As2], device=device, dtype=torch.int32)
        Y_cat = torch.ops.aten._grouped_mm(
            torch.cat(As2, dim=0), torch.stack(Bs2, dim=0), offs
        )
        outs: List[torch.Tensor] = []
        start = 0
        for m in offs.tolist():
            outs.append(Y_cat[start : start + m])
            start += m
        return outs
    except RuntimeError as err:
        _LOGGER.warning("torch_grouped: _grouped_mm failed (%s)", err)
        return None


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
    x_flat = hidden_states.view(tokens, hdim)
    router_logits = gate_linear(x_flat.to(routing_dtype))

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, top_k, dim=-1, sorted=False)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    sample = getattr(
        experts_module[0], "mlp", getattr(experts_module[0], "ffn", experts_module[0])
    )
    if hasattr(sample, "w1") and hasattr(sample, "w3") and hasattr(sample, "w2"):
        w13 = _stack_weights(
            experts_module, ("w1", "w3"), key="w13", dtype=expert_dtype, device=device
        )
        w2 = _stack_weights(
            experts_module, ("w2",), key="w2", dtype=expert_dtype, device=device
        )
    else:
        names13 = (
            ("gate_up_proj",)
            if hasattr(sample, "gate_up_proj")
            else ("up_proj", "gate_proj")
        )
        w13 = _stack_weights(
            experts_module, names13, key="w13", dtype=expert_dtype, device=device
        )
        w2 = _stack_weights(
            experts_module, ("down_proj",), key="w2", dtype=expert_dtype, device=device
        )

    flat_idx = topk_idx.view(-1)
    x_rep = x_flat.to(expert_dtype).repeat_interleave(top_k, dim=0)

    as_list: List[torch.Tensor] = []
    bs_list: List[torch.Tensor] = []
    slices: List[Tuple[int, torch.Tensor]] = []
    for i, _ in enumerate(experts_module):
        sel = flat_idx == i
        if sel.any():
            as_list.append(x_rep[sel])
            bs_list.append(w13[i])
            slices.append((i, sel))

    if not as_list:
        return torch.zeros_like(x_flat).view(bsz, seqlen, hdim), router_logits

    up_out = _call_grouped_mm(as_list, bs_list, expert_dtype)
    if up_out is None:
        return None, None

    down_inputs: List[torch.Tensor] = []
    down_weights: List[torch.Tensor] = []
    buf = torch.empty_like(x_rep)
    for (i, _sel), Yi in zip(slices, up_out, strict=False):
        mid = Yi.shape[-1] // 2
        hidden = F.silu(Yi[:, :mid]) * Yi[:, mid:]
        down_inputs.append(hidden)
        down_weights.append(w2[i])

    down_out = _call_grouped_mm(down_inputs, down_weights, expert_dtype)
    if down_out is None:
        return None, None

    for (_i, sel), tensor in zip(slices, down_out, strict=False):
        buf[sel] = tensor

    combined = (
        buf.view(tokens, top_k, -1) * topk_weight.to(expert_dtype).unsqueeze(-1)
    ).sum(dim=1)
    return combined.view(bsz, seqlen, hdim), router_logits
