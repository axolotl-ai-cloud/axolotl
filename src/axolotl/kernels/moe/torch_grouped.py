"""Minimal grouped GEMM fast path for MoE experts using PyTorch _grouped_mm."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass
class _GroupedWeightStorage:
    pattern: str
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    dtype: torch.dtype
    device: torch.device
    base_gate: Optional[torch.Tensor] = None


def _ensure_grouped_weights(
    experts_module, expert_impls: List[torch.nn.Module], sample_mod: torch.nn.Module
) -> _GroupedWeightStorage:
    storage: Optional[_GroupedWeightStorage] = getattr(
        experts_module, "_ax_grouped_storage", None
    )

    def _store(new_storage: _GroupedWeightStorage) -> _GroupedWeightStorage:
        experts_module._ax_grouped_storage = new_storage
        return new_storage

    # Identify expert parameter layout
    if (
        hasattr(sample_mod, "w1")
        and hasattr(sample_mod, "w3")
        and hasattr(sample_mod, "w2")
    ):
        pattern = "swi_glu"
        if (
            storage is not None
            and storage.pattern == pattern
            and storage.dtype == sample_mod.w1.weight.dtype
            and storage.device == sample_mod.w1.weight.device
        ):
            return storage

        num_experts = len(expert_impls)
        w1_shape = sample_mod.w1.weight.shape
        w3_shape = sample_mod.w3.weight.shape
        w2_shape = sample_mod.w2.weight.shape
        gate = torch.empty(
            (num_experts, *w1_shape),
            device=sample_mod.w1.weight.device,
            dtype=sample_mod.w1.weight.dtype,
        )
        up = torch.empty(
            (num_experts, *w3_shape),
            device=sample_mod.w3.weight.device,
            dtype=sample_mod.w3.weight.dtype,
        )
        down = torch.empty(
            (num_experts, *w2_shape),
            device=sample_mod.w2.weight.device,
            dtype=sample_mod.w2.weight.dtype,
        )
        with torch.no_grad():
            for idx, mod in enumerate(expert_impls):
                gate[idx].copy_(mod.w1.weight.detach())
                up[idx].copy_(mod.w3.weight.detach())
                down[idx].copy_(mod.w2.weight.detach())
                mod.w1.weight.detach_()
                mod.w1.weight.set_(gate[idx])
                mod.w3.weight.detach_()
                mod.w3.weight.set_(up[idx])
                mod.w2.weight.detach_()
                mod.w2.weight.set_(down[idx])

        return _store(
            _GroupedWeightStorage(
                pattern=pattern,
                gate=gate,
                up=up,
                down=down,
                dtype=gate.dtype,
                device=gate.device,
            )
        )

    if hasattr(sample_mod, "gate_up_proj") and hasattr(sample_mod, "down_proj"):
        pattern = "fused_gate_up"
        gate_weight = sample_mod.gate_up_proj.weight
        down_weight = sample_mod.down_proj.weight
        if (
            storage is not None
            and storage.pattern == pattern
            and storage.dtype == gate_weight.dtype
            and storage.device == gate_weight.device
        ):
            return storage

        num_experts = len(expert_impls)
        gate_full = torch.empty(
            (num_experts, *gate_weight.shape),
            device=gate_weight.device,
            dtype=gate_weight.dtype,
        )
        down = torch.empty(
            (num_experts, *down_weight.shape),
            device=down_weight.device,
            dtype=down_weight.dtype,
        )
        with torch.no_grad():
            for idx, mod in enumerate(expert_impls):
                gate_full[idx].copy_(mod.gate_up_proj.weight.detach())
                down[idx].copy_(mod.down_proj.weight.detach())
                mod.gate_up_proj.weight.detach_()
                mod.gate_up_proj.weight.set_(gate_full[idx])
                mod.down_proj.weight.detach_()
                mod.down_proj.weight.set_(down[idx])

        inter = gate_weight.shape[0] // 2
        gate = gate_full[:, :inter]
        up = gate_full[:, inter:]
        return _store(
            _GroupedWeightStorage(
                pattern=pattern,
                gate=gate,
                up=up,
                down=down,
                dtype=gate.dtype,
                device=gate.device,
                base_gate=gate_full,
            )
        )

    if (
        hasattr(sample_mod, "up_proj")
        and hasattr(sample_mod, "gate_proj")
        and hasattr(sample_mod, "down_proj")
    ):
        pattern = "dual_proj"
        if (
            storage is not None
            and storage.pattern == pattern
            and storage.dtype == sample_mod.up_proj.weight.dtype
            and storage.device == sample_mod.up_proj.weight.device
        ):
            return storage

        num_experts = len(expert_impls)
        up_weight = sample_mod.up_proj.weight
        gate_weight = sample_mod.gate_proj.weight
        down_weight = sample_mod.down_proj.weight
        up = torch.empty(
            (num_experts, *up_weight.shape),
            device=up_weight.device,
            dtype=up_weight.dtype,
        )
        gate = torch.empty(
            (num_experts, *gate_weight.shape),
            device=gate_weight.device,
            dtype=gate_weight.dtype,
        )
        down = torch.empty(
            (num_experts, *down_weight.shape),
            device=down_weight.device,
            dtype=down_weight.dtype,
        )
        with torch.no_grad():
            for idx, mod in enumerate(expert_impls):
                up[idx].copy_(mod.up_proj.weight.detach())
                gate[idx].copy_(mod.gate_proj.weight.detach())
                down[idx].copy_(mod.down_proj.weight.detach())
                mod.up_proj.weight.detach_()
                mod.up_proj.weight.set_(up[idx])
                mod.gate_proj.weight.detach_()
                mod.gate_proj.weight.set_(gate[idx])
                mod.down_proj.weight.detach_()
                mod.down_proj.weight.set_(down[idx])

        return _store(
            _GroupedWeightStorage(
                pattern=pattern,
                gate=gate,
                up=up,
                down=down,
                dtype=gate.dtype,
                device=gate.device,
            )
        )

    raise RuntimeError(
        "torch_grouped: unsupported expert module layout for grouped weights"
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

    expert_impls = _iter_expert_impls(experts_module)
    sample_mod = expert_impls[0]
    storage = _ensure_grouped_weights(experts_module, expert_impls, sample_mod)
    w_gate = storage.gate
    w_up = storage.up
    w2 = storage.down

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

    perm = torch.argsort(flat_idx, stable=True)
    sorted_experts = flat_idx[perm]
    assignments = torch.bincount(sorted_experts, minlength=num_experts)
    if assignments.sum() == 0:
        zero = torch.zeros_like(x_flat)
        return zero.view(bsz, seqlen, hdim), router_logits

    token_indices_sorted = perm // top_k
    scores_sorted = topk_weight.view(-1)[perm]

    gather_index = token_indices_sorted.unsqueeze(-1).expand(-1, hdim)
    routed_input = torch.gather(x_flat, 0, gather_index).contiguous()

    active_idx = torch.nonzero(assignments, as_tuple=False).squeeze(-1)
    if active_idx.numel() == 0:
        zero = torch.zeros_like(x_flat)
        return zero.view(bsz, seqlen, hdim), router_logits

    counts_active = assignments[active_idx]
    offsets = torch.cumsum(counts_active.to(device=device, dtype=torch.int32), dim=0)
    offsets = offsets.to(torch.int32)
    if offsets[-1].item() == 0:
        zero = torch.zeros_like(x_flat)
        return zero.view(bsz, seqlen, hdim), router_logits

    w_gate_t = w_gate[active_idx].transpose(-2, -1).contiguous()
    w_up_t = w_up[active_idx].transpose(-2, -1).contiguous()
    w2_t = w2[active_idx].transpose(-2, -1).contiguous()

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
