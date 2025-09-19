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


def _iter_expert_impls(
    experts_module, visited: Optional[set[int]] = None
) -> List[torch.nn.Module]:
    if visited is None:
        visited = set()
    module_id = id(experts_module)
    if module_id in visited:
        return []
    visited.add(module_id)

    impls: List[torch.nn.Module] = []
    for exp in experts_module:
        candidate = getattr(exp, "mlp", getattr(exp, "ffn", exp))
        if hasattr(candidate, "gate_proj") and hasattr(candidate, "up_proj"):
            impls.append(candidate)
            continue
        nested = getattr(candidate, "experts", None)
        if nested is not None:
            impls.extend(_iter_expert_impls(nested, visited))
            continue
        raise RuntimeError(
            "torch_grouped: unable to resolve expert implementation for module"
        )
    return impls


@dataclass
class _GroupedWeightStorage:
    pattern: str
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    fused_gate_up: torch.Tensor
    dtype: torch.dtype
    device: torch.device


def _allocate_fused_gate_up(
    num_experts: int,
    gate_shape: torch.Size,
    up_shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gate_shape[1] != up_shape[1]:
        raise RuntimeError(
            "torch_grouped: gate and up projections must share the hidden dimension"
        )

    fused = torch.empty(
        (num_experts, gate_shape[0] + up_shape[0], gate_shape[1]),
        device=device,
        dtype=dtype,
    )
    gate_view = fused[:, : gate_shape[0]]
    up_view = fused[:, gate_shape[0] : gate_shape[0] + up_shape[0]]
    return fused, gate_view, up_view


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
        num_experts = len(expert_impls)
        w1_shape = sample_mod.w1.weight.shape
        w3_shape = sample_mod.w3.weight.shape
        w2_shape = sample_mod.w2.weight.shape
        if (
            storage is not None
            and storage.pattern == pattern
            and storage.dtype == sample_mod.w1.weight.dtype
            and storage.device == sample_mod.w1.weight.device
            and storage.gate.shape[1:] == w1_shape
        ):
            return storage

        fused, gate, up = _allocate_fused_gate_up(
            num_experts,
            w1_shape,
            w3_shape,
            device=sample_mod.w1.weight.device,
            dtype=sample_mod.w1.weight.dtype,
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
                fused_gate_up=fused,
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
            and storage.gate.shape[1:]
            == (gate_weight.shape[0] // 2, gate_weight.shape[1])
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
                fused_gate_up=gate_full,
                dtype=gate.dtype,
                device=gate.device,
            )
        )

    if (
        hasattr(sample_mod, "up_proj")
        and hasattr(sample_mod, "gate_proj")
        and hasattr(sample_mod, "down_proj")
    ):
        pattern = "dual_proj"
        up_weight = sample_mod.up_proj.weight
        gate_weight = sample_mod.gate_proj.weight
        down_weight = sample_mod.down_proj.weight
        if (
            storage is not None
            and storage.pattern == pattern
            and storage.dtype == sample_mod.up_proj.weight.dtype
            and storage.device == sample_mod.up_proj.weight.device
            and storage.gate.shape[1:] == gate_weight.shape
        ):
            return storage

        num_experts = len(expert_impls)
        fused, gate, up = _allocate_fused_gate_up(
            num_experts,
            gate_weight.shape,
            up_weight.shape,
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
                gate[idx].copy_(mod.gate_proj.weight.detach())
                up[idx].copy_(mod.up_proj.weight.detach())
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
                fused_gate_up=fused,
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

    parent_block = None
    parent_ref = getattr(experts_module, "_ax_parent_block_ref", None)
    if parent_ref is not None:
        try:
            parent_block = parent_ref()
        except TypeError:
            parent_block = None

    expert_container = getattr(experts_module, "experts", experts_module)

    expert_impls = _iter_expert_impls(expert_container)
    sample_mod = expert_impls[0]
    storage = _ensure_grouped_weights(expert_container, expert_impls, sample_mod)
    w_gate = storage.gate
    w_up = storage.up
    w2 = storage.down

    x_flat = hidden_states.view(tokens, hdim).to(expert_dtype)
    router_logits = gate_linear(x_flat.to(routing_dtype))

    shared_out_flat: Optional[torch.Tensor] = None
    shared_owner = parent_block if parent_block is not None else experts_module
    if hasattr(shared_owner, "shared_expert"):
        shared_expert = shared_owner.shared_expert
        shared_out_flat = shared_expert(x_flat)
        shared_out_flat = shared_out_flat.to(expert_dtype)
        shared_gate = getattr(shared_owner, "shared_expert_gate", None)
        if shared_gate is not None:
            gate_input = shared_gate(x_flat.to(shared_gate.weight.dtype))
            gate_vals = torch.sigmoid(gate_input)
            shared_out_flat.mul_(gate_vals.to(expert_dtype))

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, top_k, dim=-1, sorted=False)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    flat_idx = topk_idx.view(-1)
    num_experts = len(expert_impls)
    if flat_idx.numel() == 0:
        zero = torch.zeros_like(x_flat)
        return zero.view(bsz, seqlen, hdim), router_logits

    sorted_experts, perm = torch.sort(flat_idx)
    assignments = torch.bincount(sorted_experts, minlength=num_experts)
    if assignments.sum() == 0:
        zero = torch.zeros_like(x_flat)
        return zero.view(bsz, seqlen, hdim), router_logits

    token_indices_sorted = torch.div(perm, top_k, rounding_mode="floor").contiguous()
    scores_sorted = topk_weight.reshape(-1).index_select(0, perm)

    gather_index = token_indices_sorted.unsqueeze(-1).expand(-1, hdim)
    routed_input = torch.gather(x_flat, 0, gather_index)

    counts_i32 = assignments.to(device=device, dtype=torch.int32)
    offsets = torch.cumsum(counts_i32, dim=0).to(dtype=torch.int32)
    mm_dtype = torch.bfloat16 if expert_dtype == torch.bfloat16 else expert_dtype
    routed_in = routed_input.to(mm_dtype)
    w_gate_t = w_gate.transpose(-2, -1).to(mm_dtype)
    w_up_t = w_up.transpose(-2, -1).to(mm_dtype)
    w2_t = w2.transpose(-2, -1).to(mm_dtype)

    routed_in = routed_in.contiguous()
    w_gate_t = w_gate_t.contiguous()
    gate_out = torch._grouped_mm(routed_in, w_gate_t, offs=offsets)
    torch.ops.aten.silu_(gate_out)
    w_up_t = w_up_t.contiguous()
    up_out = torch._grouped_mm(routed_in, w_up_t, offs=offsets)
    gate_out.mul_(up_out)
    gate_out = gate_out.contiguous()
    w2_t = w2_t.contiguous()
    down_out = torch._grouped_mm(gate_out, w2_t, offs=offsets).to(expert_dtype)

    weights = scores_sorted.unsqueeze(-1).to(expert_dtype)
    down_out.mul_(weights)

    combined = torch.zeros_like(x_flat)
    combined.scatter_add_(0, gather_index, down_out)

    output = combined.view(bsz, seqlen, hdim)
    if shared_out_flat is not None:
        output = output + shared_out_flat.view(bsz, seqlen, hdim)
    return output, router_logits
