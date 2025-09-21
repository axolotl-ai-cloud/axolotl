"""Monkeypatches for DeepSeek V3 MoE to use Triton contiguous grouped GEMM kernels."""

from __future__ import annotations

import contextlib
import math
from typing import Callable

import torch

from axolotl.kernels.moe import ContiguousGroupedGEMM

_GROUP_SIZE_M = 128


def _align_to(value: int, alignment: int) -> int:
    if value <= 0:
        return 0
    return math.ceil(value / alignment) * alignment


def _is_triton_eligible(hidden_states: torch.Tensor) -> bool:
    return hidden_states.is_cuda and hidden_states.shape[0] > 0


def _collect_expert_weights(module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_weights = []
    up_weights = []
    down_weights = []
    for expert in module.experts:
        gate_weights.append(expert.gate_proj.weight)
        up_weights.append(expert.up_proj.weight)
        down_weights.append(expert.down_proj.weight)
    gate = torch.stack(gate_weights, dim=0).contiguous()
    up = torch.stack(up_weights, dim=0).contiguous()
    down = torch.stack(down_weights, dim=0).contiguous()
    return gate, up, down


def _moe_triton_forward(
    module,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    group_size_m: int,
    fallback: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if not _is_triton_eligible(hidden_states):
        return fallback(hidden_states, topk_indices, topk_weights)

    device = hidden_states.device
    hidden_dtype = hidden_states.dtype
    num_tokens, hidden_dim = hidden_states.shape
    top_k = topk_indices.size(-1)

    expanded_hidden = hidden_states.repeat_interleave(top_k, dim=0)
    expert_assignments = topk_indices.reshape(-1)
    if expanded_hidden.numel() == 0:
        return hidden_states.new_zeros_like(hidden_states)

    sort_perm = torch.argsort(expert_assignments)
    sorted_hidden = expanded_hidden.index_select(0, sort_perm)
    sorted_assignments = expert_assignments.index_select(0, sort_perm)

    num_experts = len(module.experts)
    counts = torch.bincount(sorted_assignments, minlength=num_experts)
    counts_cpu = counts.to(torch.int64).cpu().tolist()
    padded_counts = [_align_to(c, group_size_m) for c in counts_cpu]

    total_actual = sum(counts_cpu)
    total_padded = sum(padded_counts)
    if total_actual == 0 or total_padded == 0:
        return hidden_states.new_zeros_like(hidden_states)

    actual_offsets = [0]
    padded_offsets = [0]
    for count, padded in zip(counts_cpu, padded_counts, strict=False):
        actual_offsets.append(actual_offsets[-1] + count)
        padded_offsets.append(padded_offsets[-1] + padded)

    grouped_hidden = hidden_states.new_zeros((total_padded, hidden_dim))
    expert_index_tensor = torch.empty(total_padded, dtype=torch.int32, device=device)

    for idx, (count, padded) in enumerate(zip(counts_cpu, padded_counts, strict=False)):
        dst_start = padded_offsets[idx]
        dst_end = dst_start + padded
        if padded == 0:
            continue
        expert_index_tensor[dst_start:dst_end] = idx
        if count > 0:
            src_start = actual_offsets[idx]
            src_end = src_start + count
            grouped_hidden[dst_start : dst_start + count].copy_(
                sorted_hidden[src_start:src_end]
            )

    gate_weights, up_weights, down_weights = _collect_expert_weights(module)

    gate_out = ContiguousGroupedGEMM.apply(
        grouped_hidden,
        gate_weights,
        expert_index_tensor,
        group_size_m,
    )
    up_out = ContiguousGroupedGEMM.apply(
        grouped_hidden,
        up_weights,
        expert_index_tensor,
        group_size_m,
    )

    act_fn: Callable[[torch.Tensor], torch.Tensor] = module.experts[0].act_fn

    hidden_chunks = []
    for idx, count in enumerate(counts_cpu):
        if count == 0:
            continue
        pad_start = padded_offsets[idx]
        pad_end = pad_start + count
        gate_slice = gate_out[pad_start:pad_end].to(hidden_dtype)
        up_slice = up_out[pad_start:pad_end].to(hidden_dtype)
        hidden_chunks.append(act_fn(gate_slice) * up_slice)

    hidden_concat = torch.cat(hidden_chunks, dim=0)

    intermediate_dim = hidden_concat.shape[-1]
    hidden_grouped = hidden_states.new_zeros((total_padded, intermediate_dim))

    for idx, count in enumerate(counts_cpu):
        if count == 0:
            continue
        pad_start = padded_offsets[idx]
        src_start = actual_offsets[idx]
        src_end = src_start + count
        hidden_grouped[pad_start : pad_start + count].copy_(
            hidden_concat[src_start:src_end]
        )

    down_out = ContiguousGroupedGEMM.apply(
        hidden_grouped,
        down_weights,
        expert_index_tensor,
        group_size_m,
    )

    down_chunks = []
    for idx, count in enumerate(counts_cpu):
        if count == 0:
            continue
        pad_start = padded_offsets[idx]
        pad_end = pad_start + count
        down_chunks.append(down_out[pad_start:pad_end].to(hidden_dtype))

    down_concat = torch.cat(down_chunks, dim=0)

    expanded_output = expanded_hidden.new_empty(expanded_hidden.shape)
    expanded_output.index_copy_(0, sort_perm, down_concat.to(hidden_dtype))
    expert_outputs = expanded_output.view(num_tokens, top_k, hidden_dim)

    weighted = expert_outputs * topk_weights.unsqueeze(-1).to(hidden_dtype)
    return weighted.sum(dim=1)


def patch_deepseek_v3_moe(group_size_m: int = _GROUP_SIZE_M) -> None:
    """Patch HuggingFace DeepseekV3MoE to use Triton contiguous group GEMM kernels."""

    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    if getattr(DeepseekV3MoE, "_axolotl_triton_patch", False):
        return

    original_moe = DeepseekV3MoE.moe

    def patched_moe(self, hidden_states, topk_indices, topk_weights):
        with contextlib.suppress(RuntimeError):
            return _moe_triton_forward(
                self,
                hidden_states,
                topk_indices,
                topk_weights,
                group_size_m,
                original_moe,
            )
        return original_moe(self, hidden_states, topk_indices, topk_weights)

    DeepseekV3MoE.moe = patched_moe
    DeepseekV3MoE._axolotl_triton_patch = True
