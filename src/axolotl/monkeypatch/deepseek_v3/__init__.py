"""Monkeypatches for DeepSeek V3 MoE to use Triton contiguous grouped GEMM kernels."""

from __future__ import annotations

import contextlib
from typing import Callable

import torch

from axolotl.kernels.moe import ContiguousGroupedGEMM

_GROUP_SIZE_M = 128


def _is_triton_eligible(hidden_states: torch.Tensor) -> bool:
    return hidden_states.is_cuda and hidden_states.shape[0] > 0


def _collect_expert_weights(module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_weights = [expert.gate_proj.weight for expert in module.experts]
    up_weights = [expert.up_proj.weight for expert in module.experts]
    down_weights = [expert.down_proj.weight for expert in module.experts]
    gate = torch.stack(gate_weights, dim=0)
    up = torch.stack(up_weights, dim=0)
    down = torch.stack(down_weights, dim=0)
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
    total_actual = int(counts.sum().item())
    if total_actual == 0:
        return hidden_states.new_zeros_like(hidden_states)

    padded_counts = (
        (
            torch.where(
                counts > 0,
                counts,
                torch.full_like(counts, group_size_m),
            )
            + group_size_m
            - 1
        )
        // group_size_m
    ) * group_size_m

    total_padded = int(padded_counts.sum().item())
    grouped_hidden = hidden_states.new_zeros((total_padded, hidden_dim))

    write_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts
    actual_offsets = torch.cumsum(counts, dim=0) - counts

    repeated_offsets = torch.repeat_interleave(actual_offsets, counts)
    token_index = torch.arange(total_actual, device=device) - repeated_offsets
    dest_indices = write_offsets[sorted_assignments] + token_index

    grouped_hidden.index_copy_(0, dest_indices, sorted_hidden)
    padded_counts_idx = padded_counts.to(torch.int64)
    expert_index_tensor = (
        torch.arange(num_experts, device=device, dtype=torch.int64)
        .repeat_interleave(padded_counts_idx)
        .to(torch.int32)
        .contiguous()
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
    valid_gate = gate_out.index_select(0, dest_indices).to(hidden_dtype)
    valid_up = up_out.index_select(0, dest_indices).to(hidden_dtype)
    hidden_concat = act_fn(valid_gate) * valid_up

    intermediate_dim = hidden_concat.shape[-1]
    hidden_grouped = hidden_states.new_zeros((total_padded, intermediate_dim))
    hidden_grouped.index_copy_(0, dest_indices, hidden_concat)

    down_out = ContiguousGroupedGEMM.apply(
        hidden_grouped,
        down_weights,
        expert_index_tensor,
        group_size_m,
    )

    down_valid = down_out.index_select(0, dest_indices).to(hidden_dtype)

    expanded_output = expanded_hidden.new_empty(expanded_hidden.shape)
    expanded_output.index_copy_(0, sort_perm, down_valid)
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
