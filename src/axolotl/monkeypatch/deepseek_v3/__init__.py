"""Monkeypatches for DeepSeek V3 MoE to use Triton contiguous grouped GEMM kernels."""

from __future__ import annotations

import contextlib
from typing import Callable

import torch

from axolotl.kernels.moe import ContiguousGroupedGEMM

_GROUP_SIZE_M = 128
_COMBINED_SUBMODULES = ("gate_proj", "up_proj", "down_proj")


def _is_triton_eligible(hidden_states: torch.Tensor) -> bool:
    return hidden_states.is_cuda and hidden_states.shape[0] > 0


def _ensure_combined_expert_weights(
    module, dtype: torch.dtype, device: torch.device
) -> None:
    if not hasattr(module, "_axolotl_original_specs"):
        module._axolotl_original_specs = {}
    if getattr(module, "_axolotl_combined_weights", False):
        # Move cached combined weights to the working dtype/device if required.
        for name in _COMBINED_SUBMODULES:
            param_name = f"{name}_weight"
            param = module.get_parameter(param_name)
            if param.device != device or param.dtype != dtype:
                module._parameters[param_name] = torch.nn.Parameter(
                    param.to(device=device, dtype=dtype).contiguous()
                )
        module._axolotl_combined_dtype = dtype
        module._axolotl_combined_device = device
        return

    combined = {}
    for name in _COMBINED_SUBMODULES:
        weights = []
        orig_device = None
        orig_dtype = None
        for expert in module.experts:
            lin = expert.get_submodule(name)
            weight_param = lin._parameters.get("weight")
            if weight_param is None:
                raise RuntimeError("Expected expert linear layers to have weights")
            if orig_device is None:
                orig_device = weight_param.device
                orig_dtype = weight_param.dtype
            weights.append(weight_param.detach().to(device=device, dtype=dtype))
            if "weight" in lin._parameters:
                del lin._parameters["weight"]
            if "bias" in lin._parameters:
                # DeepseekV3 MLP layers are bias-free, but keep this for safety.
                del lin._parameters["bias"]
        combined[name] = torch.stack(weights, dim=0).contiguous()
        module.register_parameter(
            f"{name}_weight", torch.nn.Parameter(combined[name])
        )
        module._axolotl_original_specs[name] = (orig_device, orig_dtype)

    module._axolotl_combined_weights = True
    module._axolotl_combined_dtype = dtype
    module._axolotl_combined_device = device


def _restore_expert_weights(module) -> None:
    if not getattr(module, "_axolotl_combined_weights", False):
        return

    for name in _COMBINED_SUBMODULES:
        param_name = f"{name}_weight"
        combined = module._parameters.pop(param_name)
        orig_device, orig_dtype = module._axolotl_original_specs.get(name, (combined.device, combined.dtype))
        for idx, expert in enumerate(module.experts):
            lin = expert.get_submodule(name)
            lin._parameters["weight"] = torch.nn.Parameter(
                combined[idx].detach().clone().to(orig_device, dtype=orig_dtype)
            )

    module._axolotl_combined_weights = False
    module._axolotl_combined_dtype = None
    module._axolotl_combined_device = None


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

    _ensure_combined_expert_weights(module, hidden_dtype, device)

    gate_weights = module.get_parameter("gate_proj_weight")
    up_weights = module.get_parameter("up_proj_weight")
    down_weights = module.get_parameter("down_proj_weight")

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
        try:
            return _moe_triton_forward(
                self,
                hidden_states,
                topk_indices,
                topk_weights,
                group_size_m,
                original_moe,
            )
        except RuntimeError:
            _restore_expert_weights(self)
            return original_moe(self, hidden_states, topk_indices, topk_weights)

    DeepseekV3MoE.moe = patched_moe
    DeepseekV3MoE._axolotl_triton_patch = True
