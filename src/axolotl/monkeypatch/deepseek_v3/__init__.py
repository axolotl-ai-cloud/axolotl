"""Monkeypatches for DeepSeek V3 MoE to use Triton contiguous grouped GEMM kernels."""

from __future__ import annotations

from typing import Callable

import torch

from axolotl.kernels.moe import ContiguousGroupedGEMM
from axolotl.kernels.moe.indices import generate_permute_indices
from axolotl.utils.logging import get_logger

_GROUP_SIZE_M = 128
_COMBINED_SUBMODULES = ("gate_proj", "up_proj", "down_proj")

LOG = get_logger(__name__)


def _is_triton_eligible(hidden_states: torch.Tensor) -> bool:
    if not hidden_states.is_cuda or hidden_states.shape[0] == 0:
        return False
    major, _ = torch.cuda.get_device_capability(hidden_states.device)
    if major < 9:
        LOG.debug(
            "Skipping Triton MoE kernels: requires compute capability >= 90, found %s",
            major,
        )
        return False
    return True


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
        module.register_parameter(f"{name}_weight", torch.nn.Parameter(combined[name]))
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
        orig_device, orig_dtype = module._axolotl_original_specs.get(
            name, (combined.device, combined.dtype)
        )
        for idx, expert in enumerate(module.experts):
            lin = expert.get_submodule(name)
            lin._parameters["weight"] = torch.nn.Parameter(
                combined[idx].detach().clone().to(orig_device, dtype=orig_dtype)
            )

    module._axolotl_combined_weights = False
    module._axolotl_combined_dtype = None
    module._axolotl_combined_device = None
    module._axolotl_original_specs = {}


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

    if not getattr(module, "_axolotl_triton_logged", False):
        min_tokens = int(counts.min().item())
        max_tokens = int(counts.max().item())
        LOG.info(
            "DeepseekV3MoE Triton: tokens per expert (min=%s, max=%s, avg=%.1f) with group_size=%s",
            min_tokens,
            max_tokens,
            total_actual / max(1, num_experts),
            group_size_m,
        )
        module._axolotl_triton_logged = True

    counts_int = counts.to(torch.int32)
    aligned_counts = (
        (torch.clamp_min(counts_int, group_size_m) + group_size_m - 1) // group_size_m
    ) * group_size_m
    max_len = int(aligned_counts.sum().item())

    permuted_indices, m_sizes, m_offsets = generate_permute_indices(
        counts_int.to(device),
        experts_per_rank=num_experts,
        num_ranks=1,
        max_len=max_len,
        alignment=group_size_m,
        use_cpu=not hidden_states.is_cuda,
    )

    if permuted_indices.device != device:
        permuted_indices = permuted_indices.to(device)
    if m_sizes.device != device:
        m_sizes = m_sizes.to(device)
    if m_offsets.device != device:
        m_offsets = m_offsets.to(device)

    permuted_indices_long = permuted_indices.to(torch.int64)
    valid_mask = permuted_indices_long >= 0
    valid_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    source_indices = permuted_indices_long[valid_mask]
    padded_positions = torch.nonzero(~valid_mask, as_tuple=False).squeeze(-1)

    grouped_hidden = hidden_states.new_empty((max_len, hidden_dim))
    if valid_positions.numel() > 0:
        grouped_hidden.index_copy_(
            0,
            valid_positions,
            sorted_hidden.index_select(0, source_indices),
        )
    if valid_positions.numel() < max_len:
        grouped_hidden.index_fill_(0, padded_positions, 0)

    expert_index_tensor = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int32),
        m_sizes.to(torch.int64),
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
    if valid_positions.numel() > 0:
        gate_valid = gate_out.index_select(0, valid_positions).to(hidden_dtype)
        up_valid = up_out.index_select(0, valid_positions).to(hidden_dtype)
        hidden_concat = act_fn(gate_valid) * up_valid
    else:
        hidden_concat = torch.empty(
            (0, gate_out.shape[-1]), device=device, dtype=hidden_dtype
        )

    intermediate_dim = hidden_concat.shape[-1]
    hidden_grouped = hidden_states.new_empty((max_len, intermediate_dim))
    if valid_positions.numel() > 0:
        hidden_grouped.index_copy_(0, valid_positions, hidden_concat)
    if valid_positions.numel() < max_len:
        hidden_grouped.index_fill_(0, padded_positions, 0)

    down_out = ContiguousGroupedGEMM.apply(
        hidden_grouped,
        down_weights,
        expert_index_tensor,
        group_size_m,
    )

    if valid_positions.numel() > 0:
        down_valid = down_out.index_select(0, valid_positions).to(hidden_dtype)
    else:
        down_valid = torch.empty(
            (0, down_out.shape[-1]), device=device, dtype=hidden_dtype
        )

    sorted_outputs = hidden_states.new_zeros((total_actual, hidden_dim))
    if down_valid.numel() > 0:
        sorted_outputs.index_copy_(0, source_indices, down_valid)

    expanded_output = expanded_hidden.new_empty(expanded_hidden.shape)
    expanded_output.index_copy_(0, sort_perm, sorted_outputs)
    expert_outputs = expanded_output.view(num_tokens, top_k, hidden_dim)

    weighted = expert_outputs * topk_weights.unsqueeze(-1).to(hidden_dtype)
    return weighted.sum(dim=1)


def patch_deepseek_v3_moe(group_size_m: int = _GROUP_SIZE_M) -> None:
    """Patch HuggingFace DeepseekV3MoE to use Triton contiguous group GEMM kernels."""

    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    # Record the unpatched implementation so callers can access a true baseline even
    # after the Triton patch has been applied (e.g. repeated microbenchmarks).
    if not hasattr(DeepseekV3MoE, "_axolotl_triton_original_moe"):
        DeepseekV3MoE._axolotl_triton_original_moe = DeepseekV3MoE.moe

    if getattr(DeepseekV3MoE, "_axolotl_triton_patch", False):
        return

    original_moe = DeepseekV3MoE._axolotl_triton_original_moe

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
        except Exception as err:  # fall back if Triton compilation or runtime fails
            if not getattr(self, "_axolotl_triton_warned", False):
                LOG.warning(
                    "DeepseekV3MoE Triton path failed; falling back to baseline: %s",
                    err,
                )
                self._axolotl_triton_warned = True
            _restore_expert_weights(self)
            return original_moe(self, hidden_states, topk_indices, topk_weights)

    DeepseekV3MoE.moe = patched_moe
    DeepseekV3MoE._axolotl_triton_patch = True
