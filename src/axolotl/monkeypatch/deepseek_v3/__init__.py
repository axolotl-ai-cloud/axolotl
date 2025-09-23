"""Monkeypatches for DeepSeek V3 MoE to use Triton contiguous grouped GEMM kernels."""

from __future__ import annotations

from typing import Callable

import torch

from axolotl.kernels.moe import ContiguousGroupedGEMM
from axolotl.kernels.moe.indices import generate_permute_indices
from axolotl.kernels.moe.tt_mg_gemm import grouped_gemm_forward as mg_grouped_gemm
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
    module, dtype: torch.dtype, device: torch.device, backend: str
) -> None:
    if not hasattr(module, "_axolotl_original_specs"):
        module._axolotl_original_specs = {}
    if not hasattr(module, "_axolotl_mg_shapes"):
        module._axolotl_mg_shapes = {}

    prev_backend = getattr(module, "_axolotl_combined_backend", None)
    if getattr(module, "_axolotl_combined_weights", False):
        if prev_backend != backend:
            _restore_expert_weights(module)
        else:
            for name in _COMBINED_SUBMODULES:
                param_name = f"{name}_weight"
                param = module.get_parameter(param_name)
                if param.device != device or param.dtype != dtype:
                    module._parameters[param_name] = torch.nn.Parameter(
                        param.to(device=device, dtype=dtype).contiguous()
                    )
            module._axolotl_combined_dtype = dtype
            module._axolotl_combined_device = device
            module._axolotl_combined_backend = backend
            return

    module._axolotl_mg_shapes = {}
    for name in _COMBINED_SUBMODULES:
        weights = []
        orig_device = None
        orig_dtype = None
        orig_shape = None
        for expert in module.experts:
            lin = expert.get_submodule(name)
            weight_param = lin._parameters.get("weight")
            if weight_param is None:
                raise RuntimeError("Expected expert linear layers to have weights")
            if orig_device is None:
                orig_device = weight_param.device
                orig_dtype = weight_param.dtype
                orig_shape = tuple(weight_param.shape)
            weights.append(weight_param.detach().to(device=device, dtype=dtype))
            if "weight" in lin._parameters:
                del lin._parameters["weight"]
            if "bias" in lin._parameters:
                del lin._parameters["bias"]
        if backend == "cg":
            combined_weight = torch.stack(weights, dim=0).contiguous()
        else:
            combined_weight = torch.cat(weights, dim=0).contiguous()
            module._axolotl_mg_shapes[name] = orig_shape
        module.register_parameter(f"{name}_weight", torch.nn.Parameter(combined_weight))
        module._axolotl_original_specs[name] = (orig_device, orig_dtype, orig_shape)

    module._axolotl_combined_weights = True
    module._axolotl_combined_dtype = dtype
    module._axolotl_combined_device = device
    module._axolotl_combined_backend = backend


def _restore_expert_weights(module) -> None:
    if not getattr(module, "_axolotl_combined_weights", False):
        return

    for name in _COMBINED_SUBMODULES:
        param_name = f"{name}_weight"
        combined = module._parameters.pop(param_name)
        orig_device, orig_dtype, orig_shape = module._axolotl_original_specs.get(
            name, (combined.device, combined.dtype, None)
        )
        rows_per = orig_shape[0] if orig_shape else None
        for idx, expert in enumerate(module.experts):
            lin = expert.get_submodule(name)
            if combined.dim() == 3:
                slice_tensor = combined[idx]
            elif rows_per is not None:
                start = idx * rows_per
                end = start + rows_per
                slice_tensor = combined[start:end]
            else:
                raise RuntimeError(
                    "Unable to recover expert weight shape during restore"
                )
            lin._parameters["weight"] = torch.nn.Parameter(
                slice_tensor.detach().clone().to(orig_device, dtype=orig_dtype)
            )

    module._axolotl_combined_weights = False
    module._axolotl_combined_dtype = None
    module._axolotl_combined_device = None
    module._axolotl_combined_backend = None
    module._axolotl_original_specs = {}
    module._axolotl_mg_shapes = {}


def _run_cg_grouped_gemm(
    module,
    grouped_hidden: torch.Tensor,
    m_sizes: torch.Tensor,
    num_experts: int,
    group_size_m: int,
    hidden_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_combined_expert_weights(module, hidden_dtype, device, backend="cg")

    expert_index_tensor = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int32),
        m_sizes.to(torch.int64),
    )

    gate_weights = module.get_parameter("gate_proj_weight")
    if gate_weights.dim() == 2:
        out_dim = gate_weights.shape[0] // num_experts
        gate_weights = gate_weights.view(num_experts, out_dim, gate_weights.shape[1])

    up_weights = module.get_parameter("up_proj_weight")
    if up_weights.dim() == 2:
        out_dim = up_weights.shape[0] // num_experts
        up_weights = up_weights.view(num_experts, out_dim, up_weights.shape[1])

    down_weights = module.get_parameter("down_proj_weight")
    if down_weights.dim() == 2:
        out_dim = down_weights.shape[0] // num_experts
        down_weights = down_weights.view(num_experts, out_dim, down_weights.shape[1])

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
    down_out = ContiguousGroupedGEMM.apply(
        grouped_hidden,
        down_weights,
        expert_index_tensor,
        group_size_m,
    )

    return (
        gate_out.to(hidden_dtype),
        up_out.to(hidden_dtype),
        down_out.to(hidden_dtype),
    )

    gate_out = mg_grouped_gemm(
        grouped_hidden,
        module.get_parameter("gate_proj_weight"),
        m_sizes_tensor,
    )
    up_out = mg_grouped_gemm(
        grouped_hidden,
        module.get_parameter("up_proj_weight"),
        m_sizes_tensor,
    )
    down_out = mg_grouped_gemm(
        hidden_grouped,
        module.get_parameter("down_proj_weight"),
        m_sizes_tensor,
    )

    return (
        gate_out.to(hidden_dtype),
        up_out.to(hidden_dtype),
        down_out.to(hidden_dtype),
    )


def _moe_triton_forward(
    module,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    group_size_m: int,
    backend: str,
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

    permuted_indices, m_sizes, _ = generate_permute_indices(
        counts_int.to(device),
        experts_per_rank=num_experts,
        num_ranks=1,
        max_len=max_len,
        alignment=group_size_m,
        use_cpu=not hidden_states.is_cuda,
    )

    permuted_indices = permuted_indices.to(device)
    m_sizes = m_sizes.to(device)

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

    m_sizes_tensor = m_sizes.to(device=device, dtype=torch.int32)

    if backend == "mg":
        _ensure_combined_expert_weights(module, hidden_dtype, device, backend)
        gate_out = mg_grouped_gemm(
            grouped_hidden,
            module.get_parameter("gate_proj_weight"),
            m_sizes_tensor,
        ).to(hidden_dtype)
        up_out = mg_grouped_gemm(
            grouped_hidden,
            module.get_parameter("up_proj_weight"),
            m_sizes_tensor,
        ).to(hidden_dtype)
    else:
        gate_out, up_out, down_out_cg = _run_cg_grouped_gemm(
            module,
            grouped_hidden,
            m_sizes,
            num_experts,
            group_size_m,
            hidden_dtype,
            device,
        )

    act_fn: Callable[[torch.Tensor], torch.Tensor] = module.experts[0].act_fn
    if valid_positions.numel() > 0:
        gate_valid = gate_out.index_select(0, valid_positions)
        up_valid = up_out.index_select(0, valid_positions)
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

    if backend == "mg":
        down_out = mg_grouped_gemm(
            hidden_grouped,
            module.get_parameter("down_proj_weight"),
            m_sizes_tensor,
        ).to(hidden_dtype)
    else:
        down_out = down_out_cg

    if valid_positions.numel() > 0:
        down_valid = down_out.index_select(0, valid_positions)
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


def patch_deepseek_v3_moe(
    group_size_m: int = _GROUP_SIZE_M, backend: str = "mg"
) -> None:
    """Patch HuggingFace DeepseekV3MoE to use Triton contiguous group GEMM kernels."""

    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

    if backend not in {"cg", "mg"}:
        raise ValueError(f"Unsupported MoE kernel backend: {backend}")

    # Record the unpatched implementation so callers can access a true baseline even
    # after the Triton patch has been applied (e.g. repeated microbenchmarks).
    if not hasattr(DeepseekV3MoE, "_axolotl_triton_original_moe"):
        DeepseekV3MoE._axolotl_triton_original_moe = DeepseekV3MoE.moe

    if getattr(DeepseekV3MoE, "_axolotl_triton_patch", False):
        return

    original_moe = DeepseekV3MoE._axolotl_triton_original_moe
    DeepseekV3MoE._axolotl_triton_backend = backend
    DeepseekV3MoE._axolotl_group_size_m = group_size_m

    def patched_moe(self, hidden_states, topk_indices, topk_weights):
        backend_sel = getattr(self, "_axolotl_triton_backend", backend)
        group_size_sel = getattr(self, "_axolotl_group_size_m", group_size_m)
        try:
            return _moe_triton_forward(
                self,
                hidden_states,
                topk_indices,
                topk_weights,
                group_size_sel,
                backend_sel,
                original_moe,
            )
        except Exception as err:  # surface Triton failures explicitly
            _restore_expert_weights(self)
            LOG.error("DeepseekV3MoE Triton path failed: %s", err)
            raise

    DeepseekV3MoE.moe = patched_moe
    DeepseekV3MoE._axolotl_triton_patch = True
