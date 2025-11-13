"""Grouped MoE kernels for GLM4 MoE architectures."""

from __future__ import annotations

from types import MethodType
from typing import Iterable, Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from axolotl.monkeypatch.models.utils import clone_expert_parameter
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

try:
    # Reuse the shared MegaBlocks loader so both adapters hit the same cache.
    from axolotl.monkeypatch.models.bailing_moe_v2.modeling import (  # noqa: WPS433
        _load_megablocks_backend,
    )
except ImportError:  # pragma: no cover - MegaBlocks optional dependency

    def _load_megablocks_backend() -> Optional[object]:
        return None


class Glm4MoeGroupedExperts(nn.Module):
    """Vectorised grouped expert MLP for GLM4 MoE blocks."""

    def __init__(self, config, experts: Iterable[nn.Module], backend_impl: str):
        super().__init__()
        experts = list(experts)
        if not experts:
            raise ValueError("Expected at least one expert module to convert.")

        self.config = config
        self.hidden_size = getattr(config, "hidden_size", None)
        if self.hidden_size is None:
            raise ValueError("GLM4 config missing hidden_size.")
        self.intermediate_size = getattr(config, "moe_intermediate_size", None)
        if self.intermediate_size is None:
            raise ValueError("GLM4 config missing moe_intermediate_size.")
        self.num_experts = getattr(config, "n_routed_experts", len(experts))
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]
        self.backend_impl = backend_impl
        self._warned_grouped_mm_fallback = False
        self._warned_megablocks_missing = False

        device = experts[0].gate_proj.weight.device
        dtype = experts[0].gate_proj.weight.dtype

        gate_dims = tuple(
            getattr(experts[0].gate_proj.weight, "ds_shape", experts[0].gate_proj.weight.shape)
        )
        up_dims = tuple(
            getattr(experts[0].up_proj.weight, "ds_shape", experts[0].up_proj.weight.shape)
        )
        down_dims = tuple(
            getattr(experts[0].down_proj.weight, "ds_shape", experts[0].down_proj.weight.shape)
        )

        gate_weights: list[torch.Tensor] = []
        up_weights: list[torch.Tensor] = []
        down_weights: list[torch.Tensor] = []
        gate_bias = experts[0].gate_proj.bias
        up_bias = experts[0].up_proj.bias
        down_bias = experts[0].down_proj.bias

        if len(experts) != self.num_experts:
            raise ValueError(
                f"Config expects {self.num_experts} experts, but received {len(experts)}."
            )

        for idx, expert in enumerate(experts):
            if (
                expert.gate_proj.weight.numel() == 0
                or expert.up_proj.weight.numel() == 0
                or expert.down_proj.weight.numel() == 0
            ):
                LOG.warning(
                    "Encountered expert %s with empty weights during GLM4 grouped conversion; "
                    "gate=%s up=%s down=%s (dtype=%s device=%s)",
                    idx,
                    tuple(expert.gate_proj.weight.shape),
                    tuple(expert.up_proj.weight.shape),
                    tuple(expert.down_proj.weight.shape),
                    expert.gate_proj.weight.dtype,
                    expert.gate_proj.weight.device,
                )
            gate_weights.append(clone_expert_parameter(expert.gate_proj.weight))
            up_weights.append(clone_expert_parameter(expert.up_proj.weight))
            down_weights.append(clone_expert_parameter(expert.down_proj.weight))

        gate_stack = torch.stack(gate_weights, dim=0).to(device=device, dtype=dtype)
        up_stack = torch.stack(up_weights, dim=0).to(device=device, dtype=dtype)
        down_stack = torch.stack(down_weights, dim=0).to(device=device, dtype=dtype)

        if gate_stack.numel() == 0 and all(dim > 0 for dim in gate_dims):
            gate_stack = torch.zeros(
                (self.num_experts,) + gate_dims,
                device=device,
                dtype=dtype,
            )
        if up_stack.numel() == 0 and all(dim > 0 for dim in up_dims):
            up_stack = torch.zeros(
                (self.num_experts,) + up_dims,
                device=device,
                dtype=dtype,
            )
        if down_stack.numel() == 0 and all(dim > 0 for dim in down_dims):
            down_stack = torch.zeros(
                (self.num_experts,) + down_dims,
                device=device,
                dtype=dtype,
            )

        self.gate_weight = nn.Parameter(gate_stack)
        self.up_weight = nn.Parameter(up_stack)
        self.down_weight = nn.Parameter(down_stack)

        if gate_bias is not None:
            stacked = torch.stack(
                [clone_expert_parameter(expert.gate_proj.bias) for expert in experts],
                dim=0,
            ).to(device=device, dtype=gate_bias.dtype)
            self.gate_bias = nn.Parameter(stacked)
        else:
            self.register_parameter("gate_bias", None)

        if up_bias is not None:
            stacked = torch.stack(
                [clone_expert_parameter(expert.up_proj.bias) for expert in experts],
                dim=0,
            ).to(device=device, dtype=up_bias.dtype)
            self.up_bias = nn.Parameter(stacked)
        else:
            self.register_parameter("up_bias", None)

        if down_bias is not None:
            stacked = torch.stack(
                [clone_expert_parameter(expert.down_proj.bias) for expert in experts],
                dim=0,
            ).to(device=device, dtype=down_bias.dtype)
            self.down_bias = nn.Parameter(stacked)
        else:
            self.register_parameter("down_bias", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected (batch, seq, hidden) hidden_states but received shape {tuple(hidden_states.shape)}."
            )

        bsz, seq_len, hidden = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden)
        num_tokens = hidden_flat.size(0)
        if num_tokens == 0:
            return hidden_states

        topk = topk_idx.shape[-1]
        dispatch_indices = torch.arange(
            num_tokens,
            device=hidden_flat.device,
            dtype=torch.long,
        ).repeat_interleave(topk)

        flat_expert = topk_idx.reshape(-1)
        flat_weight = topk_weight.reshape(-1).to(hidden_states.dtype)

        active_counts = torch.bincount(flat_expert, minlength=self.num_experts)
        active_expert_indices = torch.nonzero(active_counts, as_tuple=False).flatten()
        if active_expert_indices.numel() == 0:
            return hidden_states.new_zeros((bsz, seq_len, hidden))

        counts = active_counts.index_select(0, active_expert_indices)
        repeat_sizes = counts.to(torch.long)
        offsets = torch.cumsum(
            counts.to(dtype=torch.int32),
            dim=0,
            dtype=torch.int32,
        ).contiguous()

        _, sort_perm = torch.sort(flat_expert)
        sorted_tokens = dispatch_indices.index_select(0, sort_perm)
        expert_inputs = hidden_flat.index_select(0, sorted_tokens).to(self.gate_weight.dtype)

        gate_active = self.gate_weight.index_select(0, active_expert_indices)
        up_active = self.up_weight.index_select(0, active_expert_indices)
        down_active = self.down_weight.index_select(0, active_expert_indices)

        gate_bias_active = (
            self.gate_bias.index_select(0, active_expert_indices) if self.gate_bias is not None else None
        )
        up_bias_active = (
            self.up_bias.index_select(0, active_expert_indices) if self.up_bias is not None else None
        )
        down_bias_active = (
            self.down_bias.index_select(0, active_expert_indices) if self.down_bias is not None else None
        )

        def _fallback_linear(
            inputs: torch.Tensor,
            weight_bank: torch.Tensor,
            bias_bank: torch.Tensor | None,
        ) -> torch.Tensor:
            outputs: list[torch.Tensor] = []
            start_idx = 0
            bias_iter = bias_bank if bias_bank is not None else [None] * weight_bank.size(0)
            for count, weight, bias in zip(
                repeat_sizes.tolist(),
                weight_bank,
                bias_iter,
                strict=True,
            ):
                if count == 0:
                    continue
                end_idx = start_idx + count
                chunk = inputs[start_idx:end_idx]
                result = chunk @ weight.transpose(0, 1)
                if bias is not None:
                    result = result + bias
                outputs.append(result)
                start_idx = end_idx
            if outputs:
                return torch.cat(outputs, dim=0)
            return inputs.new_empty((0, weight_bank.size(1)))

        def _run_linear(
            inputs: torch.Tensor,
            weight_bank: torch.Tensor,
            bias_bank: torch.Tensor | None,
        ) -> torch.Tensor:
            weight_t = weight_bank.transpose(-2, -1).contiguous()
            batch_sizes_cpu = repeat_sizes.to(device="cpu", dtype=torch.long)

            use_megablocks = self.backend_impl == "megablocks"
            if use_megablocks:
                mb_backend = _load_megablocks_backend()
                if mb_backend is None:
                    if not self._warned_megablocks_missing:
                        LOG.warning(
                            "Requested mlp_impl=megablocks but MegaBlocks backend not found; "
                            "falling back to torch grouped_mm / Python loops."
                        )
                        self._warned_megablocks_missing = True
                    use_megablocks = False
                elif (
                    not inputs.is_cuda
                    or inputs.dtype != torch.bfloat16
                    or weight_t.dtype != torch.bfloat16
                ):
                    if not self._warned_megablocks_missing:
                        LOG.warning(
                            "MegaBlocks backend requires CUDA tensors in bf16; "
                            "falling back to torch grouped_mm / Python loops."
                        )
                        self._warned_megablocks_missing = True
                    use_megablocks = False
                else:
                    outputs = mb_backend.gmm(
                        inputs,
                        weight_t,
                        batch_sizes_cpu,
                        trans_b=False,
                    )
                    if bias_bank is not None:
                        outputs = outputs + torch.repeat_interleave(
                            bias_bank,
                            repeat_sizes,
                            dim=0,
                        ).to(outputs.dtype)
                    return outputs

            if inputs.is_cuda and hasattr(torch, "_grouped_mm"):
                try:
                    outputs = torch._grouped_mm(inputs, weight_t, offs=offsets)
                    if bias_bank is not None:
                        expanded = torch.repeat_interleave(
                            bias_bank,
                            repeat_sizes,
                            dim=0,
                        ).to(outputs.dtype)
                        outputs = outputs + expanded
                    return outputs
                except (RuntimeError, NotImplementedError) as exc:
                    if isinstance(exc, RuntimeError) and "grouped gemm is not supported" not in str(exc).lower():
                        raise
                    mb_backend = _load_megablocks_backend()
                    if (
                        mb_backend is not None
                        and inputs.is_cuda
                        and inputs.dtype == torch.bfloat16
                        and weight_t.dtype == torch.bfloat16
                    ):
                        outputs = mb_backend.gmm(
                            inputs,
                            weight_t,
                            batch_sizes_cpu,
                            trans_b=False,
                        )
                        if bias_bank is not None:
                            outputs = outputs + torch.repeat_interleave(
                                bias_bank,
                                repeat_sizes,
                                dim=0,
                            ).to(outputs.dtype)
                        return outputs
                    if not self._warned_grouped_mm_fallback:
                        LOG.warning(
                            "torch._grouped_mm unavailable on this backend (%s); falling back to per-expert matmuls.",
                            exc,
                        )
                        self._warned_grouped_mm_fallback = True

            return _fallback_linear(inputs, weight_bank, bias_bank)

        gate_out = _run_linear(expert_inputs, gate_active, gate_bias_active)
        up_out = _run_linear(expert_inputs, up_active, up_bias_active)

        activated = self.act_fn(gate_out).to(up_out.dtype)
        activated = activated * up_out

        combined = _run_linear(activated, down_active, down_bias_active)

        inverse_perm = torch.empty_like(sort_perm)
        inverse_perm.index_copy_(
            0,
            sort_perm,
            torch.arange(
                sort_perm.size(0),
                device=sort_perm.device,
                dtype=sort_perm.dtype,
            ),
        )
        unsorted_outputs = combined.index_select(0, inverse_perm)
        weighted = unsorted_outputs.to(hidden_states.dtype) * flat_weight.unsqueeze(-1)

        output_flat = hidden_states.new_zeros((num_tokens, hidden))
        scatter_index = dispatch_indices.unsqueeze(-1).expand_as(weighted)
        output_flat.scatter_add_(0, scatter_index, weighted)

        return output_flat.view(bsz, seq_len, hidden)


def _grouped_moe(self, hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor):
    hidden_rank = hidden_states.dim()
    if hidden_rank == 2:
        reshaped = hidden_states.view(1, hidden_states.size(0), hidden_states.size(1))
    elif hidden_rank == 3:
        reshaped = hidden_states
    else:
        raise ValueError(f"Unexpected hidden_states rank {hidden_rank} for GLM4 MoE grouped kernel.")

    expert_out = self.experts(reshaped, topk_idx, topk_weight)
    return expert_out.view(-1, expert_out.size(-1))


def patch_glm4_moe_grouped_experts(model: nn.Module, mlp_impl: str = "grouped") -> int:
    """Convert Glm4MoeMoE modules to grouped expert implementations."""
    if mlp_impl not in {"grouped", "megablocks"}:
        raise ValueError(f"Unsupported mlp_impl={mlp_impl} for GLM4 grouped experts.")

    if mlp_impl == "grouped" and not hasattr(torch, "_grouped_mm"):
        raise RuntimeError(
            "torch._grouped_mm is required for grouped MoE kernels but is unavailable in this torch build."
        )

    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE
    except ImportError:
        LOG.warning("transformers.models.glm4_moe not available; skipping grouped patch.")
        return 0

    patched = 0
    for module in model.modules():
        if not isinstance(module, Glm4MoeMoE):
            continue
        if getattr(module, "_axolotl_grouped_moe", False):
            patched += 1
            continue

        old_experts = module.experts
        try:
            grouped = Glm4MoeGroupedExperts(module.config, old_experts, backend_impl=mlp_impl)
        except Exception as exc:  # pragma: no cover - defensive safeguard
            LOG.warning("Failed to convert GLM4 MoE block to grouped experts: %s", exc)
            continue

        module.experts = grouped
        module.moe = MethodType(_grouped_moe, module)
        module._axolotl_grouped_moe = True
        module._axolotl_mlp_impl = mlp_impl
        patched += 1

    return patched
