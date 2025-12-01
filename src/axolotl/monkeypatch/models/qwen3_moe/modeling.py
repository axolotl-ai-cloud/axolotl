"""Grouped MoE kernels for Qwen3 MoE architectures."""

from __future__ import annotations

from types import MethodType
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from axolotl.monkeypatch.models.utils import clone_expert_parameter
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class Qwen3MoeGroupedExperts(nn.Module):
    """Vectorised grouped expert MLP for Qwen3 MoE blocks."""

    def __init__(self, block: nn.Module, experts: Iterable[nn.Module]):
        super().__init__()
        experts = list(experts)
        if not experts:
            raise ValueError("Expected at least one expert module to convert.")

        self.num_experts = getattr(block, "num_experts", len(experts))
        top_k = getattr(block, "top_k", None)
        if top_k is None:
            top_k = getattr(block, "num_experts_per_tok", None)
        self.top_k = top_k
        hidden_size = getattr(block.gate, "in_features", None)
        if hidden_size is None:
            hidden_size = getattr(experts[0].gate_proj, "in_features", None)
        self.hidden_size = hidden_size
        self.intermediate_size = getattr(experts[0].gate_proj, "out_features", None)

        if len(experts) != self.num_experts:
            raise ValueError(
                f"Config expects {self.num_experts} experts, but received {len(experts)}."
            )

        device = experts[0].gate_proj.weight.device
        dtype = experts[0].gate_proj.weight.dtype

        gate_dims = tuple(getattr(experts[0].gate_proj.weight, "ds_shape", experts[0].gate_proj.weight.shape))
        up_dims = tuple(getattr(experts[0].up_proj.weight, "ds_shape", experts[0].up_proj.weight.shape))
        down_dims = tuple(getattr(experts[0].down_proj.weight, "ds_shape", experts[0].down_proj.weight.shape))

        gate_weights: list[torch.Tensor] = []
        up_weights: list[torch.Tensor] = []
        down_weights: list[torch.Tensor] = []

        gate_bias = experts[0].gate_proj.bias
        up_bias = experts[0].up_proj.bias
        down_bias = experts[0].down_proj.bias

        for idx, expert in enumerate(experts):
            if expert.gate_proj.weight.numel() == 0:
                LOG.warning(
                    "Encountered expert %s with empty gate weight during Qwen3 grouped conversion; shape: %s, dtype: %s",
                    idx,
                    tuple(expert.gate_proj.weight.shape),
                    expert.gate_proj.weight.dtype,
                )
            gate_weights.append(clone_expert_parameter(expert.gate_proj.weight))
            up_weights.append(clone_expert_parameter(expert.up_proj.weight))
            down_weights.append(clone_expert_parameter(expert.down_proj.weight))

        gate_stack = torch.stack(gate_weights, dim=0).to(device=device, dtype=dtype)
        up_stack = torch.stack(up_weights, dim=0).to(device=device, dtype=dtype)
        down_stack = torch.stack(down_weights, dim=0).to(device=device, dtype=dtype)

        if gate_stack.numel() == 0 and all(dim > 0 for dim in gate_dims):
            gate_stack = torch.zeros((self.num_experts,) + gate_dims, device=device, dtype=dtype)
        if up_stack.numel() == 0 and all(dim > 0 for dim in up_dims):
            up_stack = torch.zeros((self.num_experts,) + up_dims, device=device, dtype=dtype)
        if down_stack.numel() == 0 and all(dim > 0 for dim in down_dims):
            down_stack = torch.zeros((self.num_experts,) + down_dims, device=device, dtype=dtype)

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
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, hidden = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden)
        topk = selected_experts.shape[-1]

        routing_weights = routing_weights.to(hidden_states.dtype)

        token_indices = torch.arange(hidden_flat.size(0), device=hidden_flat.device, dtype=torch.long)
        token_indices = token_indices.repeat_interleave(topk)
        expert_ids = selected_experts.reshape(-1)
        routing_flat = routing_weights.reshape(-1)

        active_counts = torch.bincount(expert_ids, minlength=self.num_experts)
        active_expert_ids = torch.nonzero(active_counts, as_tuple=False).flatten()
        if active_expert_ids.numel() == 0:
            return hidden_states.new_zeros((bsz, seq_len, hidden))

        counts = active_counts.index_select(0, active_expert_ids)
        offsets = torch.cumsum(counts.to(dtype=torch.int32), dim=0, dtype=torch.int32).contiguous()

        _, sort_perm = torch.sort(expert_ids)
        sorted_tokens = token_indices.index_select(0, sort_perm)
        expert_inputs = hidden_flat.index_select(0, sorted_tokens).to(self.gate_weight.dtype)
        repeat_sizes = counts.to(torch.long)

        gate_active = self.gate_weight.index_select(0, active_expert_ids)
        up_active = self.up_weight.index_select(0, active_expert_ids)
        down_active = self.down_weight.index_select(0, active_expert_ids)

        gate_bias_active = (
            self.gate_bias.index_select(0, active_expert_ids) if self.gate_bias is not None else None
        )
        up_bias_active = (
            self.up_bias.index_select(0, active_expert_ids) if self.up_bias is not None else None
        )
        down_bias_active = (
            self.down_bias.index_select(0, active_expert_ids) if self.down_bias is not None else None
        )

        def _run_linear(inputs: torch.Tensor, weight_bank: torch.Tensor, bias_bank: torch.Tensor | None) -> torch.Tensor:
            weight_t = weight_bank.transpose(-2, -1).contiguous()
            if inputs.is_cuda and hasattr(torch, "_grouped_mm"):
                try:
                    out = torch._grouped_mm(inputs, weight_t, offs=offsets)
                    if bias_bank is not None:
                        bias_active = bias_bank
                        out = out + torch.repeat_interleave(bias_active, repeat_sizes, dim=0).to(out.dtype)
                    return out
                except (RuntimeError, NotImplementedError) as exc:
                    if isinstance(exc, RuntimeError) and "grouped gemm is not supported" not in str(exc).lower():
                        raise

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
                result = torch.matmul(chunk, weight.transpose(0, 1))
                if bias is not None:
                    result = result + bias
                outputs.append(result)
                start_idx = end_idx
            if outputs:
                return torch.cat(outputs, dim=0)
            return inputs.new_empty((0, weight_bank.size(1)))

        gate_out = _run_linear(expert_inputs, gate_active, gate_bias_active)
        up_out = _run_linear(expert_inputs, up_active, up_bias_active)

        activated = F.silu(gate_out).to(up_out.dtype) * up_out
        down_out = _run_linear(activated, down_active, down_bias_active)

        inverse_perm = torch.empty_like(sort_perm)
        inverse_perm.index_copy_(
            0,
            sort_perm,
            torch.arange(sort_perm.numel(), device=sort_perm.device, dtype=sort_perm.dtype),
        )
        unsorted = down_out.index_select(0, inverse_perm)
        weighted = unsorted.to(hidden_states.dtype) * routing_flat.unsqueeze(-1)

        output = hidden_states.new_zeros((hidden_flat.size(0), hidden))
        scatter_index = token_indices.unsqueeze(-1).expand_as(weighted)
        output.scatter_add_(0, scatter_index, weighted)

        return output.view(bsz, seq_len, hidden)


def _grouped_forward(self, hidden_states: torch.Tensor):
    batch_size, seq_len, hidden = hidden_states.shape
    hidden_flat = hidden_states.view(-1, hidden)
    router_logits = self.gate(hidden_flat)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if getattr(self, "norm_topk_prob", False):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    expert_out = self.experts(hidden_states, routing_weights, selected_experts)
    return expert_out, router_logits


def patch_qwen3_moe_grouped_experts(model: nn.Module, mlp_impl: str = "grouped") -> int:
    """Convert Qwen3MoeSparseMoeBlock modules to grouped expert implementations."""
    if mlp_impl != "grouped":
        raise ValueError(f"Unsupported mlp_impl={mlp_impl} for Qwen3 grouped experts.")

    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    except ImportError:
        LOG.warning("transformers.models.qwen3_moe not available; skipping grouped patch.")
        return 0

    patched = 0
    for module in model.modules():
        if not isinstance(module, Qwen3MoeSparseMoeBlock):
            continue
        if getattr(module, "_axolotl_grouped_moe", False):
            patched += 1
            continue

        try:
            grouped = Qwen3MoeGroupedExperts(module, module.experts)
        except Exception as exc:
            LOG.warning("Failed to convert Qwen3 MoE block to grouped experts: %s", exc)
            continue

        module.experts = grouped
        module.forward = MethodType(_grouped_forward, module)
        module._axolotl_grouped_moe = True
        module._axolotl_mlp_impl = mlp_impl
        patched += 1
    return patched
