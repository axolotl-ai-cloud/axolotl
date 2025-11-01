"""Grouped MoE kernels for Bailing/Ring (BailingMoeV2) architectures."""

from __future__ import annotations

from types import MethodType
from typing import Iterable, Optional
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


_MEGABLOCKS_BACKEND = None
_MEGABLOCKS_IMPORT_ERROR: Optional[Exception] = None


def _load_megablocks_backend() -> Optional[object]:
    global _MEGABLOCKS_BACKEND, _MEGABLOCKS_IMPORT_ERROR
    if _MEGABLOCKS_BACKEND is not None or _MEGABLOCKS_IMPORT_ERROR is not None:
        return _MEGABLOCKS_BACKEND

    try:
        from megablocks.grouped_gemm import backend as mb_backend  # type: ignore

        _MEGABLOCKS_BACKEND = mb_backend
        backend_path = getattr(mb_backend, "__file__", None)
        if backend_path:
            LOG.info("Loaded MegaBlocks grouped GEMM backend from %s", backend_path)
        return _MEGABLOCKS_BACKEND
    except ImportError:
        pass

    root = Path(__file__).resolve().parents[6]
    candidate_repo = root / "better-moe-training" / "megablocks-hip"
    search_roots: list[Path] = []

    env_path = os.environ.get("MEGABLOCKS_HIP_PATH")
    if env_path:
        search_roots.append(Path(env_path))
    if candidate_repo.exists():
        try:
            from kernels.utils import build_variant  # type: ignore

            variant = build_variant()
        except Exception:
            variant = None

        if variant:
            search_roots.append(candidate_repo / "build" / variant)
        search_roots.append(candidate_repo / "torch-ext")

    for path in search_roots:
        if path.exists():
            if str(path) not in sys.path:
                sys.path.append(str(path))

    try:
        from megablocks.grouped_gemm import backend as mb_backend  # type: ignore

        _MEGABLOCKS_BACKEND = mb_backend
        backend_path = getattr(mb_backend, "__file__", None)
        if backend_path:
            LOG.info("Loaded MegaBlocks grouped GEMM backend from %s", backend_path)
        return _MEGABLOCKS_BACKEND
    except Exception as exc:  # pragma: no cover - we record and fallback
        _MEGABLOCKS_IMPORT_ERROR = exc
        LOG.warning("Failed to load MegaBlocks grouped GEMM backend: %s", exc)
        return None


class BailingMoeV2GroupedExperts(nn.Module):
    """Grouped expert MLP that replaces per-expert Python loops."""

    def __init__(self, config, experts: Iterable[nn.Module], backend_impl: str):
        super().__init__()
        experts = list(experts)
        if not experts:
            raise ValueError("Expected at least one expert module to convert.")

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.act_fn = ACT2FN[config.hidden_act]
        self.backend_impl = backend_impl
        self._warned_grouped_mm_fallback = False
        self._warned_megablocks_missing = False

        device = experts[0].gate_proj.weight.device
        dtype = experts[0].gate_proj.weight.dtype

        gate_weights = []
        up_weights = []
        down_weights = []
        gate_bias = experts[0].gate_proj.bias
        up_bias = experts[0].up_proj.bias
        down_bias = experts[0].down_proj.bias

        if len(experts) != self.num_experts:
            raise ValueError(
                f"Config expects {self.num_experts} experts, but received {len(experts)}."
            )

        for expert in experts:
            gate_weights.append(expert.gate_proj.weight.detach().clone())
            up_weights.append(expert.up_proj.weight.detach().clone())
            down_weights.append(expert.down_proj.weight.detach().clone())

        self.gate_weight = nn.Parameter(torch.stack(gate_weights, dim=0).to(device=device, dtype=dtype))
        self.up_weight = nn.Parameter(torch.stack(up_weights, dim=0).to(device=device, dtype=dtype))
        self.down_weight = nn.Parameter(torch.stack(down_weights, dim=0).to(device=device, dtype=dtype))

        if gate_bias is not None:
            stacked = torch.stack(
                [expert.gate_proj.bias.detach().clone() for expert in experts], dim=0
            ).to(device=device, dtype=gate_bias.dtype)
            self.gate_bias = nn.Parameter(stacked)
        else:
            self.register_parameter("gate_bias", None)

        if up_bias is not None:
            stacked = torch.stack(
                [expert.up_proj.bias.detach().clone() for expert in experts], dim=0
            ).to(device=device, dtype=up_bias.dtype)
            self.up_bias = nn.Parameter(stacked)
        else:
            self.register_parameter("up_bias", None)

        if down_bias is not None:
            stacked = torch.stack(
                [expert.down_proj.bias.detach().clone() for expert in experts], dim=0
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
        """Compute grouped expert outputs.

        Args:
            hidden_states: (batch, seq, hidden)
            topk_idx: (batch*seq, top_k)
            topk_weight: (batch*seq, top_k)
        """
        bsz, seq_len, hidden = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden)
        num_tokens = hidden_flat.size(0)
        if num_tokens == 0:
            return hidden_states

        topk = topk_idx.shape[-1]
        dispatch_indices = torch.arange(num_tokens, device=hidden_flat.device, dtype=torch.long)
        dispatch_indices = dispatch_indices.repeat_interleave(topk)

        flat_expert = topk_idx.reshape(-1)
        flat_weight = topk_weight.reshape(-1).to(hidden_states.dtype)

        active_counts = torch.bincount(flat_expert, minlength=self.num_experts)
        active_expert_indices = torch.nonzero(active_counts, as_tuple=False).flatten()
        if active_expert_indices.numel() == 0:
            return hidden_states.new_zeros((bsz, seq_len, hidden))

        active_counts = active_counts.index_select(0, active_expert_indices)
        offsets = torch.cumsum(
            active_counts.to(dtype=torch.int32), dim=0, dtype=torch.int32
        ).contiguous()

        _, sort_perm = torch.sort(flat_expert)
        sorted_tokens = dispatch_indices.index_select(0, sort_perm)
        expert_inputs = hidden_flat.index_select(0, sorted_tokens).to(self.gate_weight.dtype)

        repeat_sizes = active_counts.to(torch.long)

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

        def _run_linear(inputs: torch.Tensor, weight_bank: torch.Tensor, bias_bank: torch.Tensor | None) -> torch.Tensor:
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
                elif not inputs.is_cuda or inputs.dtype != torch.bfloat16 or weight_t.dtype != torch.bfloat16:
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
                            bias_bank, repeat_sizes, dim=0
                        ).to(outputs.dtype)
                    return outputs

            try:
                outputs = torch._grouped_mm(inputs, weight_t, offs=offsets)
                if bias_bank is not None:
                    expanded_bias = torch.repeat_interleave(bias_bank, repeat_sizes, dim=0).to(outputs.dtype)
                    outputs = outputs + expanded_bias
                return outputs
            except RuntimeError as exc:
                if "grouped gemm is not supported" in str(exc).lower():
                    mb_backend = _load_megablocks_backend()
                    if mb_backend is not None and inputs.is_cuda and inputs.dtype == torch.bfloat16 and weight_t.dtype == torch.bfloat16:
                        outputs = mb_backend.gmm(
                            inputs,
                            weight_t,
                            batch_sizes_cpu,
                            trans_b=False,
                        )
                        if bias_bank is not None:
                            outputs = outputs + torch.repeat_interleave(
                                bias_bank, repeat_sizes, dim=0
                            ).to(outputs.dtype)
                        return outputs
                    if not self._warned_grouped_mm_fallback:
                        LOG.warning(
                            "torch._grouped_mm unavailable on this backend (%s); falling back to per-expert matmuls.",
                            exc,
                        )
                        self._warned_grouped_mm_fallback = True
                else:
                    raise

                outputs_list: list[torch.Tensor] = []
                start_idx = 0
                bias_iter = (
                    bias_bank if bias_bank is not None else [None] * weight_bank.size(0)
                )
                for count, weight, bias in zip(repeat_sizes.tolist(), weight_bank, bias_iter):
                    if count == 0:
                        continue
                    end_idx = start_idx + count
                    chunk = inputs[start_idx:end_idx]
                    result = torch.matmul(chunk, weight.transpose(0, 1))
                    if bias is not None:
                        result = result + bias
                    outputs_list.append(result)
                    start_idx = end_idx
                if outputs_list:
                    return torch.cat(outputs_list, dim=0)
                return inputs.new_empty((0, weight_bank.size(1)))

        gate_out = _run_linear(expert_inputs, gate_active, gate_bias_active)
        up_out = _run_linear(expert_inputs, up_active, up_bias_active)

        activated = self.act_fn(gate_out).to(up_out.dtype)
        activated = activated * up_out

        combined = _run_linear(activated, down_active, down_bias_active)

        inverse_perm = torch.empty_like(sort_perm)
        inverse_perm.index_copy_(
            0,
            sort_perm,
            torch.arange(sort_perm.size(0), device=sort_perm.device, dtype=sort_perm.dtype),
        )
        unsorted_outputs = combined.index_select(0, inverse_perm)
        weighted = unsorted_outputs.to(hidden_states.dtype) * flat_weight.unsqueeze(-1)

        output_flat = hidden_states.new_zeros((num_tokens, hidden))
        scatter_index = dispatch_indices.unsqueeze(-1).expand_as(weighted)
        output_flat.scatter_add_(0, scatter_index, weighted)

        return output_flat.view(bsz, seq_len, hidden)


def _grouped_forward(self, hidden_states: torch.Tensor):
    identity = hidden_states
    bsz, seq_len, _ = hidden_states.shape
    topk_idx, topk_weight, router_logits = self.gate(hidden_states)
    expert_out = self.experts(hidden_states, topk_idx, topk_weight)
    if getattr(self, "shared_experts", None) is not None:
        expert_out = expert_out + self.shared_experts(identity)

    return expert_out, (
        router_logits.view(bsz, seq_len, -1),
        topk_idx.view(bsz, seq_len, -1),
    )


def _grouped_moe_infer(self, hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weight: torch.Tensor):
    seq_len, hidden = hidden_states.shape
    expert_out = self.experts(
        hidden_states.view(1, seq_len, hidden),
        topk_idx,
        topk_weight,
    )
    return expert_out.view(seq_len, hidden)


def patch_model_with_grouped_experts(model: nn.Module, mlp_impl: str = "grouped") -> int:
    """Convert BailingMoeV2SparseMoeBlock modules to grouped expert implementations.

    Returns:
        Number of blocks patched.
    """

    if mlp_impl not in {"grouped", "megablocks"}:
        raise ValueError(f"Unsupported mlp_impl={mlp_impl} for grouped experts patch.")

    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError(
            "torch._grouped_mm is required for grouped MoE kernels but is unavailable in this torch build."
        )

    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "BailingMoeV2SparseMoeBlock":
            continue
        if getattr(module, "_axolotl_grouped_moe", False):
            patched += 1
            continue

        old_experts = module.experts
        try:
            grouped = BailingMoeV2GroupedExperts(module.config, old_experts, backend_impl=mlp_impl)
        except Exception as exc:  # pragma: no cover - safety net for unexpected configs
            LOG.warning("Failed to convert Bailing MoE block to grouped experts: %s", exc)
            continue

        module.experts = grouped
        module.forward = MethodType(_grouped_forward, module)
        module.moe_infer = MethodType(_grouped_moe_infer, module)
        module._axolotl_grouped_moe = True
        module._axolotl_mlp_impl = mlp_impl
        patched += 1
    return patched
