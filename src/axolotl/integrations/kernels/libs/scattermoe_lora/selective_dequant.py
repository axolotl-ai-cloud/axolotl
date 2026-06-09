"""
Selective Expert Dequantization
===============================

Instead of dequantizing all E expert weight matrices at once (which creates
a ~1 GB transient buffer for 256 experts), only dequantize the experts that
are actually routed to by the current batch's top-k selection.

For Qwen3.5-35B-A3B (E=256, top_k=8, hidden=2048, intermediate=512):
  - Full dequant: [256, 2048, 1024] = 1,074 MB per projection
  - Selective (8 active): [8, 2048, 1024] = 33.5 MB per projection
  - Savings: ~97% memory reduction per layer

This module provides format-agnostic selective weight extraction:
  - BnB 4-bit (nf4/fp4): slice quantized data + absmax per expert
  - MXFP4 (torchao MXTensor with elem_dtype=float4_e2m1fn_x2): slice
    qdata + E8M0 scale per expert and dequantize via torchao
  - bf16/fp32: direct indexing (no dequant needed)
  - FP8: slice + cast

The ScatterMoE kernel itself doesn't change — we remap expert indices
from global (0..E-1) to compact (0..num_active-1) and pass the smaller
weight tensor.
"""

import torch
import torch.nn as nn

from .mx_weights import (
    _construct_mxtensor_subset,
    _mx_qdata,
    _mx_scale,
    _torchao_mxtensor_cls,
    _torchao_nvfp4tensor_cls,
)


def is_mxfp4_param(param) -> bool:
    """True iff ``param`` is a torchao MXTensor with MXFP4 element dtype."""
    MXTensor = _torchao_mxtensor_cls()
    if MXTensor is None or not isinstance(param, MXTensor):
        return False
    return param.elem_dtype == torch.float4_e2m1fn_x2


def is_nvfp4_param(param) -> bool:
    """True iff ``param`` is a torchao NVFP4Tensor (FP4 E2M1, block-16 E4M3 scales)."""
    NVFP4Tensor = _torchao_nvfp4tensor_cls()
    return NVFP4Tensor is not None and isinstance(param, NVFP4Tensor)


def get_active_experts(sorted_expert_idxs: torch.Tensor, E: int) -> torch.Tensor:
    """Get sorted unique expert indices from the routing output.

    Args:
        sorted_expert_idxs: Expert assignments sorted by expert id [T*k]
        E: Total number of experts

    Returns:
        active: Sorted unique expert indices [num_active]
    """
    return torch.unique(sorted_expert_idxs)


def remap_expert_indices(
    sorted_expert_idxs: torch.Tensor,
    expert_offsets: torch.Tensor,
    active_experts: torch.Tensor,
    E: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap global expert indices to compact indices.

    Maps expert ids from [0..E-1] to [0..num_active-1], preserving the
    sort order. Also compacts expert_offsets to only active experts.

    Args:
        sorted_expert_idxs: [T*k] expert ids in sorted order
        expert_offsets: [E] cumulative token counts (original)
        active_experts: [num_active] sorted unique expert ids
        E: Total number of experts

    Returns:
        remapped_idxs: [T*k] expert ids in [0..num_active-1]
        compact_offsets: [num_active] cumulative token counts
    """
    # Build remap table: global_id -> compact_id
    remap = torch.empty(E, dtype=torch.long, device=sorted_expert_idxs.device)
    remap[active_experts] = torch.arange(
        len(active_experts), device=sorted_expert_idxs.device
    )

    remapped_idxs = remap[sorted_expert_idxs]

    # Compact the expert_offsets: only keep active experts' cumulative counts
    compact_offsets = expert_offsets[active_experts]

    return remapped_idxs, compact_offsets


def _selective_dequant_bnb4(
    raw_param: torch.Tensor,
    quant_state,
    active_experts: torch.Tensor,
    expert_shape: tuple[int, int],
) -> torch.Tensor:
    """Dequantize only selected experts from BnB 4-bit packed data.

    The raw parameter is a flattened 4-bit packed tensor. Each expert's
    data is contiguous (stored in expert-major order), so we can gather
    the packed data and absmax blocks for active experts, then dequantize
    as one contiguous block.

    Args:
        raw_param: Flattened uint8 tensor of packed 4-bit weights
        quant_state: BnB QuantState with absmax, blocksize, code, etc.
        active_experts: [num_active] expert indices to dequantize
        expert_shape: (dim1, dim2) shape per expert (e.g. (1024, 2048))

    Returns:
        Dequantized weights [num_active, dim1, dim2] in original dtype
    """
    import bitsandbytes.functional as F  # noqa: N812
    from bitsandbytes.functional import QuantState

    expert_numel = expert_shape[0] * expert_shape[1]
    packed_per_expert = expert_numel // 2  # 4-bit = 2 values per byte
    blocks_per_expert = expert_numel // quant_state.blocksize
    num_active = len(active_experts)

    if blocks_per_expert == 0:
        # Expert is smaller than one quantization block — blocks span across
        # expert boundaries, so per-expert slicing isn't possible.
        # Fallback: full dequantize + index.
        full = F.dequantize_4bit(raw_param, quant_state)
        E_total = full.numel() // expert_numel
        return full.reshape(E_total, *expert_shape)[active_experts]

    # Use fused Triton kernel for NF4 (handles selective gather + dequant in one pass)
    if quant_state.quant_type == "nf4" and raw_param.dtype == torch.uint8:
        from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant_kernel import (
            selective_dequant_nf4_triton,
        )

        # Handle nested (double) quantization: dequantize absmax first
        # BnB uses dequantize_blockwise (not _4bit) for nested absmax + offset
        if quant_state.nested:
            absmax = F.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax += quant_state.offset
            if absmax.dtype != torch.float32:
                absmax = absmax.float()
        else:
            absmax = quant_state.absmax

        return selective_dequant_nf4_triton(
            packed_data=raw_param,
            absmax=absmax,
            active_experts=active_experts,
            expert_shape=expert_shape,
            blocksize=quant_state.blocksize,
            dtype=quant_state.dtype,
            codebook=quant_state.code,
        )

    # Fallback: gather + BnB dequant (for fp4 or non-uint8 packed formats)
    raw_flat = raw_param.reshape(-1)

    offsets_qt = (
        active_experts.long()[:, None] * packed_per_expert
        + torch.arange(packed_per_expert, device=raw_param.device)[None, :]
    ).reshape(-1)
    qt_gathered = raw_flat[offsets_qt]

    offsets_abs = (
        active_experts.long()[:, None] * blocks_per_expert
        + torch.arange(blocks_per_expert, device=raw_param.device)[None, :]
    ).reshape(-1)

    if quant_state.nested:
        full_absmax = F.dequantize_blockwise(quant_state.absmax, quant_state.state2)
        full_absmax += quant_state.offset
        if full_absmax.dtype != torch.float32:
            full_absmax = full_absmax.float()
        absmax_gathered = full_absmax[offsets_abs]
    else:
        absmax_gathered = quant_state.absmax[offsets_abs]

    qt_gathered = qt_gathered.unsqueeze(1) if qt_gathered.dim() == 1 else qt_gathered

    gathered_qs = QuantState(
        absmax=absmax_gathered,
        shape=torch.Size([num_active * expert_numel]),
        blocksize=quant_state.blocksize,
        quant_type=quant_state.quant_type,
        code=quant_state.code,
        dtype=quant_state.dtype,
    )

    deq = F.dequantize_4bit(qt_gathered, gathered_qs)
    return deq.reshape(num_active, *expert_shape)


def _selective_dequant_mxfp4(
    mx_param,
    active_experts: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Selectively dequantize active experts from a torchao MXFP4 ``MXTensor``.

    Layout assumption: the MXTensor's last axis is the OCP MX block axis.
    For ScatterMoE experts this matches the natural storage where
    ``experts.gate_up_proj``/``down_proj`` is ``[E, dim1, dim2]`` and
    ``dim2`` is the contraction axis post ``.transpose(2, 1)`` performed by
    the caller. Indexing ``[active_experts]`` on the qdata and scale yields
    a compact MX tensor that we dequantize via torchao.

    Args:
        mx_param: ``torchao.prototype.mx_formats.mx_tensor.MXTensor`` of
            logical shape ``[E, dim1, dim2]`` with ``elem_dtype=float4_e2m1fn_x2``.
        active_experts: ``[num_active]`` sorted unique expert indices.
        out_dtype: dtype of the dequantized buffer (default ``bfloat16``).

    Returns:
        Dequantized bf16/fp16 tensor of shape ``[num_active, dim1, dim2]``.
    """
    if _torchao_mxtensor_cls() is None:
        raise ImportError(
            "MXFP4 expert dequantization requires torchao>=0.7 "
            "(install with `pip install torchao`)."
        )

    sub_qdata = _mx_qdata(mx_param)[active_experts].contiguous()
    sub_scale = _mx_scale(mx_param)[active_experts].contiguous()

    sub_mx = _construct_mxtensor_subset(mx_param, sub_qdata, sub_scale)
    return sub_mx.dequantize(out_dtype)


def _selective_index_dense(
    param: torch.Tensor,
    active_experts: torch.Tensor,
) -> torch.Tensor:
    """Select experts from a dense (bf16/fp32) weight tensor.

    Simple indexing — no dequantization needed.
    """
    return param[active_experts]


def selective_expert_weights(
    experts_module: nn.Module,
    param_name: str,
    active_experts: torch.Tensor,
) -> torch.Tensor:
    """Extract and dequantize only the active experts' weights.

    Format-agnostic: dispatches based on whether the parameter is
    BnB 4-bit quantized (via parametrize), FP8, or dense bf16/fp32.

    Args:
        experts_module: The base experts module (e.g. Qwen3_5MoeExperts)
        param_name: "gate_up_proj" or "down_proj"
        active_experts: [num_active] sorted unique expert indices

    Returns:
        Compact weight tensor [num_active, dim1, dim2] ready for ScatterMoE
    """
    # Check if the parameter is BnB-quantized via parametrize
    if (
        hasattr(experts_module, "parametrizations")
        and param_name in experts_module.parametrizations
    ):
        param_list = experts_module.parametrizations[param_name]
        parametrization = param_list[0]

        # BnB 4-bit parametrization
        if hasattr(parametrization, "quant_state"):
            # The raw quantized data is on the ParametrizationList, not the
            # individual Bnb4bitParametrization module
            raw_param = param_list.original
            qs = parametrization.quant_state
            # qs.shape is the original tensor shape before flattening.
            # For MoE experts it's [E, d1, d2] (3D) or [total_elements] (1D).
            orig_shape = qs.shape
            if isinstance(orig_shape, torch.Size) and len(orig_shape) == 3:
                expert_shape = (orig_shape[1], orig_shape[2])
            elif isinstance(orig_shape, torch.Size) and len(orig_shape) == 1:
                # Flattened — need to infer from module attributes
                E_total = getattr(experts_module, "num_experts", None)
                if E_total is None:
                    E_total = int(active_experts.max().item()) + 1
                expert_numel = orig_shape[0] // E_total
                d2 = getattr(experts_module, "hidden_dim", None) or getattr(
                    experts_module, "intermediate_dim", None
                )
                if d2 and expert_numel % d2 == 0:
                    expert_shape = (expert_numel // d2, d2)
                else:
                    full = getattr(experts_module, param_name)
                    return full[active_experts]
            else:
                full = getattr(experts_module, param_name)
                return full[active_experts]

            return _selective_dequant_bnb4(raw_param, qs, active_experts, expert_shape)

    # Pull the parameter out before format dispatch — used by every branch below.
    param = getattr(experts_module, param_name)

    # MXFP4 (torchao MXTensor) — dequantize the subset, return [num_active, d1, d2]
    if is_mxfp4_param(param):
        return _selective_dequant_mxfp4(param, active_experts)

    # Dense parameter (bf16/fp32) — direct indexing
    if param.dim() == 3:
        return param[active_experts]

    # Fallback: full access
    return param


def shared_dequant_across_shards(
    experts_module: nn.Module,
    param_name: str,
    sei_per_shard: list[torch.Tensor],
    E: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Dequantize the union of active experts across N shards exactly once.

    The orthogonal Strategy A path calls :func:`selective_expert_weights`
    once per shard, which re-dequantizes the active experts redundantly
    when the active-expert sets overlap. For seq-dim sharding with a
    softmax-routed MoE, that overlap is the common case.

    This helper hoists the dequant: it computes the union of active
    experts across all shards, calls :func:`selective_expert_weights`
    once on the union, and returns per-shard index tables that map each
    shard's local active experts into rows of the union buffer.

    Parameters
    ----------
    experts_module:
        The base experts module (e.g. ``OlmoeExperts``). Same object the
        per-shard path would pass to :func:`selective_expert_weights`.
    param_name:
        ``"gate_up_proj"`` or ``"down_proj"``.
    sei_per_shard:
        List of ``sorted_expert_idxs`` tensors, one per shard.
    E:
        Total number of experts.

    Returns
    -------
    union_active:
        ``[U]`` sorted unique expert ids across all shards.
    union_buffer:
        Dequantized weights for ``union_active``,
        ``[U, dim1, dim2]`` in the param's natural storage dtype
        (typically bf16). Same buffer each shard's call would have built
        had it dequantized only its own active set, just shared.
    shard_into_union:
        List of length ``len(sei_per_shard)``. Entry ``i`` is a 1-D
        ``long`` tensor that indexes ``union_buffer`` along dim 0 to
        produce the same ``[num_active_i, dim1, dim2]`` slice the
        per-shard path would have produced. Callers feed this through
        ``union_buffer.index_select(0, shard_into_union[i])`` (or
        equivalent advanced indexing) before handing the slice to
        ``parallel_linear_lora``.

    Bitwise contract: composing ``union_buffer.index_select(0,
    shard_into_union[i])`` is byte-identical to
    ``selective_expert_weights(experts_module, param_name,
    get_active_experts(sei_per_shard[i], E))`` because both paths slice
    the same dequantized MX subset by the same expert ids. The
    ``test_shared_dequant_helper.py`` parity test asserts this.
    """
    if not sei_per_shard:
        raise ValueError("sei_per_shard must contain at least one tensor")

    device = sei_per_shard[0].device
    per_shard_active = [get_active_experts(sei, E) for sei in sei_per_shard]
    union_active = torch.unique(torch.cat(per_shard_active))

    union_buffer = selective_expert_weights(experts_module, param_name, union_active)

    # Build the global-id → union-row remap once, then gather per shard.
    # ``union_active`` is sorted and unique by construction, so the inverse
    # lookup is dense over ``E``.
    union_remap = torch.empty(E, dtype=torch.long, device=device)
    union_remap[union_active] = torch.arange(
        len(union_active), device=device, dtype=torch.long
    )
    shard_into_union = [union_remap[active] for active in per_shard_active]

    return union_active, union_buffer, shard_into_union


def selective_lora_weights(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    active_experts: torch.Tensor,
    E: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select LoRA A and B weights for only the active experts.

    LoRA layout (scattermoe format):
      A: [r*E, K] — expert e occupies rows [e*r : (e+1)*r]
      B: [N, r*E] — expert e occupies cols [e*r : (e+1)*r]

    Returns compact:
      A: [r*num_active, K]
      B: [N, r*num_active]
    """
    R = lora_A.size(0) // E

    # Vectorized gather: active_experts[:, None] * R + arange(R)[None, :]
    row_idx = (
        active_experts.long()[:, None] * R
        + torch.arange(R, device=lora_A.device)[None, :]
    ).reshape(-1)

    compact_A = lora_A[row_idx]  # [r*num_active, K]
    compact_B = lora_B[:, row_idx]  # [N, r*num_active]

    return compact_A, compact_B
