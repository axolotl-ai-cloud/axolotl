# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Reconstruct a dense per-expert LoRA weight delta from a saved adapter.

Step 1 of merging a trained expert LoRA into an NVFP4-quantized MoE base: turn the
PEFT ``target_parameters`` adapter (``lora_A`` / ``lora_B`` per targeted expert
parameter) into the exact dense delta the ScatterMoE training kernel applies, in the
same ``[E, out, in]`` orientation as the stored fused base expert weight.

Kernel math (confirmed against ``kernels/lora_ops.py`` ``_compute_expert_block_lora``
and ``parallel_linear_lora``): the base weight handed to the kernel is the dequantized
fused expert ``.transpose(2, 1)`` => ``[E, K, N] = [E, in, out]`` (so the STORED base
param is ``[E, out, in]``). For each expert e the kernel adds
``scaling * (X @ A_e^T) @ B_e^T`` to ``X @ W_e``, with ``A_e = lora_A[e*r:(e+1)*r, :]``
of shape ``[r, in]`` and ``B_e = lora_B[:, e*r:(e+1)*r]`` of shape ``[out, r]`` AFTER the
PEFT-rank-major -> scattermoe-expert-major B layout conversion. Hence the effective
per-expert weight delta in the kernel's ``[in, out]`` orientation is
``scaling * (B_e @ A_e)^T``, i.e. in the stored ``[out, in]`` orientation it is
``scaling * (B_e @ A_e)``. So ``base.dequantize()[E, out, in] + delta`` is the merge.

Importable without triton: this module imports only torch and inlines the pure-torch
B-reshape so it runs on a triton-less CPU host (the package ``__init__`` and
``layers.py`` import triton at top).
"""

from __future__ import annotations

import torch

ProjName = str  # "gate_up_proj" | "down_proj"


def _peft_lora_B_to_scattermoe(
    peft_B: torch.Tensor, num_experts: int, rank: int
) -> torch.Tensor:
    """Pure-torch copy of ``layers.peft_lora_B_to_scattermoe`` (no triton import).

    peft stores B rank-major ``[out, r*E]`` (reshape to ``[out, r, E]``); scattermoe
    slices B expert-major as ``[:, e*r:(e+1)*r]``, so permute r<->E then reflatten.
    """
    out = peft_B.shape[0]
    return (
        peft_B.reshape(out, rank, num_experts)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(out, num_experts * rank)
    )


def extract_expert_lora_from_state_dict(
    adapter_sd: dict[str, torch.Tensor],
    num_experts: int,
    scaling: float,
) -> dict[tuple[int, ProjName], tuple[torch.Tensor, torch.Tensor, float]]:
    """Pull per-expert LoRA A/B out of a saved adapter state dict into scattermoe layout.

    Returns ``{(layer_index, proj): (sm_A, sm_B, scaling)}`` where ``sm_A``/``sm_B`` are
    in the SAME layout ``_unwrap_experts_lora`` produces at train time: ``sm_A`` is the
    saved ``lora_A`` unchanged (``[r*E, in]``); ``sm_B`` is the saved ``lora_B``
    (``[out, r*E]``) run through the rank-major -> expert-major conversion. ``rank`` is
    derived as ``lora_A.shape[0] // num_experts``.

    Key-naming assumption: PEFT ``target_parameters`` keys look like
    ``...layers.<idx>....<proj>....lora_A.weight`` / ``lora_B.weight`` (adapter name
    already stripped by ``get_peft_model_state_dict``); we pair A/B by their shared
    prefix and read the layer index and proj ({gate_up_proj, down_proj}) from the path.
    """
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in adapter_sd.items():
        if key.endswith("lora_A.weight"):
            prefix = key[: -len("lora_A.weight")]
            pairs.setdefault(prefix, {})["A"] = tensor
        elif key.endswith("lora_B.weight"):
            prefix = key[: -len("lora_B.weight")]
            pairs.setdefault(prefix, {})["B"] = tensor

    out: dict[tuple[int, ProjName], tuple[torch.Tensor, torch.Tensor, float]] = {}
    for prefix, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        proj = _proj_from_prefix(prefix)
        layer_index = _layer_index_from_prefix(prefix)
        if proj is None or layer_index is None:
            continue

        lora_A = ab["A"]
        lora_B = ab["B"]
        rank = lora_A.shape[0] // num_experts
        sm_A = lora_A
        sm_B = _peft_lora_B_to_scattermoe(lora_B, num_experts, rank)
        out[(layer_index, proj)] = (sm_A, sm_B, scaling)

    return out


def _proj_from_prefix(prefix: str) -> ProjName | None:
    if "gate_up_proj" in prefix:
        return "gate_up_proj"
    if "down_proj" in prefix:
        return "down_proj"
    return None


def _layer_index_from_prefix(prefix: str) -> int | None:
    parts = prefix.split(".")
    for i, part in enumerate(parts):
        if part in ("layers", "h", "blocks") and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt.isdigit():
                return int(nxt)
    return None


def reconstruct_expert_delta(
    sm_A: torch.Tensor,
    sm_B: torch.Tensor,
    scaling: float,
    num_experts: int,
    rank: int,
) -> torch.Tensor:
    """Dense per-expert LoRA delta ``[E, out, in]`` matching the training-kernel math.

    ``sm_A`` ``[r*E, in]``, ``sm_B`` ``[out, r*E]`` (scattermoe expert-major). For expert
    e the kernel adds ``scaling * (X @ A_e^T) @ B_e^T`` with ``A_e = sm_A[e*r:(e+1)*r]``
    ``[r, in]`` and ``B_e = sm_B[:, e*r:(e+1)*r]`` ``[out, r]``, i.e. effective weight
    delta ``scaling * (B_e @ A_e)`` in the stored ``[out, in]`` orientation. Plain torch
    (no triton); orientation matches the stored fused base expert weight so
    ``base.dequantize() + delta`` merges.
    """
    out_features = sm_B.shape[0]
    in_features = sm_A.shape[1]

    A = sm_A.reshape(num_experts, rank, in_features)  # [E, r, in]
    # sm_B is [out, E*r] expert-major: view as [out, E, r] then move E to front.
    B = sm_B.reshape(out_features, num_experts, rank).permute(1, 0, 2)  # [E, out, r]

    delta = torch.matmul(B, A)  # [E, out, r] @ [E, r, in] -> [E, out, in]
    return delta.mul(scaling)


def _nvfp4_requant_cls():
    """torchao NVFP4Tensor + per_tensor_amax_to_scale, imported lazily (no torchao at import)."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    return NVFP4Tensor, per_tensor_amax_to_scale


def _requant_proj_weight(
    weight_2d: torch.Tensor,
    per_tensor_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Requantize one 2D ``[out, in]`` bf16 weight to NVFP4 and let torchao recompute the per-block
    E4M3 scale.

    ``per_tensor_scale`` lets the caller pin the per-tensor scale; when omitted it is recomputed from
    this weight's amax. The loaders read ONE shared weight_scale_2 for the fused gate||up block (gate
    proj's only), so gate and up must be requantized under a shared per-tensor scale (the NVIDIA
    checkpoint invariant) rather than distinct per-proj scales. Pure torch: ``to_nvfp4`` /
    ``nvfp4_quantize`` run on CPU (no CUDA needed; only the NVFP4 matmul path uses ``_scaled_mm``).
    Returns ``(qdata uint8, weight_scale e4m3, weight_scale_2 fp32 scalar)`` ready for the writer.
    """
    NVFP4Tensor, per_tensor_amax_to_scale = _nvfp4_requant_cls()
    w = weight_2d.contiguous()
    if w.dtype not in (torch.bfloat16, torch.float32):
        w = w.to(torch.bfloat16)
    pts = (
        per_tensor_amax_to_scale(w.abs().max())
        if per_tensor_scale is None
        else per_tensor_scale
    )
    t = NVFP4Tensor.to_nvfp4(w, block_size=16, per_tensor_scale=pts)
    return t.qdata, t.scale, pts.to(torch.float32)


def _dequant_fused(base_nvfp4) -> torch.Tensor:
    """Dequantize a fused per-expert NVFP4Tensor to bf16 ``[E, out, in]``.

    torchao's ``fused[e]`` index does NOT slice the ``[E,1,1]`` per_tensor_scale, so dequantize
    each expert from its own scalar scale (matching the per-expert kernel) and stack."""
    NVFP4Tensor, _ = _nvfp4_requant_cls()
    pe = base_nvfp4.per_tensor_scale.reshape(-1)  # [E]
    if pe.numel() != base_nvfp4.qdata.shape[0]:
        raise ValueError(
            f"per_tensor_scale has {pe.numel()} entries but qdata has "
            f"{base_nvfp4.qdata.shape[0]} experts; expected one per-tensor scale per expert"
        )
    block_size = base_nvfp4.block_size
    orig_dtype = base_nvfp4.orig_dtype
    experts = []
    for e in range(base_nvfp4.qdata.shape[0]):
        t_e = NVFP4Tensor(
            base_nvfp4.qdata[e],
            base_nvfp4.scale[e],
            block_size,
            orig_dtype,
            per_tensor_scale=pe[e],
        )
        experts.append(t_e.dequantize(torch.bfloat16))
    return torch.stack(experts, 0)  # [E, out, in]


def _merged_fused(
    base_nvfp4, lora: tuple[torch.Tensor, torch.Tensor, float] | None, num_experts: int
) -> torch.Tensor:
    """Dequantize the fused base to bf16 ``[E, out, in]`` and add the reconstructed LoRA delta."""
    merged = _dequant_fused(base_nvfp4)
    if lora is None:
        return merged
    sm_A, sm_B, scaling = lora
    rank = sm_A.shape[0] // num_experts
    delta = reconstruct_expert_delta(sm_A, sm_B, scaling, num_experts, rank)
    return (merged.float() + delta.to(merged.device).float()).to(torch.bfloat16)


def _gate_up_split_sizes(out_features: int, n_sources: int) -> list[int]:
    """Even split of the fused out (row) axis into the gate_up source projections."""
    if out_features % n_sources != 0:
        raise ValueError(
            f"fused gate_up out_features {out_features} not divisible by "
            f"{n_sources} source projections"
        )
    size = out_features // n_sources
    return [size] * n_sources


def merge_layer_experts(
    base_gate_up_nvfp4,
    base_down_nvfp4,
    gup_lora: tuple[torch.Tensor, torch.Tensor, float] | None,
    down_lora: tuple[torch.Tensor, torch.Tensor, float] | None,
    scheme: dict,
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    """Dequant -> merge LoRA -> requant -> UN-FUSE one MoE layer's experts.

    ``base_gate_up_nvfp4`` is the fused ``[E, 2I, H]`` expert NVFP4Tensor (gate_up = cat of the
    two ``scheme["gate_up"]`` source projs on the out/row axis), ``base_down_nvfp4`` the fused
    ``[E, H, I]`` down NVFP4Tensor. ``gup_lora``/``down_lora`` are ``(sm_A, sm_B, scaling)`` in
    scattermoe layout (or None). Each fused proj is dequantized to bf16, the reconstructed dense
    LoRA delta is added, then gate_up is split on the out axis into its source projs while down is
    kept whole; finally each (expert, source_proj) 2D ``[out, in]`` weight is requantized. gate and
    up of an expert share ONE per-tensor scale (computed from the combined gate||up amax of the
    merged fused block) so the on-disk weight_scale_2 is identical for both, matching the NVIDIA
    checkpoint invariant the loaders depend on (they read only gate's weight_scale_2 and apply it to
    the whole fused gate||up block). down keeps its own per-tensor scale.

    Returns ``{source_proj_name: {expert_index: {"weight", "weight_scale", "weight_scale_2"}}}``
    keyed by the scheme's source proj names (dsv4 w1/w3/w2, gemma4 gate/up/down). Proj names and
    the gate_up split are driven by ``scheme`` (no dsv4-vs-gemma4 hardcoding).
    """
    gate_up_projs = list(scheme["gate_up"])
    down_projs = list(scheme["down"])

    E = base_gate_up_nvfp4.qdata.shape[0]

    gate_up_merged = _merged_fused(base_gate_up_nvfp4, gup_lora, E)  # [E, 2I, H]
    down_merged = _merged_fused(base_down_nvfp4, down_lora, E)  # [E, H, I]

    out: dict[str, dict[int, dict[str, torch.Tensor]]] = {}

    # Un-fuse gate_up on the out (row) axis into the source projs (the inverse of the loader's cat).
    _, per_tensor_amax_to_scale = _nvfp4_requant_cls()
    split_sizes = _gate_up_split_sizes(gate_up_merged.shape[1], len(gate_up_projs))
    gate_up_parts = torch.split(gate_up_merged, split_sizes, dim=1)
    # One shared per-tensor scale per expert from the combined gate||up amax: the loaders read only
    # gate's weight_scale_2 and apply it to the whole fused block, so both projs must use this scale.
    shared_pts = [
        per_tensor_amax_to_scale(gate_up_merged[e].abs().max()) for e in range(E)
    ]
    for proj, part in zip(gate_up_projs, gate_up_parts, strict=True):
        per_expert: dict[int, dict[str, torch.Tensor]] = {}
        for e in range(E):
            qd, sc, pts = _requant_proj_weight(part[e], shared_pts[e])
            per_expert[e] = {
                "weight": qd,
                "weight_scale": sc,
                "weight_scale_2": pts,
            }
        out[proj] = per_expert

    # down has a single source proj kept whole.
    for proj in down_projs:
        per_expert = {}
        for e in range(E):
            qd, sc, pts = _requant_proj_weight(down_merged[e])
            per_expert[e] = {
                "weight": qd,
                "weight_scale": sc,
                "weight_scale_2": pts,
            }
        out[proj] = per_expert

    return out


def checkpoint_keys_for(
    scheme: dict, layer: int, source_proj: str, expert: int
) -> dict[str, str]:
    """Checkpoint key strings for one (layer, source_proj, expert) under ``scheme["base_fmt"]``.

    Inverse of the loader's per-proj naming: returns ``{"weight", "weight_scale",
    "weight_scale_2"}`` keys the writer assigns the requantized tensors to.
    """
    base = scheme["base_fmt"].format(layer=layer, e=expert, proj=source_proj)
    return {
        "weight": f"{base}.weight",
        "weight_scale": f"{base}.weight_scale",
        "weight_scale_2": f"{base}.weight_scale_2",
    }
