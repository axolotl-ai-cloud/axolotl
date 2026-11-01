"""Requantize block-fp8 (FineGrainedFP8) MoE experts to NVFP4 for the scattermoe grouped path.

Model-agnostic: any MoE checkpoint whose routed experts are genuine 128x128 block-fp8 (weight
``float8_e4m3fn`` + ``*_scale_inv``, the transformers ``FP8Experts`` layout) and which runs with
``moe_grouped_mode: nvfp4`` + ``use_scattermoe``. Dequantizing block-fp8 to bf16 would double the
(dominant) expert memory, so at load we requantize each fused-3D expert tensor fp8 -> bf16 ->
two-level NVFP4 (per-tensor fp32 + block-16 e4m3), producing a fused ``NVFP4Tensor`` the grouped
kernel consumes directly (Marlin sm120 / DeepGEMM sm90,100).

Native NVFP4-modelopt checkpoints load as-is (this is a no-op on already-NVFP4 params). Under FSDP2
``cpu_ram_efficient_loading`` the ``rank0_only`` path keeps non-rank-0 on a meta ``NVFP4Tensor``
placeholder — the scale rides *inside* the tensor subclass, so FSDP2's own nvfp4-aware rank-0
broadcast fills it (no separate scale broadcast, unlike mxfp8's plain buffer).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _is_nonrank0_process() -> bool:
    """True on every process except local-rank-0 of the node (keyed off torchrun's LOCAL_RANK)."""
    import os

    return int(os.environ.get("LOCAL_RANK", "-1")) > 0


def _to_projlist(w3d: torch.Tensor):
    """Per-expert (qdata, scale, per_tensor_scale) for a fused-3D bf16 weight ``[E, N, K]``."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    qd, sc, pts = [], [], []
    for e in range(w3d.shape[0]):
        we = w3d[e]
        # two-level per-tensor scale: amax / (F4_E2M1_MAX * F8E4M3_MAX) = amax / (6 * 448)
        p = (we.abs().max() / (6.0 * 448.0)).reshape(1).float().clamp(min=1e-12)
        nv = NVFP4Tensor.to_nvfp4(
            we[None], per_tensor_scale=p, is_swizzled_scales=False
        )
        qd.append(nv.qdata[0])
        sc.append(nv.scale[0])
        pts.append(nv.per_tensor_scale.reshape(()))
    return {"qd": qd, "sc": sc, "pts": pts}


def _dequant_fp8_3d(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Batched block-fp8 dequant of a fused-3D expert weight ``[E, N, K]`` -> bf16.

    Real dims are 128-multiples, so the block-scale multiply vectorizes over all experts at once
    (one op vs a per-expert Python loop). Meta-safe (the vectorized ops propagate meta shapes).
    """
    E, N, K = w.shape
    sr, sc = scale_inv.shape[-2], scale_inv.shape[-1]
    if sr and sc and N % sr == 0 and K % sc == 0:
        bn, bk = N // sr, K // sc
        wf = w.to(torch.float32).reshape(E, sr, bn, sc, bk)
        s = scale_inv.to(torch.float32)[:, :, None, :, None]
        return (wf * s).to(torch.bfloat16).reshape(E, N, K)
    from axolotl.kernels.quantize import dequantize_fp8

    return torch.stack(
        [dequantize_fp8(w[e], scale_inv[e], torch.bfloat16) for e in range(E)]
    )


def _fused_nvfp4(w3d_bf16: torch.Tensor):
    from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
        fuse_nvfp4_experts,
    )

    return fuse_nvfp4_experts([_to_projlist(w3d_bf16)])


def _is_block_fp8_3d(t) -> bool:
    return (
        isinstance(t, torch.Tensor) and t.dtype == torch.float8_e4m3fn and t.ndim == 3
    )


def _drop(mod: nn.Module, attr: str) -> None:
    if hasattr(mod, attr):
        try:
            delattr(mod, attr)
        except Exception:  # registered buffer/param
            setattr(mod, attr, None)


def requantize_fp8_experts_to_nvfp4(
    model: nn.Module, rank0_only: bool | None = None
) -> int:
    """Replace block-fp8 fused-3D expert params with fused ``NVFP4Tensor`` params, in place.

    Matches any module exposing block-fp8 ``gate_up_proj``/``down_proj`` (float8 3D) + ``*_scale_inv``.
    ``rank0_only`` (FSDP2 cpu_ram_efficient): non-rank-0 emits a **meta** NVFP4 placeholder that
    FSDP2's nvfp4-aware rank-0 broadcast fills (``None``/auto also takes the meta path when the source
    is already meta). Returns the number of expert modules converted. No-op for already-NVFP4 /
    non-block-fp8 modules."""
    from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (
        is_nvfp4_param,
    )

    force_meta = bool(rank0_only) and _is_nonrank0_process()
    n = 0
    for mod in model.modules():
        gu = getattr(mod, "gate_up_proj", None)
        dn = getattr(mod, "down_proj", None)
        if gu is None or dn is None:
            continue
        if is_nvfp4_param(gu):  # already nvfp4 (native modelopt ckpt)
            continue
        gus = getattr(mod, "gate_up_proj_scale_inv", None)
        dns = getattr(mod, "down_proj_scale_inv", None)
        if not (
            _is_block_fp8_3d(gu)
            and _is_block_fp8_3d(dn)
            and gus is not None
            and dns is not None
        ):
            continue
        use_meta = force_meta or gu.is_meta
        with torch.no_grad():
            if use_meta:
                # meta placeholder: keep the whole pipeline on meta (NO .to(dev)); the NVFP4Tensor
                # param carries its own scale, so FSDP2's rank-0 broadcast fills it.
                _m = lambda t: t if t.is_meta else t.to("meta")  # noqa: E731
                gu_nv = _fused_nvfp4(_dequant_fp8_3d(_m(gu), _m(gus)))
                dn_nv = _fused_nvfp4(_dequant_fp8_3d(_m(dn), _m(dns)))
            else:
                dev = gu.device
                gu_nv = _fused_nvfp4(_dequant_fp8_3d(gu, gus).to(dev))
                dn_nv = _fused_nvfp4(_dequant_fp8_3d(dn, dns).to(dev))
        mod.gate_up_proj = nn.Parameter(gu_nv, requires_grad=False)
        mod.down_proj = nn.Parameter(dn_nv, requires_grad=False)
        for attr in ("gate_up_proj_scale_inv", "down_proj_scale_inv"):
            _drop(mod, attr)
        n += 1
    if n:
        LOG.info(
            "requantized %d block-fp8 expert modules to fused NVFP4 for scattermoe%s",
            n,
            " (rank0-only meta placeholders)" if force_meta else "",
        )
    return n
