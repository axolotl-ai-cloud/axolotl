"""Requantize fused-3D MoE experts to OCP-MX fp8 (e4m3 data + e8m0 per-32-block scale) for the
scattermoe ``moe_grouped_mode: mxfp8`` path.

mxfp8 is the native Blackwell 8-bit tensor-core format: 1 byte/elem e4m3 plus a power-of-2 e8m0
scale every 32 elements along the contraction axis. Unlike ``fp8_block`` (128x128 fp32 scale) it is
hardware-MX-native and finer-grained. Since these models don't ship mxfp8 checkpoints, experts are
requantized at load — from bf16 (bf16 base) or from genuine 128x128 block-fp8 (FineGrainedFP8 base,
dequantized first). The e8m0 FLOOR math here matches the in-kernel decode (``2^(byte-127)``) and the
Triton ``mxfp8_quant`` activation quantizer, so forward/backward stay grad-consistent.

Replaces ``gate_up_proj``/``down_proj`` with the e4m3 tensor and attaches ``*_mx_scale`` e8m0 buffers
that :func:`grouped_mxfp8_moe_train` (and the experts dispatch) read directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_MX = 32
_E4M3_MAX = 448.0
# floor(log2(448)) = 8 (matches quant_mxfp8's FLOOR e8m0 rule)
_LOG2_E4M3_MAX = 8.0


def _quant_mx_3d(w: torch.Tensor):
    """Fused-3D bf16 ``[E, N, K]`` -> (qdata e4m3 [E, N, K], scale uint8 [E, N, K/32]). FLOOR e8m0,
    identical to the Triton ``mxfp8_quant`` and the kernel decode ``2^(byte-127)``. The scale is the
    raw e8m0 biased-exponent BYTE kept as ``uint8`` (not ``float8_e8m0fnu``) so DDP/FSDP can broadcast
    it — NCCL has no Float8_e8m0fnu collective; the kernel decodes ``2^(byte-127)`` either way."""
    E, N, K = w.shape
    assert K % _MX == 0, f"mxfp8 requant expects K % 32 == 0, got K={K}"
    nb = K // _MX
    wb = w.to(torch.float32).reshape(E, N, nb, _MX)
    amax = wb.abs().amax(dim=-1)  # [E, N, nb]
    exp = torch.floor(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax))))
    exp = exp - _LOG2_E4M3_MAX
    sc = torch.exp2(exp)  # [E, N, nb]
    q = torch.clamp(wb / sc[..., None], -_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    q = q.reshape(E, N, K)
    ebyte = (
        (exp + 127.0).clamp_(0, 254).to(torch.uint8)
    )  # e8m0 biased exponent, raw byte
    return q, ebyte


def _dequant_fp8_block_3d(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Batched 128x128 block-fp8 dequant of a fused-3D ``[E, N, K]`` weight -> bf16."""
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


def _source_to_bf16(mod: nn.Module, w: torch.Tensor, name: str) -> torch.Tensor:
    """Coerce an expert weight to bf16 for requant: block-fp8 (e4m3 + ``*_scale_inv``) is
    dequantized, everything else is cast."""
    if w.dtype == torch.float8_e4m3fn:
        scale_inv = getattr(mod, f"{name}_scale_inv", None)
        if scale_inv is not None:
            return _dequant_fp8_block_3d(w, scale_inv)
    return w.to(torch.bfloat16)


def _drop(mod: nn.Module, attr: str) -> None:
    if hasattr(mod, attr):
        try:
            delattr(mod, attr)
        except Exception:  # registered buffer/param
            setattr(mod, attr, None)


def _is_nonrank0_process() -> bool:
    """True on every process except local-rank-0 of the node (keyed off torchrun's LOCAL_RANK).
    Mirrors the NVFP4 fast-load gate; ``-1`` (unset) = single non-distributed process."""
    import os

    return int(os.environ.get("LOCAL_RANK", "-1")) > 0


def _meta_mxfp8_shapes(mod: nn.Module, gu: torch.Tensor, dn: torch.Tensor):
    """(e4m3 qdata shape, e8m0 scale shape) for gate_up/down, from the (possibly meta) source
    shapes — no data read. Scale is [E, N, K/32]."""

    def _shp(w):
        E, N, K = w.shape
        return (E, N, K), (E, N, K // _MX)

    return _shp(gu), _shp(dn)


def _set_mxfp8(mod, gq, gs, dq, ds):
    mod.gate_up_proj = nn.Parameter(gq, requires_grad=False)
    mod.down_proj = nn.Parameter(dq, requires_grad=False)
    # persistent=False: the e8m0 scale must stay OUT of state_dict — FSDP2's cpu_ram_efficient loader
    # wraps every state_dict entry in nn.Parameter (fsdp2_load_full_state_dict), and uint8 cannot be a
    # Parameter. The e4m3 weight (a valid float Parameter) rides FSDP's rank-0 broadcast; the scale is
    # broadcast separately by _broadcast_scales below when rank0_only.
    mod.register_buffer("gate_up_proj_mx_scale", gs, persistent=False)
    mod.register_buffer("down_proj_mx_scale", ds, persistent=False)
    for attr in ("gate_up_proj_scale_inv", "down_proj_scale_inv"):
        _drop(mod, attr)


def _broadcast_scales(mods) -> None:
    """Fill non-rank-0's empty e8m0 scale buffers from rank 0 (their weights are meta and get filled
    by FSDP's own rank-0 broadcast; the scale is a non-persistent buffer FSDP won't touch). Modules
    are iterated in the same order on every rank, so the collective sequence matches."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    dev = torch.device("cuda", torch.cuda.current_device())
    for mod in mods:
        for attr in ("gate_up_proj_mx_scale", "down_proj_mx_scale"):
            s = getattr(mod, attr).to(dev)
            dist.broadcast(s, src=0)
            setattr(mod, attr, None)  # replace the buffer's tensor in place
            mod.register_buffer(attr, s, persistent=False)


def requantize_experts_to_mxfp8(
    model: nn.Module, rank0_only: bool | None = None
) -> int:
    """Replace fused-3D bf16/block-fp8 expert params with mxfp8 (e4m3 param + e8m0 ``*_mx_scale``
    non-persistent buffer), in place. Matches any module exposing 3D ``gate_up_proj``/``down_proj``.

    ``rank0_only`` handles FSDP2 ``cpu_ram_efficient_loading``, where non-rank-0 expert params arrive
    on ``meta``: those ranks emit a **meta** e4m3 weight (FSDP's rank-0 broadcast fills it) + an empty
    real e8m0 scale, and :func:`_broadcast_scales` fills the scales from rank 0. ``True`` forces this
    on non-rank-0 even if the source is real; ``None`` (auto) engages per-module when the source is
    already ``meta``. Returns the number of expert modules converted. No-op for modules already
    mxfp8."""
    force_meta = bool(rank0_only) and _is_nonrank0_process()
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    converted = []
    for mod in model.modules():
        gu = getattr(mod, "gate_up_proj", None)
        dn = getattr(mod, "down_proj", None)
        if not (isinstance(gu, torch.Tensor) and isinstance(dn, torch.Tensor)):
            continue
        if gu.ndim != 3 or dn.ndim != 3:
            continue
        if getattr(mod, "gate_up_proj_mx_scale", None) is not None:
            continue  # already mxfp8
        if force_meta or gu.is_meta or dn.is_meta:
            # rank0-only: meta e4m3 weight (FSDP fills) + empty real e8m0 scale (_broadcast_scales fills)
            (gq_s, gs_s), (dq_s, ds_s) = _meta_mxfp8_shapes(mod, gu, dn)
            _set_mxfp8(
                mod,
                torch.empty(gq_s, dtype=torch.float8_e4m3fn, device="meta"),
                torch.empty(gs_s, dtype=torch.uint8, device=dev),
                torch.empty(dq_s, dtype=torch.float8_e4m3fn, device="meta"),
                torch.empty(ds_s, dtype=torch.uint8, device=dev),
            )
            converted.append(mod)
            continue
        with torch.no_grad():
            gu_bf16 = _source_to_bf16(mod, gu, "gate_up_proj").to(gu.device)
            dn_bf16 = _source_to_bf16(mod, dn, "down_proj").to(dn.device)
            gq, gs = _quant_mx_3d(gu_bf16)
            dq, ds = _quant_mx_3d(dn_bf16)
        _set_mxfp8(mod, gq, gs, dq, ds)
        converted.append(mod)
    if bool(rank0_only):
        _broadcast_scales(converted)
    if converted:
        LOG.info(
            "requantized %d expert modules to mxfp8 for scattermoe%s",
            len(converted),
            " (rank0-only + scale broadcast)" if rank0_only else "",
        )
    return len(converted)
