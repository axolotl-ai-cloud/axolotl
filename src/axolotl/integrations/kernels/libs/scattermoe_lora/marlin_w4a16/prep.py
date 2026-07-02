"""vLLM-FREE NVFP4 -> Marlin weight prep. Pure-torch scale processing (copied verbatim from
vLLM marlin_utils.py / marlin_utils_fp4.py so it is bit-identical) + the standalone CUDA
gptq_marlin_repack (ported, bit-exact to vLLM). Produces (b_qweight int32, b_scales fp8_e4m3,
global_scale fp32) ready for the standalone moe_wna16_marlin_gemm.

One function does ONE weight tensor [E, size_n, size_k//2] (gate_up fused or down). The training
integration calls it 4x: gate_up + down (forward), and their transposes (backward dX).
"""

import torch


# Copied verbatim from vllm marlin_utils.py (keep bit-identical).
def _num_compute_units(idx):
    return torch.cuda.get_device_properties(idx).multi_processor_count


def marlin_make_workspace_new(device, max_blocks_per_sm=1):
    sms = _num_compute_units(device.index)
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def get_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s, size_k, size_n, group_size, is_a_8bit=False):
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s


# Copied verbatim from vllm marlin_utils_fp4.py (keep bit-identical).
def _nvfp4_compute_scale_factor(marlin_scales, a_dtype=None):
    if a_dtype is not None and a_dtype == torch.half:
        return 1.0
    # Value-identical, memory-cheap form of vllm's `ws_float[ws_float>0].max()`: the max of the
    # positive entries equals the overall amax whenever any positive exists. Computing amax in the
    # native (bf16) dtype avoids materializing a full float copy AND a boolean-index copy of the
    # [E,N,K/16] scales (~9 GiB at GLM-5.2 dims) — that copy OOMed the gradient-checkpoint backward
    # recompute. bf16 amax selects the exact max element; the *128 scaling is then a scalar float op.
    max_val = marlin_scales.amax().float() * (2**7)
    if max_val > 0 and max_val < 448 * (2**7):
        sf = (448 * (2**7) / max_val).log2().floor().exp2()
        return sf.item()
    return 1.0


def nvfp4_marlin_process_scales(marlin_scales, scale_factor=None, a_dtype=None):
    marlin_scales = marlin_scales.to(torch.half)
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        marlin_scales.size(0), -1
    )
    if scale_factor is None:
        scale_factor = _nvfp4_compute_scale_factor(marlin_scales, a_dtype)
    if scale_factor > 1.0:
        marlin_scales = (marlin_scales.float() * scale_factor).to(torch.half)
    marlin_scales = marlin_scales * (2**7)
    marlin_scales[marlin_scales < 2] = 0
    marlin_scales = marlin_scales.view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()
    return marlin_scales, scale_factor


def nvfp4_marlin_process_global_scale(global_scale, a_dtype=None):
    if a_dtype is None:
        a_dtype = global_scale.dtype
    assert a_dtype in [torch.half, torch.bfloat16]
    fp4_exponent = 2
    target_exponent = 5 if a_dtype == torch.half else 8
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


# vllm::ScalarType packed enum ids (stable; the bit-packing is replicated + validated in tests).
FLOAT4_E2M1F_ID = 562949953487106
FLOAT8_E4M3FN_ID = 2814749767172868


def marlin_moe_gemm(
    ext,
    a,
    b_qweight,
    b_scales,
    global_scale,
    workspace,
    sorted_token_ids,
    expert_ids,
    num_tokens_past_padded,
    topk_weights,
    moe_block_size,
    top_k,
    mul_topk_weights,
    size_m,
    size_n,
    size_k,
    out=None,
    b_type_id=FLOAT4_E2M1F_ID,
):
    """Thin vLLM-free call into the standalone moe_wna16_marlin_gemm (bf16 act/out).
    ``b_type_id`` selects the weight format: FLOAT4_E2M1F_ID (NVFP4 W4A16, requires global_scale)
    or FLOAT8_E4M3FN_ID (block-fp8 W8A16, bf16 group-128 scales; global_scale must be None)."""
    if out is None:
        out = torch.empty((size_m * top_k, size_n), device=a.device, dtype=a.dtype)
    return ext.moe_wna16_marlin_gemm(
        a,
        out,
        b_qweight,
        None,
        b_scales,
        None,
        global_scale,
        None,
        None,
        None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        b_type_id,
        size_m,
        size_n,
        size_k,
        True,
        False,
        True,
        False,
        -1,
        -1,
        -1,
    )


GROUP_SIZE = 16

# fp4 e2m1 magnitude decision boundaries (round-to-nearest between codebook entries).
_CB_BOUND = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_nvfp4_blockwise(W: torch.Tensor, pt: torch.Tensor):
    """bf16/f32 ``W [C, N, K]`` + per-expert scale ``pt [C]`` -> loader-layout NVFP4:
    ``qdata [C, N, K/2]`` uint8 (low nibble = even k), ``scale [C, N, K/16]`` e4m3.
    Encodes against the e4m3-DECODED block scale so dequant reproduces the encode grid."""
    C, N, K = W.shape
    bound = torch.tensor(_CB_BOUND, device=W.device)
    wb = W.float().reshape(C, N, K // 16, 16)
    amax = wb.abs().amax(-1)
    sc = (
        (amax / (6.0 * pt.view(C, 1, 1)))
        .clamp_(min=0.0, max=448.0)
        .to(torch.float8_e4m3fn)
    )
    sc_dec = sc.float().clamp_min(1e-12) * pt.view(C, 1, 1)
    v = (wb / sc_dec.unsqueeze(-1)).reshape(C, N, K)
    mag_idx = torch.bucketize(v.abs(), bound).to(torch.uint8)
    idx = torch.where(v < 0, mag_idx + 8, mag_idx)
    return idx[..., 0::2] | (idx[..., 1::2] << 4), sc


def prepare_transposed_dx_weight(qdata, block_scale, pt, size_n, size_k, repack_fn):
    """Build the backward-dX Marlin weight: requantize ``W^T`` with fresh block-16 scales along its
    new reduction axis (the forward scales sit on K and land on the wrong axis under transpose),
    then Marlin-pack. dX then runs as a plain W4A16 grouped GEMM: ``g [Mt, size_k] -> [Mt, size_n]``.

    The requant is a second 4-bit rounding, so dX reads a slightly different (but FIXED) weight than
    the forward (~9% rel on the weight; same tradeoff class as the fp8-read dX path's ~3%). Chunked
    over experts to bound the bf16 transient.

    qdata ``[E, size_k, size_n//2]`` / block_scale ``[E, size_k, size_n//16]`` are the FORWARD-layout
    tensors (forward N = our size_k reduction). Returns the (qw, scales, gscale) marlin tuple for a
    ``[E, size_n, size_k]`` logical weight."""
    from ..dequant_grouped import nvfp4_dequant_bf16

    E = qdata.size(0)
    dev = qdata.device
    # amax is transpose-invariant, but the checkpoint pt may be smaller than amax/(6*448) for some
    # expert -> fresh block scales would clip at 448; lift pt per-expert to the safe floor.
    pt = pt.reshape(-1).float()
    if pt.numel() == 1:
        pt = pt.expand(E)
    # Requant AND marlin-repack chunk-by-chunk: the requantized qT never coexists whole with the
    # packed output (only the small e4m3 block scales are held across chunks), so the build peak
    # stays ~one pack + one chunk instead of two full packs.
    from .batched_repack import batched_repack4

    CHUNK = 16
    sT = torch.empty(E, size_n, size_k // 16, dtype=torch.float8_e4m3fn, device=dev)
    ptT = pt.clone()
    qw = []
    for c0 in range(0, E, CHUNK):
        c1 = min(c0 + CHUNK, E)
        Wd = nvfp4_dequant_bf16(qdata[c0:c1], block_scale[c0:c1], pt[c0:c1])
        WdT = Wd.transpose(1, 2).contiguous()
        del Wd
        amax = WdT.float().abs().amax(dim=(1, 2))
        ptT[c0:c1] = torch.maximum(pt[c0:c1], amax / (6.0 * 448.0)).clamp_min(1e-12)
        qc, sT[c0:c1] = quantize_nvfp4_blockwise(WdT, ptT[c0:c1])
        del WdT
        qw.append(batched_repack4(qc, size_n, size_k))
        del qc
    qw = torch.cat(qw, 0)

    # Shared scale_factor across experts, then per-expert permute+process (mirrors
    # prepare_nvfp4_weight_for_marlin, operating on the held scales).
    scales_p = sT.to(torch.bfloat16)
    csf = _nvfp4_compute_scale_factor(scales_p, torch.bfloat16)
    sc = []
    for i in range(E):
        ms = marlin_permute_scales(
            scales_p[i].T, size_k=size_k, size_n=size_n, group_size=GROUP_SIZE
        )
        ms, _ = nvfp4_marlin_process_scales(
            ms, scale_factor=csf, a_dtype=torch.bfloat16
        )
        sc.append(ms)
    sc = torch.stack(sc, 0)
    g = nvfp4_marlin_process_global_scale(ptT.to(torch.float32), torch.bfloat16) / csf
    return qw, sc, g.float()


FP8_BLOCK = 128


def prepare_blockfp8_dx_weight(wq, ws, repack_fn):
    """EXACT transposed Marlin W8A16 pack for the block-fp8 backward dX (zero requant error).

    128x128 block scales are transpose-symmetric: ``W^T`` = byte-permute the e4m3 qdata +
    transpose the scale grid — no dequant, no requant (the TE#2168 property). dX then runs as a
    plain W8A16 grouped GEMM ``g [Mt, N] -> [Mt, K]`` with the weight in marlin's natural
    orientation (``size_n = K``, ``size_k = N``, bf16 group-128 scales).

    wq ``[E, N, K]`` float8_e4m3fn, ws ``[E, N/128, K/128]`` fp32 (``weight_scale_inv``).
    Returns ``(qw int32 [E, ...], scales bf16 [E, ...], None)`` — the marlin tuple shape used by
    ``marlin_moe_gemm`` with ``b_type_id=FLOAT8_E4M3FN_ID`` (no global scale)."""
    E, N, K = wq.shape

    # qdata: fused transpose+repack for ALL experts in one launch (bit-identical to the
    # per-expert gptq_marlin_repack loop; see batched_repack.py).
    from .batched_repack import batched_repack8_transposed

    qw = batched_repack8_transposed(wq, _ext_from_repack(repack_fn))

    return qw, _marlin_fp8_group_scales(ws, K), None


def _marlin_fp8_group_scales(ws_grid, size_n):
    """Marlin bf16 group-128 scales ``[E, size_k/128, size_n]`` from a block-scale grid
    ``[E, size_k/128, size_n/128]``, vectorized across E. The marlin fp8 dequant bit-trick
    leaves values scaled by 2^-120 for bf16 (2^-8 for fp16), so the exponent bias is fused
    into the scales (vllm fp8_fused_exponent_bias_into_scales)."""
    E = ws_grid.size(0)
    s = (ws_grid.float() * 2.0**120).to(torch.bfloat16)
    s = s.repeat_interleave(FP8_BLOCK, dim=2)
    scale_perm, _ = get_scale_perms()
    return (
        s.reshape(E, -1, len(scale_perm))[:, :, scale_perm]
        .reshape(E, -1, size_n)
        .contiguous()
    )


def prepare_blockfp8_fwd_weight(wq, ws, repack_fn):
    """Marlin W8A16 pack of the block-fp8 weight in FORWARD orientation: ``x [Mt, K] -> [Mt, N]``
    (``size_n = N``, ``size_k = K``). Exact — same byte permute + scale recipe as the dX pack.

    wq ``[E, N, K]`` float8_e4m3fn, ws ``[E, N/128, K/128]`` fp32. Returns the marlin tuple
    ``(qw, scales bf16, None)`` for ``marlin_moe_gemm`` with ``b_type_id=FLOAT8_E4M3FN_ID``."""
    from .batched_repack import batched_repack8_fwd

    _, N, _ = wq.shape
    qw = batched_repack8_fwd(wq, _ext_from_repack(repack_fn))
    return qw, _marlin_fp8_group_scales(ws.transpose(1, 2).contiguous(), N), None


def _ext_from_repack(repack_fn):  # noqa: ARG001 - signature kept for call-site symmetry
    """Return the (cached) ext module for the LUT probe. pybind functions don't expose a usable
    ``__self__``, so this just loads the cached extension."""
    from . import load_ext

    return load_ext()


def prepare_nvfp4_weight_for_marlin(
    qdata, block_scale, global_scale, size_n, size_k, param_dtype, repack_fn
):
    """qdata[E,size_n,size_k//2] uint8, block_scale[E,size_n,size_k//16] e4m3,
    global_scale scalar/[E] fp32. Returns (qw[E,...] int32, scales[E,...] fp8, gscale fp32)."""
    E = qdata.size(0)

    # one-launch batched repack (bit-identical to the per-expert gptq_marlin_repack loop; also
    # 2x lower build transient — no per-expert transpose copies, no stack duplication)
    from .batched_repack import batched_repack4

    qw = batched_repack4(qdata, size_n, size_k)

    # Shared scale_factor across experts, then per-expert permute+process.
    scales_p = block_scale.to(param_dtype)
    csf = _nvfp4_compute_scale_factor(scales_p, param_dtype)
    sc = []
    for i in range(E):
        s = scales_p[i].T  # [size_k//16, size_n]
        ms = marlin_permute_scales(
            s, size_k=size_k, size_n=size_n, group_size=GROUP_SIZE
        )
        ms, _ = nvfp4_marlin_process_scales(ms, scale_factor=csf, a_dtype=param_dtype)
        sc.append(ms)
    sc = torch.stack(sc, 0)

    g = (
        nvfp4_marlin_process_global_scale(global_scale.to(torch.float32), param_dtype)
        / csf
    )
    return qw, sc, g.float()
