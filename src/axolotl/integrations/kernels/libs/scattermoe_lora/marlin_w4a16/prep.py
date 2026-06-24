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
    ws_float = marlin_scales.float() * (2**7)
    nonzero_mask = ws_float > 0
    if nonzero_mask.any():
        max_val = ws_float[nonzero_mask].max()
        if max_val < 448 * (2**7):
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


# vllm::ScalarType float4_e2m1f.id (stable packed enum id).
FLOAT4_E2M1F_ID = 562949953487106


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
):
    """Thin vLLM-free call into the standalone moe_wna16_marlin_gemm (NVFP4 W4A16, bf16 act/out)."""
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
        FLOAT4_E2M1F_ID,
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


def prepare_nvfp4_weight_for_marlin(
    qdata, block_scale, global_scale, size_n, size_k, param_dtype, repack_fn
):
    """qdata[E,size_n,size_k//2] uint8, block_scale[E,size_n,size_k//16] e4m3,
    global_scale scalar/[E] fp32. Returns (qw[E,...] int32, scales[E,...] fp8, gscale fp32)."""
    E = qdata.size(0)
    dev = qdata.device
    perm = torch.empty(0, dtype=torch.int, device=dev)

    qw = []
    for i in range(E):
        qw_i = qdata[i].view(torch.int32).T.contiguous()  # [size_k//8, size_n]
        qw.append(repack_fn(qw_i, perm, size_k, size_n, 4, False))
    qw = torch.stack(qw, 0)

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
