"""Smoke 1: upstream ground truth + host-prep parity.

Runs quack's OWN dense blockscaled NVFP4 GEMM (GemmDefaultSm100, the path its
tests cover) and checks it against the dequantized-operand fp32 oracle. Also
asserts our CPU scale packer is bit-identical to quack's and sanity-checks our
quantize/dequantize reference. Nothing sonicmoe-new runs here; if this fails,
the environment (not our driver) is the problem.
"""

from _common import check, finish, report, require_sm100


def main():
    import torch

    require_sm100()
    import cutlass
    from nvfp4_quant import dequantize_nvfp4_ref, quantize_nvfp4_ref
    from quack.blockscaled_gemm_utils import (
        compile_blockscaled_gemm_tvm_ffi,
        create_blockscaled_operand_quantized,
        pack_scale_2d_to_blocked_contig,
    )
    from sf_layout import pack_scales_blocked

    torch.manual_seed(0)

    # --- parity: our packer vs quack's, bit for bit ---
    raw = torch.randint(0, 256, (3, 200, 6), dtype=torch.uint8, device="cuda")
    raw_e4m3 = raw.view(torch.float8_e4m3fn)
    ours = pack_scales_blocked(raw_e4m3).view(torch.uint8)
    theirs = pack_scale_2d_to_blocked_contig(raw_e4m3).view(torch.uint8)
    check("pack_scales_blocked bit-parity vs quack", bool(torch.equal(ours, theirs)))

    # --- our quantize/dequantize reference roundtrip sanity ---
    x = torch.randn(64, 256, device="cuda") * 0.3
    packed, scale, pts = quantize_nvfp4_ref(x)
    x_dq = dequantize_nvfp4_ref(packed, scale, pts)
    rel = ((x - x_dq).abs().mean() / x.abs().mean()).item()
    check(f"quantize/dequantize roundtrip (mean rel err {rel:.3f})", rel < 0.10)

    # --- upstream dense NVFP4 GEMM vs fp32 oracle ---
    L, M, N, K = 3, 256, 512, 256
    a_ref, qa, sfa = create_blockscaled_operand_quantized(
        L, M, K, False, 16, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN
    )
    b_ref, qb, sfb = create_blockscaled_operand_quantized(
        L, N, K, False, 16, cutlass.Float4E2M1FN, cutlass.Float8E4M3FN
    )
    d = torch.empty(L, M, N, dtype=torch.bfloat16, device="cuda").permute(1, 2, 0)

    run = compile_blockscaled_gemm_tvm_ffi(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        cutlass.BFloat16,
        (128, 128),
        (1, 1),
        qa,
        qb,
        d,
        sfa,
        sfb,
    )
    run(qa, qb, d, sfa, sfb)
    torch.cuda.synchronize()

    ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
    report("upstream dense nvfp4 GemmDefault", d, ref)

    finish()


if __name__ == "__main__":
    main()
