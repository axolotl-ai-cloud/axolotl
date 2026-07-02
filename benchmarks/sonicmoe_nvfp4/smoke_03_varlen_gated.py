"""Smoke 3: grouped (varlen_m) NVFP4 + gated epilogue, per-expert weights.

The full MoE-shaped path: expert-sorted packed A rows with uneven per-expert
counts (including an empty expert and a 1-token expert), dQaccum-padded SFA,
per-expert weight slices with a folded per-expert per_tensor_scale, fused
swiglu + preact store, plus the non-gated engine (down-proj analog).
Everything checked per expert against the fp32 dequantized-operand oracle.
"""

from _common import finish, report, require_sm100


def main():
    import torch
    import torch.nn.functional as F

    require_sm100()

    from fp4_cute import GroupedNvfp4Gemm
    from nvfp4_quant import dequantize_nvfp4_ref, quantize_nvfp4_ref
    from sf_layout import build_varlen_sfa

    torch.manual_seed(2)
    seqlens = [200, 0, 37, 128, 1]
    E = len(seqlens)
    N, K = 512, 256
    alpha = 0.9
    total_m = sum(seqlens)
    cu = torch.tensor(
        [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    x = torch.randn(total_m, K, device="cuda") * K**-0.5
    a_q, a_s, _ = quantize_nvfp4_ref(x)
    sfa = build_varlen_sfa(a_s, cu)
    a_dq = dequantize_nvfp4_ref(a_q, a_s)

    w = torch.randn(E, N, K, device="cuda") * K**-0.5
    pts = torch.tensor([0.5, 1.0, 2.0, 1.5, 0.7], device="cuda")
    b_q_list, b_s_list = [], []
    for e in range(E):
        q_e, s_e, _ = quantize_nvfp4_ref(w[e], per_tensor_scale=pts[e])
        b_q_list.append(q_e)
        b_s_list.append(s_e)
    b_q = torch.stack(b_q_list)
    b_s = torch.stack(b_s_list)
    # The kernel consumes pts FOLDED into e4m3 block scales (a re-rounding),
    # so the tight kernel oracle dequantizes with the folded scales. The
    # folding cost itself (nonzero for non-power-of-2 pts like 1.5/0.7) is
    # reported separately; it is an accepted quantization-scheme error, not a
    # kernel error (design doc open question 2).
    from sf_layout import fold_per_tensor_scale

    b_s_folded, _ = fold_per_tensor_scale(b_s, pts)
    w_dq = dequantize_nvfp4_ref(b_q, b_s_folded)
    w_dq_exact = torch.stack(
        [dequantize_nvfp4_ref(b_q[e], b_s[e], pts[e]) for e in range(E)]
    )
    fold_rel = (
        (w_dq - w_dq_exact).abs().sum() / w_dq_exact.abs().sum().clamp(min=1e-12)
    ).item()
    print(f"[info] pts-folding mean rel weight error: {fold_rel:.4e}")

    def oracle():
        pre = torch.empty(total_m, N, device="cuda")
        for e in range(E):
            s, t = int(cu[e]), int(cu[e + 1])
            pre[s:t] = alpha * (a_dq[s:t] @ w_dq[e].T)
        post = F.silu(pre[:, 0::2]) * pre[:, 1::2]
        return pre, post

    pre_ref, post_ref = oracle()

    # gated grouped engine (up-projection analog)
    up = GroupedNvfp4Gemm(N, K, E, gated=True, activation="swiglu")
    up.set_weights(b_q, b_s, per_tensor_scale=pts)
    postact, preact = up.forward(a_q, sfa, cu, alpha=alpha)
    torch.cuda.synchronize()
    report("varlen gated nvfp4: postact", postact, post_ref)
    report("varlen gated nvfp4: preact", preact, pre_ref)

    # second forward with a different token distribution, same compiled engine
    seqlens2 = [64, 100, 1, 0, 129]
    total2 = sum(seqlens2)
    cu2 = torch.tensor(
        [0] + list(torch.tensor(seqlens2).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    x2 = torch.randn(total2, K, device="cuda") * K**-0.5
    a2_q, a2_s, _ = quantize_nvfp4_ref(x2)
    sfa2 = build_varlen_sfa(a2_s, cu2)
    a2_dq = dequantize_nvfp4_ref(a2_q, a2_s)
    pre2 = torch.empty(total2, N, device="cuda")
    for e in range(E):
        s, t = int(cu2[e]), int(cu2[e + 1])
        pre2[s:t] = alpha * (a2_dq[s:t] @ w_dq[e].T)
    post2_ref = F.silu(pre2[:, 0::2]) * pre2[:, 1::2]
    postact2, preact2 = up.forward(a2_q, sfa2, cu2, alpha=alpha)
    torch.cuda.synchronize()
    report(
        "varlen gated nvfp4: postact (2nd total_m, no recompile)", postact2, post2_ref
    )
    report("varlen gated nvfp4: preact (2nd total_m)", preact2, pre2)

    # non-gated grouped engine (down-projection analog)
    down = GroupedNvfp4Gemm(N, K, E, gated=False)
    down.set_weights(b_q, b_s, per_tensor_scale=pts)
    out = down.forward(a_q, sfa, cu, alpha=alpha)
    torch.cuda.synchronize()
    report("varlen default nvfp4", out, pre_ref)

    finish()


if __name__ == "__main__":
    main()
