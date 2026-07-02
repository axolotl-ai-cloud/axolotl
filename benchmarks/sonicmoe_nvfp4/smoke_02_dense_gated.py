"""Smoke 2: THE composition bet. Dense blockscaled NVFP4 + gated epilogue.

First run of ``GemmGatedSm100(sf_vec_size=16)``: quack's blockscaled mainloop
with its gated (swiglu) epilogue, which no stock quack driver wires together.
Dense (non-varlen) to isolate the composition from the varlen machinery.
Checks postact, the stored preact, and the alpha scalar against an fp32 oracle.
"""

from _common import finish, report, require_sm100


def main():
    import torch
    import torch.nn.functional as F

    require_sm100()

    from fp4_cute import dense_nvfp4_gemm
    from nvfp4_quant import dequantize_nvfp4_ref, quantize_nvfp4_ref

    torch.manual_seed(1)
    L, M, N, K = 3, 384, 512, 256
    alpha = 0.75

    x = torch.randn(L, M, K, device="cuda") * K**-0.5
    w = torch.randn(L, N, K, device="cuda") * K**-0.5

    a_q, a_s, _ = quantize_nvfp4_ref(x.reshape(L * M, K))
    b_q, b_s, _ = quantize_nvfp4_ref(w.reshape(L * N, K))
    a_q, a_s = a_q.view(L, M, -1), a_s.view(L, M, -1)
    b_q, b_s = b_q.view(L, N, -1), b_s.view(L, N, -1)

    a_dq = dequantize_nvfp4_ref(a_q, a_s)
    b_dq = dequantize_nvfp4_ref(b_q, b_s)
    h = alpha * torch.einsum("lmk,lnk->lmn", a_dq, b_dq)
    # quack's gated epilogue pairs interleaved columns: post[j] = act(h[2j], h[2j+1])
    post_ref = F.silu(h[..., 0::2]) * h[..., 1::2]

    # gated, with preact
    postact, preact = dense_nvfp4_gemm(
        a_q, a_s, b_q, b_s, gated=True, activation="swiglu", alpha=alpha
    )
    torch.cuda.synchronize()
    report("dense gated nvfp4: postact (swiglu)", postact, post_ref)
    report("dense gated nvfp4: preact (D, alpha applied)", preact, h)

    # gated, postact only (no preact store)
    postact2, none_pre = dense_nvfp4_gemm(
        a_q,
        a_s,
        b_q,
        b_s,
        gated=True,
        activation="swiglu",
        alpha=alpha,
        store_preact=False,
    )
    torch.cuda.synchronize()
    assert none_pre is None
    report("dense gated nvfp4: postact only", postact2, post_ref)

    # non-gated (down-proj shape), default epilogue + alpha
    out = dense_nvfp4_gemm(a_q, a_s, b_q, b_s, gated=False, alpha=alpha)
    torch.cuda.synchronize()
    report("dense default nvfp4 + alpha", out, h)

    finish()


if __name__ == "__main__":
    main()
