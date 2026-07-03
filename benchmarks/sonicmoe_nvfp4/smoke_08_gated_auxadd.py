"""Smoke 8: fused gated up-GEMM with concat-B view, colvec pts, and aux add.

Kernel level: ``GroupedNvfp4Gemm(gated=True, concat_b=True)`` consumes CONCAT
[gate; up] weights zero-copy through quack's ``concat_layout`` interleaved
view (SFB packed from row-permuted scales), multiplies the exact per-row pts
colvec into the fp32 accumulator, ADDs a preact-space aux (the LoRA delta,
concat layout, viewed interleaved by the TileLoad), then applies swiglu, all
in one kernel. Checked against the fp32 dequantized-operand oracle: the
INTERLEAVED preact D must be BITWISE (single bf16 rounding), the postact
within activation-math tolerance.

Module level: the fused up+act autograd path (AXOLOTL_SONICMOE_NVFP4_FUSED_UP)
vs the unfused path on identical inputs: forward within bf16 rounding noise,
LoRA grads within bf16 accumulation noise.
"""

from _common import check, finish, report, report_norm, require_sm100


def kernel_level():
    import torch
    import torch.nn.functional as F
    from fp4_cute import GroupedNvfp4Gemm
    from nvfp4_quant import dequantize_nvfp4_ref, quantize_nvfp4_ref
    from sf_layout import build_varlen_sfa

    torch.manual_seed(3)
    seqlens = [200, 0, 37, 128, 1]
    E = len(seqlens)
    N, K = 512, 256  # N = 2I concat [gate; up]
    half = N // 2
    total_m = sum(seqlens)
    cu = torch.tensor(
        [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    counts = (cu[1:] - cu[:-1]).long()

    x = torch.randn(total_m, K, device="cuda") * K**-0.5
    a_q, a_s, _ = quantize_nvfp4_ref(x)
    sfa = build_varlen_sfa(a_s, cu)
    a_dq = dequantize_nvfp4_ref(a_q, a_s)

    w = torch.randn(E, N, K, device="cuda") * K**-0.5  # CONCAT layout
    pts = torch.tensor([0.5, 1.0, 2.0, 1.5, 0.7], device="cuda")
    b_q = torch.stack([quantize_nvfp4_ref(w[e])[0] for e in range(E)])
    b_s = torch.stack([quantize_nvfp4_ref(w[e])[1] for e in range(E)])
    # exact scheme: stored scales, FULL pts per row via colvec
    w_dq = dequantize_nvfp4_ref(b_q, b_s)  # [E, N, K] fp32, concat rows
    colvec = torch.repeat_interleave(pts, counts)
    aux = (torch.randn(total_m, N, device="cuda") * 0.1).to(torch.bfloat16)

    def oracle():
        pre = torch.empty(total_m, N, device="cuda")  # CONCAT columns
        for e in range(E):
            s, t = int(cu[e]), int(cu[e + 1])
            pre[s:t] = pts[e] * (a_dq[s:t] @ w_dq[e].T)
        pre += aux.float()
        post = F.silu(pre[:, :half]) * pre[:, half:]
        return pre, post

    pre_ref, post_ref = oracle()

    eng = GroupedNvfp4Gemm(N, K, E, gated=True, activation="swiglu", concat_b=True)
    eng.set_weights(b_q, b_s)
    postact, preact_il = eng.forward(a_q, sfa, cu, colvec=colvec, aux=aux)
    torch.cuda.synchronize()

    # preact memory is interleaved: view (T, I, 2) -> gate at 0, up at 1
    pre_il = preact_il.view(total_m, half, 2)
    pre_kernel_concat = torch.cat([pre_il[..., 0], pre_il[..., 1]], dim=1)
    check(
        "concat_b gated+colvec+aux: preact BITWISE vs fp32 oracle",
        torch.equal(pre_kernel_concat, pre_ref.to(torch.bfloat16)),
    )
    report("concat_b gated+colvec+aux: preact", pre_kernel_concat, pre_ref)
    report("concat_b gated+colvec+aux: postact", postact, post_ref)

    # second forward, different token distribution, same compiled engine
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
    colvec2 = torch.repeat_interleave(pts, (cu2[1:] - cu2[:-1]).long())
    aux2 = (torch.randn(total2, N, device="cuda") * 0.1).to(torch.bfloat16)
    pre2 = torch.empty(total2, N, device="cuda")
    for e in range(E):
        s, t = int(cu2[e]), int(cu2[e + 1])
        pre2[s:t] = pts[e] * (a2_dq[s:t] @ w_dq[e].T)
    pre2 += aux2.float()
    post2_ref = F.silu(pre2[:, :half]) * pre2[:, half:]
    postact2, preact2_il = eng.forward(a2_q, sfa2, cu2, colvec=colvec2, aux=aux2)
    torch.cuda.synchronize()
    pre2_il = preact2_il.view(total2, half, 2)
    check(
        "concat_b (2nd total_m, no recompile): preact BITWISE",
        torch.equal(
            torch.cat([pre2_il[..., 0], pre2_il[..., 1]], dim=1),
            pre2.to(torch.bfloat16),
        ),
    )
    report("concat_b (2nd total_m): postact", postact2, post2_ref)


def module_level():
    import torch
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
        grouped_moe_reference_forward,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        quantize_nvfp4_ref,
    )

    torch.manual_seed(7)
    E, H, I, r, top_k, T = 6, 256, 192, 8, 2, 311
    scaling1, scaling2 = 0.5, 0.25
    dtype = torch.bfloat16
    dev = "cuda"

    def make_weight(n, k, pts_vals, seed):
        torch.manual_seed(seed)
        pts = torch.tensor(pts_vals, device=dev)
        dense = torch.randn(E, n, k, device=dev) * k**-0.5
        qs, ss = [], []
        for e in range(E):
            q, s, _ = quantize_nvfp4_ref(dense[e], pts[e])
            qs.append(q)
            ss.append(s)
        return NVFP4Tensor(
            torch.stack(qs),
            torch.stack(ss),
            16,
            dtype,
            per_tensor_scale=pts.view(-1, 1, 1),
        )

    w1 = make_weight(2 * I, H, [0.5, 1.3, 2.0, 0.25, 0.7, 4.0], seed=10)
    w2 = make_weight(H, I, [1.0, 0.6, 1.1, 2.0, 0.5, 1.7], seed=11)

    torch.manual_seed(21)
    hidden = torch.randn(T, H, device=dev, dtype=dtype) * 0.5
    # randint high is exclusive: expert E-1 stays empty
    top_i = torch.randint(0, E - 1, (T, top_k), device=dev)
    top_w = torch.softmax(torch.randn(T, top_k, device=dev), dim=-1).to(dtype)

    def run(fused: bool):
        import os

        os.environ["AXOLOTL_SONICMOE_NVFP4_FUSED_UP"] = "1" if fused else "0"
        torch.manual_seed(33)
        A1 = torch.randn(r * E, H, device=dev, dtype=dtype, requires_grad=True)
        B1 = torch.randn(2 * I, r * E, device=dev, dtype=dtype, requires_grad=True)
        A2 = torch.randn(r * E, I, device=dev, dtype=dtype, requires_grad=True)
        B2 = torch.randn(H, r * E, device=dev, dtype=dtype, requires_grad=True)
        for t in (A1, B1, A2, B2):
            t.data *= 0.05
        hid = hidden.clone().requires_grad_(True)
        out = grouped_moe_reference_forward(
            hid,
            top_i,
            top_w,
            w1,
            None,
            w2,
            None,
            (A1, B1),
            (A2, B2),
            E,
            act="silu",
            backend="fp4_cute",
            concat=True,
            scaling1=scaling1,
            scaling2=scaling2,
        )
        out.float().square().mean().backward()
        return out, hid.grad, A1.grad, B1.grad, A2.grad, B2.grad

    out_f, dh_f, dA1_f, dB1_f, dA2_f, dB2_f = run(fused=True)
    out_u, dh_u, dA1_u, dB1_u, dA2_u, dB2_u = run(fused=False)

    # fused rounds the preact once (fp32 acc + delta -> bf16) where unfused
    # rounds base and delta separately and acts on the bf16 sum: bf16-level
    # differences are expected, not bugs.
    report_norm("module fused vs unfused: forward", out_f, out_u, tol=2e-2)
    report_norm("module fused vs unfused: d_hidden", dh_f, dh_u, tol=3e-2)
    report_norm("module fused vs unfused: d_lora_A1", dA1_f, dA1_u, tol=3e-2)
    report_norm("module fused vs unfused: d_lora_B1", dB1_f, dB1_u, tol=3e-2)
    report_norm("module fused vs unfused: d_lora_A2", dA2_f, dA2_u, tol=3e-2)
    report_norm("module fused vs unfused: d_lora_B2", dB2_f, dB2_u, tol=3e-2)


def main():
    import os

    os.environ.setdefault("AXOLOTL_SONICMOE_NVFP4_BWD", "bf16")
    require_sm100()
    kernel_level()
    module_level()
    finish()


if __name__ == "__main__":
    main()
