"""Smoke 4: end-to-end grouped MoE-LoRA forward + backward, fp4_cute vs STE oracle.

Real torchao NVFP4Tensor base weights (constructed as ``fuse_nvfp4_experts``
builds them), bf16 LoRA A/B and router, ``backend="fp4_cute"`` through
``grouped_moe_reference_forward``. The tight oracle replicates the
implementation in pure torch: the same activation quantization at both GEMMs
(straight-through in backward, matching the chunked-dequant dX), fp32
matmuls rounded to bf16 like the kernel's D store, the same post-GEMM
per-expert ``per_tensor_scale`` row scaling (fp4_cute_ops never folds pts
into the e4m3 block scales, so non-power-of-2 values are exact), and it
reuses the implementation's own LoRA-delta / activation / route / combine
functions so only the base GEMM differs.

The dequant-backend diff is printed as info: it quantifies the W4A4
activation-quant error, not a kernel bug.
"""

from _common import finish, report, report_norm, require_sm100


def main():
    import torch

    require_sm100()

    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import gated_activation
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
        _lora_delta_per_group,
        combine_expert_outputs,
        grouped_moe_reference_forward,
        route_and_group,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        dequantize_nvfp4_ref,
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

    torch.manual_seed(12)
    A1 = torch.randn(r * E, H, device=dev, dtype=dtype) * H**-0.5
    B1 = torch.randn(2 * I, r * E, device=dev, dtype=dtype) * 0.02
    A2 = torch.randn(r * E, I, device=dev, dtype=dtype) * I**-0.5
    B2 = torch.randn(H, r * E, device=dev, dtype=dtype) * 0.02
    hidden = torch.randn(T, H, device=dev, dtype=dtype)
    # randint high is exclusive: expert E-1 stays empty
    top_k_index = torch.randint(0, E - 1, (T, top_k), device=dev)
    top_k_weights = torch.softmax(torch.randn(T, top_k, device=dev), dim=-1)
    g_out = torch.randn(T, H, device=dev, dtype=dtype)

    def leaves():
        return tuple(t.clone().requires_grad_() for t in (hidden, A1, B1, A2, B2))

    # --- implementation: fp4_cute backend ---
    h_i, A1_i, B1_i, A2_i, B2_i = leaves()
    out = grouped_moe_reference_forward(
        h_i,
        top_k_index,
        top_k_weights,
        w1,
        None,
        w2,
        None,
        (A1_i, B1_i),
        (A2_i, B2_i),
        E,
        act="silu",
        backend="fp4_cute",
        concat=True,
        scaling1=scaling1,
        scaling2=scaling2,
    )
    (out.float() * g_out.float()).sum().backward()
    torch.cuda.synchronize()

    # --- STE oracle ---
    def ste_quant(x):
        q, s, _ = quantize_nvfp4_ref(x.detach())
        dq = dequantize_nvfp4_ref(q, s).to(x.dtype)
        return x + (dq - x).detach()

    def base_gemm(x, w, offsets):
        # kernel view of the weights: stored scales, NO pts; then the same
        # post-GEMM per-expert row scaling in fp32 that fp4_cute_ops applies
        w_np = dequantize_nvfp4_ref(w.qdata, w.scale)
        pts = w.per_tensor_scale.view(-1)
        xq = ste_quant(x)
        outs = []
        for e in range(E):
            s0, e0 = int(offsets[e]), int(offsets[e + 1])
            o = (xq[s0:e0].float() @ w_np[e].t()).to(dtype)
            outs.append((o.float() * pts[e]).to(dtype))
        return torch.cat(outs, dim=0)

    h_o, A1_o, B1_o, A2_o, B2_o = leaves()
    xg, offsets, gidx, wg = route_and_group(h_o, top_k_index, top_k_weights, E)
    h = base_gemm(xg, w1, offsets) + _lora_delta_per_group(
        xg, offsets, A1_o, B1_o, scaling1, E, 2 * I, H
    )
    a = gated_activation(h, "silu", concat=True)
    y = base_gemm(a, w2, offsets) + _lora_delta_per_group(
        a, offsets, A2_o, B2_o, scaling2, E, H, I
    )
    out_ref = combine_expert_outputs(y, gidx, wg, T)
    (out_ref.float() * g_out.float()).sum().backward()

    report("e2e lora fp4_cute: forward", out, out_ref)
    # gradients: impl contracts dW = g^T x then maps to dA/dB; the oracle's
    # autograd contracts g^T (x A^T). Same math, different bf16 rounding, so
    # compare by norm (verified: both orders sit ~equally far from fp64 truth).
    report_norm("e2e lora fp4_cute: d hidden", h_i.grad, h_o.grad)
    report_norm("e2e lora fp4_cute: d lora_A1", A1_i.grad, A1_o.grad)
    report_norm("e2e lora fp4_cute: d lora_B1", B1_i.grad, B1_o.grad)
    report_norm("e2e lora fp4_cute: d lora_A2", A2_i.grad, A2_o.grad)
    report_norm("e2e lora fp4_cute: d lora_B2", B2_i.grad, B2_o.grad)

    # --- dequant backend, info only: the W4A4 activation-quant error ---
    h_d, A1_d, B1_d, A2_d, B2_d = leaves()
    out_dq = grouped_moe_reference_forward(
        h_d,
        top_k_index,
        top_k_weights,
        w1,
        None,
        w2,
        None,
        (A1_d, B1_d),
        (A2_d, B2_d),
        E,
        act="silu",
        backend="dequant",
        concat=True,
        scaling1=scaling1,
        scaling2=scaling2,
    )
    rel = float(
        (out.float() - out_dq.float()).abs().mean() / out_dq.float().abs().mean()
    )
    print(f"[info] fp4_cute vs dequant forward mean rel diff (W4A4 error): {rel:.4e}")

    finish()


if __name__ == "__main__":
    main()
