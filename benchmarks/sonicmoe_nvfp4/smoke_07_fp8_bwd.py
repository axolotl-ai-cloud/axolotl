"""Smoke 7: fp8 backward weight cache (DeepGEMM) vs the bf16 dequant backward.

Same construction as smoke 4 (real torchao NVFP4Tensor base, bf16 LoRA,
``backend="fp4_cute"``) at Qwen3-30B-like projection shapes, run twice with
``AXOLOTL_SONICMOE_NVFP4_BWD=bf16`` and ``=deepgemm``:

- forward outputs must be BITWISE identical (the backward engine cannot touch
  the forward);
- down-proj LoRA grads (dA2/dB2) must be BITWISE identical (they depend only
  on the top gradient and forward activations, not on any dX);
- hidden grad and up-proj LoRA grads shift only through the fp8 base-dX term:
  compared by relative Frobenius norm. Expected ~3e-2: DeepGEMM's own gate is
  calc_diff < 1e-3, a squared-similarity metric, and calc_diff ~= rel_fro^2 /
  2 for small errors, so their 7e-4 IS ~3.7e-2 rel_fro. The 30-step loss
  trajectory is the real quality gate;
- unit check: grouped_fp8_dx vs the dequant grouped GEMM directly;
- perf: full-backward wall time under both engines (info only; includes the
  pad/scatter/quant/gather overhead the microbench excluded).

SKIPs (exit 0) when deep_gemm is not importable; see DEEPGEMM.md for setup.
"""

import os

from _common import check, finish, report_norm, require_sm100


def main():
    import torch

    require_sm100()

    from axolotl.integrations.kernels.libs.sonicmoe.fp8_bwd import _deep_gemm

    if _deep_gemm() is None:
        print("[SKIP] deep_gemm not importable; see DEEPGEMM.md for the setup steps")
        return

    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute_ops import (
        dequantize_engine_weight,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.fp8_bwd import grouped_fp8_dx
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
        grouped_moe_reference_forward,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        quantize_nvfp4_ref,
    )

    torch.manual_seed(7)
    E, H, I, r, top_k, T = 32, 2048, 768, 16, 4, 4096
    scaling1, scaling2 = 0.5, 0.25
    dtype = torch.bfloat16
    dev = "cuda"

    def make_weight(n, k, seed):
        torch.manual_seed(seed)
        pts = (torch.rand(E, device=dev) * 3.0 + 0.3).float()
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

    w1 = make_weight(2 * I, H, seed=10)
    w2 = make_weight(H, I, seed=11)

    torch.manual_seed(12)
    A1 = torch.randn(r * E, H, device=dev, dtype=dtype) * H**-0.5
    B1 = torch.randn(2 * I, r * E, device=dev, dtype=dtype) * 0.02
    A2 = torch.randn(r * E, I, device=dev, dtype=dtype) * I**-0.5
    B2 = torch.randn(H, r * E, device=dev, dtype=dtype) * 0.02
    hidden = torch.randn(T, H, device=dev, dtype=dtype)
    # randint high is exclusive: expert E-1 stays empty (exercises 0-row segments)
    top_k_index = torch.randint(0, E - 1, (T, top_k), device=dev)
    top_k_weights = torch.softmax(torch.randn(T, top_k, device=dev), dim=-1)
    g_out = torch.randn(T, H, device=dev, dtype=dtype)

    def run(mode):
        os.environ["AXOLOTL_SONICMOE_NVFP4_BWD"] = mode
        leaves = tuple(t.clone().requires_grad_() for t in (hidden, A1, B1, A2, B2))
        h_i, A1_i, B1_i, A2_i, B2_i = leaves
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
        out.backward(g_out)
        return out.detach(), tuple(leaf.grad for leaf in leaves)

    print("running bf16 backward reference ...")
    out_ref, g_ref = run("bf16")
    print("running deepgemm fp8 backward (first call JIT-compiles) ...")
    out_fp8, g_fp8 = run("deepgemm")

    check("forward bitwise across backward engines", torch.equal(out_ref, out_fp8))
    check("dA2 bitwise (independent of dX engine)", torch.equal(g_ref[3], g_fp8[3]))
    check("dB2 bitwise (independent of dX engine)", torch.equal(g_ref[4], g_fp8[4]))
    report_norm("d_hidden fp8 vs bf16 backward", g_fp8[0], g_ref[0], tol=8e-2)
    report_norm("dA1 fp8 vs bf16 backward", g_fp8[1], g_ref[1], tol=8e-2)
    report_norm("dB1 fp8 vs bf16 backward", g_fp8[2], g_ref[2], tol=8e-2)

    # unit-level dX: fp8 grouped GEMM vs dequant grouped GEMM on the up proj
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import route_and_group

    x_grouped, expert_offsets, _, _ = route_and_group(
        hidden, top_k_index, top_k_weights, E
    )
    grad_h = torch.randn(x_grouped.shape[0], 2 * I, device=dev, dtype=dtype)
    dx_fp8 = grouped_fp8_dx(grad_h, w1, expert_offsets)
    w_dense = dequantize_engine_weight(w1).to(dtype)
    dx_ref = torch._grouped_mm(grad_h, w_dense, offs=expert_offsets[1:].to(torch.int32))
    report_norm(
        "unit dX: grouped_fp8_dx vs dequant grouped_mm", dx_fp8, dx_ref, tol=5e-2
    )

    def time_backward(mode, iters=10):
        os.environ["AXOLOTL_SONICMOE_NVFP4_BWD"] = mode
        for _ in range(3):
            run(mode)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            run(mode)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    t_ref = time_backward("bf16")
    t_fp8 = time_backward("deepgemm")
    print(
        f"[INFO] fwd+bwd wall: bf16 {t_ref:.2f} ms, deepgemm {t_fp8:.2f} ms "
        f"({t_ref / t_fp8:.2f}x)"
    )

    os.environ.pop("AXOLOTL_SONICMOE_NVFP4_BWD", None)
    finish()


if __name__ == "__main__":
    main()
