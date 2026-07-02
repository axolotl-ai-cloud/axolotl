"""Parity gates for the Triton NVFP4 codecs vs torchao / the pure-torch reference.

- dequant: triton vs torchao ``NVFP4Tensor.dequantize()`` on random expert
  weights with per-expert pts (expect bit-equal bf16 up to 1-ulp fp32-assoc
  differences; we assert exact equality and report any mismatch count).
- quantize: triton vs ``quantize_nvfp4_ref`` (expect BIT-exact packed codes and
  scales: same amax/6 clamp, same RN e4m3 cast, same ties-down E2M1 bucketing).

Also times both against their torch counterparts.
"""

import statistics

from _common import check, finish


def main():
    import torch
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        quantize_nvfp4_ref,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.triton_nvfp4 import (
        dequant_nvfp4_triton,
        quantize_nvfp4_triton,
        triton_available,
    )

    check("triton available", triton_available())
    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)

    E, N, K = 128, 1536, 2048
    dense = torch.randn(E, N, K, device=dev, dtype=dtype) * 0.02
    pts = torch.rand(E, device=dev) * 3e-4 + 5e-5
    qs, ss = [], []
    for e in range(E):
        q, s, _ = quantize_nvfp4_ref(dense[e], pts[e])
        qs.append(q)
        ss.append(s)
    w = NVFP4Tensor(
        torch.stack(qs), torch.stack(ss), 16, dtype, per_tensor_scale=pts.view(-1, 1, 1)
    )

    ref = w.dequantize()
    tri = dequant_nvfp4_triton(w.qdata, w.scale, w.per_tensor_scale, dtype)
    mism = int((ref != tri).sum())
    print(f"[info] dequant mismatches: {mism}/{ref.numel()}")
    check("dequant parity (exact)", mism == 0)

    # no-pts variant (activation-style)
    x = torch.randn(4096, K, device=dev, dtype=dtype)
    q_ref, s_ref, _ = quantize_nvfp4_ref(x)
    q_tri, s_tri = quantize_nvfp4_triton(x)
    qm = int((q_ref != q_tri).sum())
    sm = int((s_ref.view(torch.uint8) != s_tri.view(torch.uint8)).sum())
    print(
        f"[info] quant code mismatches: {qm}/{q_ref.numel()}; scale: {sm}/{s_ref.numel()}"
    )
    check("quantize codes bit-exact", qm == 0)
    check("quantize scales bit-exact", sm == 0)

    # odd width (down-proj K=768, needs masked tail)
    x2 = torch.randn(1000, 768, device=dev, dtype=dtype)
    q_ref2, s_ref2, _ = quantize_nvfp4_ref(x2)
    q_tri2, s_tri2 = quantize_nvfp4_triton(x2)
    check(
        "quantize bit-exact at K=768",
        int((q_ref2 != q_tri2).sum()) == 0
        and int((s_ref2.view(torch.uint8) != s_tri2.view(torch.uint8)).sum()) == 0,
    )

    def timed(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)

    t_dq_ao = timed(w.dequantize)
    t_dq_tri = timed(
        lambda: dequant_nvfp4_triton(w.qdata, w.scale, w.per_tensor_scale, dtype)
    )
    t_q_ref = timed(lambda: quantize_nvfp4_ref(x))
    t_q_tri = timed(lambda: quantize_nvfp4_triton(x))
    print(
        f"[info] dequant [128,1536,2048]: torchao {t_dq_ao:.2f} ms  triton {t_dq_tri:.2f} ms"
    )
    print(
        f"[info] quantize [4096,2048]: torch-ref {t_q_ref:.2f} ms  triton {t_q_tri:.2f} ms"
    )

    finish()


if __name__ == "__main__":
    main()
