"""Phase-level timing of the sonicmoe NVFP4 grouped MoE-LoRA path at training shapes.

Times the full layer forward and forward+backward for the fp4_cute and dequant
backends, plus the individual forward phases (route, activation quant, SFA build,
engine GEMM, LoRA delta, combine), at Qwen3-30B-A3B shapes. CUDA-event medians.

Env: AXOLOTL_BENCH_TOKENS (default "2048,8192"), AXOLOTL_BENCH_ITERS (default 20).
"""

import os
import statistics

from _common import require_sm100

TOKENS = [
    int(t) for t in os.environ.get("AXOLOTL_BENCH_TOKENS", "2048,8192").split(",")
]
ITERS = int(os.environ.get("AXOLOTL_BENCH_ITERS", "20"))

E, TOP_K, H, I, R = 128, 8, 2048, 768, 8  # noqa: E741


def main():
    import torch

    require_sm100()

    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.integrations.kernels.libs.sonicmoe.fp4_cute_ops import (
        _get_engine,
        quantize_grouped_rows,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
        _lora_delta_per_group,
        combine_expert_outputs,
        grouped_moe_reference_forward,
        route_and_group,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        quantize_nvfp4_ref,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.sf_layout import build_varlen_sfa

    dev, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)

    def make_weight(dim1, dim2, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        dense = torch.randn(E, dim1, dim2, generator=g, device=dev, dtype=dtype) * 0.02
        pts = torch.full((E,), 2e-4, device=dev)
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

    w1 = make_weight(2 * I, H, 10)
    w2 = make_weight(H, I, 11)
    A1 = torch.randn(R * E, H, device=dev, dtype=dtype) * H**-0.5
    B1 = torch.randn(2 * I, R * E, device=dev, dtype=dtype) * 0.02
    A2 = torch.randn(R * E, I, device=dev, dtype=dtype) * I**-0.5
    B2 = torch.randn(H, R * E, device=dev, dtype=dtype) * 0.02

    def timed(fn, iters=ITERS, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        return statistics.median(times)

    def bench_tokens(T):
        hidden = torch.randn(T, H, device=dev, dtype=dtype)
        router = torch.randn(T, E, device=dev)
        vals, idx = router.topk(TOP_K, dim=-1)
        wts = torch.softmax(vals, dim=-1)
        g_out = torch.randn(T, H, device=dev, dtype=dtype)

        def full(backend, backward):
            h = hidden.clone().requires_grad_()
            a1 = A1.clone().requires_grad_()
            b1 = B1.clone().requires_grad_()
            a2 = A2.clone().requires_grad_()
            b2 = B2.clone().requires_grad_()
            out = grouped_moe_reference_forward(
                h,
                idx,
                wts,
                w1,
                None,
                w2,
                None,
                (a1, b1),
                (a2, b2),
                E,
                act="silu",
                backend=backend,
                concat=True,
                scaling1=0.5,
                scaling2=0.25,
            )
            if backward:
                (out.float() * g_out.float()).sum().backward()

        print(f"\n=== T={T} tokens ({T * TOP_K} grouped rows) ===")
        for backend in ("fp4_cute", "dequant"):
            f = timed(lambda b=backend: full(b, False))
            fb = timed(lambda b=backend: full(b, True))
            print(
                f"{backend:9s}: fwd {f:8.2f} ms   fwd+bwd {fb:8.2f} ms   bwd {fb - f:8.2f} ms"
            )

        # forward phase breakdown (fp4_cute)
        xg, offs, gidx, wg = route_and_group(hidden, idx, wts, E)
        cu = torch.cat([offs.new_zeros(1), offs[1:]]).to(torch.int32)
        engine = _get_engine(w1).engine
        a_q, sfa = quantize_grouped_rows(xg, cu)
        phases = {
            "route_and_group": lambda: route_and_group(hidden, idx, wts, E),
            "act quant (up K=2048)": lambda: quantize_nvfp4_ref(xg),
            "sfa build": lambda: build_varlen_sfa(quantize_nvfp4_ref(xg)[1], cu),
            "engine gemm (up)": lambda: engine.forward(a_q, sfa, cu),
            "lora delta (up)": lambda: _lora_delta_per_group(
                xg, offs, A1, B1, 0.5, E, 2 * I, H
            ),
            "combine": lambda: combine_expert_outputs(
                torch.randn(T * TOP_K, H, device=dev, dtype=dtype), gidx, wg, T
            ),
        }
        for name, fn in phases.items():
            print(f"  {name:24s}: {timed(fn):8.3f} ms")

    for T in TOKENS:
        bench_tokens(T)


if __name__ == "__main__":
    main()
