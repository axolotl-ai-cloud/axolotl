#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
ScatterMoE LoRA — MXFP4 forward + backward benchmark.

Runs three configurations on a representative DeepSeek-V4-style MoE shape
(E=128, K=2048, N=1024, top_k=8, batch×seq=4096) and reports tokens/s,
peak GPU memory, and effective HBM bandwidth for each:

  * **bf16 baseline**: full-precision bf16 experts, no MX.
  * **Strategy A**: torchao MXTensor experts, selective dequant to bf16.
  * **Strategy B**: torchao MXTensor experts, fused MX dequant in Triton.

Run from the repo root:

  python tests/integrations/kernels/scattermoe_lora/bench_mxfp4.py

A markdown table is printed to stdout and written to
``bench_mxfp4_results.md`` next to this script.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

import torch
from torchao.prototype.mx_formats.mx_tensor import MXTensor

from axolotl.integrations.kernels.libs.scattermoe_lora.mx_weights import (
    selective_mx_weights_fwd,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    flatten_sort_count,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (
    parallel_linear_lora,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import (
    get_active_experts,
    remap_expert_indices,
    selective_expert_weights,
    selective_lora_weights,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def gpu_name() -> str:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
        # First GPU line: "GPU 0: NAME (UUID: ...)"
        first = out.splitlines()[0]
        if ":" in first:
            after_colon = first.split(":", 1)[1].strip()
            return after_colon.split("(", 1)[0].strip()
        return first
    except Exception:
        return torch.cuda.get_device_name(0)


def gpu_hbm_bandwidth_gbps() -> float | None:
    """Rough peak HBM BW for utilization %, looked up by name. ``None`` if
    unknown — printed as N/A."""
    name = gpu_name().lower()
    # Approximate datasheet peaks (GB/s). Order matters — more-specific
    # patterns first so a "rtx pro 6000 blackwell" doesn't match
    # "rtx 6000 ada".
    table = [
        (("rtx", "6000", "blackwell"), 1792.0),
        (("rtx", "6000", "ada"), 960.0),
        (("rtx", "5090"), 1792.0),
        (("rtx", "4090"), 1008.0),
        (("h200",), 4800.0),
        (("h100",), 3350.0),
        (("a100",), 2039.0),
        (("a40",), 696.0),
        (("a6000",), 768.0),
        (("l40",), 864.0),
        (("l4",), 300.0),
        (("b200",), 8000.0),
        (("mi300x",), 5300.0),
    ]
    for keys, bw in table:
        if all(k in name for k in keys):
            return bw
    return None


@torch.no_grad()
def _setup_bf16(E, K, N, top_k, M, rank):
    torch.manual_seed(0)
    W = torch.randn(E, N, K, device=DEVICE, dtype=DTYPE) * (1.0 / K ** 0.5)
    W_kernel = W.transpose(2, 1).contiguous()  # [E, K, N]
    return W, W_kernel


def _setup_mx(W_natural):
    """Make a torchao MXFP4 ``MXTensor`` from the bf16 weight."""
    return MXTensor.to_mx(W_natural, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)


def _routing(M, E, top_k, seed=1):
    torch.manual_seed(seed)
    logits = torch.randn(M, E, device=DEVICE)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    sei, ssi, eo = flatten_sort_count(top_idx, E)
    return sei, ssi, eo, top_idx


def _lora(E, K, N, rank, seed=2):
    torch.manual_seed(seed)
    A = torch.randn(rank * E, K, device=DEVICE, dtype=DTYPE) * 0.01
    B = torch.randn(N, rank * E, device=DEVICE, dtype=DTYPE) * 0.01
    return A, B


class _MockExperts:
    def __init__(self, p):
        self.gate_up_proj = p


# ---------------------------------------------------------------------------
# Three benchmark runners — each takes a fresh `x` and returns (output, fn_grad)
# ---------------------------------------------------------------------------


def make_runner_bf16(W_kernel, lora_A, lora_B, sei, ssi, eo, top_k, scaling):
    def run(x):
        x = x.requires_grad_(True)
        A = lora_A.detach().clone().requires_grad_(True)
        B = lora_B.detach().clone().requires_grad_(True)
        out = parallel_linear_lora(
            x, W_kernel, top_k, sei, ssi, eo,
            lora_A=A, lora_B=B, scaling=scaling,
            use_fused_dX=True, use_fused_gather=True,
        )
        out.sum().backward()
        return out
    return run


def make_runner_strategy_a(mx, lora_A, lora_B, sei, ssi, eo, top_k, scaling, E):
    experts = _MockExperts(mx)
    def run(x):
        x = x.requires_grad_(True)
        A = lora_A.detach().clone().requires_grad_(True)
        B = lora_B.detach().clone().requires_grad_(True)
        active = get_active_experts(sei, E)
        remapped, compact_off = remap_expert_indices(sei, eo, active, E)
        W_compact = selective_expert_weights(
            experts, "gate_up_proj", active
        ).transpose(2, 1).contiguous()
        A_c, B_c = selective_lora_weights(A, B, active, E)
        out = parallel_linear_lora(
            x, W_compact, top_k, remapped, ssi, compact_off,
            lora_A=A_c, lora_B=B_c, scaling=scaling,
            use_fused_dX=True, use_fused_gather=True,
        )
        out.sum().backward()
        return out
    return run


def make_runner_strategy_b(mx, lora_A, lora_B, sei, ssi, eo, top_k, scaling, E):
    def run(x):
        x = x.requires_grad_(True)
        A = lora_A.detach().clone().requires_grad_(True)
        B = lora_B.detach().clone().requires_grad_(True)
        active = get_active_experts(sei, E)
        remapped, compact_off = remap_expert_indices(sei, eo, active, E)
        mx_active = selective_mx_weights_fwd(mx, active)
        A_c, B_c = selective_lora_weights(A, B, active, E)
        out = parallel_linear_lora(
            x, mx_active, top_k, remapped, ssi, compact_off,
            lora_A=A_c, lora_B=B_c, scaling=scaling,
        )
        out.sum().backward()
        return out
    return run


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


def bench(fn: Callable, x_template: torch.Tensor, warmup: int, iters: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(warmup):
        fn(x_template.detach().clone())
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x_template.detach().clone())
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return {
        "ms_per_iter": elapsed_ms / iters,
        "peak_mem_mb": peak_mem_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--E", type=int, default=128)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--M", type=int, default=4096, help="batch * seq")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    E, K, N, top_k, M, rank = args.E, args.K, args.N, args.top_k, args.M, args.rank

    print(f"GPU: {gpu_name()}")
    print(f"Shape: E={E}, K={K}, N={N}, top_k={top_k}, M={M}, rank={rank}")
    print(f"Iters: {args.warmup} warmup + {args.iters} timed")
    print()

    # Build dense bf16 weights, then MX-quantize the natural-storage [E, N, K] form
    W_natural, W_kernel = _setup_bf16(E, K, N, top_k, M, rank)
    mx = _setup_mx(W_natural)
    sei, ssi, eo, top_idx = _routing(M, E, top_k)
    lora_A, lora_B = _lora(E, K, N, rank)
    scaling = 0.5
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)

    # Estimate the bytes read per iter for HBM BW utilization:
    #   bf16 W:     E_active * K * N * 2
    #   MX  W:      E_active * (K*N/2 + K*N/32)
    # X is M*K*2 bytes. We ignore LoRA traffic (tiny relative to W).
    num_tokens = M * top_k
    e_active = int(get_active_experts(sei, E).numel())
    bytes_bf16 = e_active * K * N * 2 + M * K * 2
    bytes_mx = e_active * (K * N // 2 + K * N // 32) + M * K * 2
    peak_bw = gpu_hbm_bandwidth_gbps()

    runners = {
        "bf16 baseline": (
            make_runner_bf16(W_kernel, lora_A, lora_B, sei, ssi, eo, top_k, scaling),
            bytes_bf16,
        ),
        "Strategy A (selective dequant)": (
            make_runner_strategy_a(mx, lora_A, lora_B, sei, ssi, eo, top_k, scaling, E),
            bytes_bf16,  # post-dequant the kernel still reads bf16
        ),
        "Strategy B (fused MX)": (
            make_runner_strategy_b(mx, lora_A, lora_B, sei, ssi, eo, top_k, scaling, E),
            bytes_mx,
        ),
    }

    results = []
    for name, (fn, bytes_per_iter) in runners.items():
        r = bench(fn, x, args.warmup, args.iters)
        tps = num_tokens / (r["ms_per_iter"] / 1000.0)
        bw = (bytes_per_iter / 1e9) / (r["ms_per_iter"] / 1000.0)
        bw_pct = (bw / peak_bw * 100.0) if peak_bw else float("nan")
        results.append(
            dict(
                name=name,
                ms_per_iter=r["ms_per_iter"],
                tokens_per_s=tps,
                peak_mem_mb=r["peak_mem_mb"],
                hbm_gbps=bw,
                hbm_pct=bw_pct,
            )
        )

    lines = []
    lines.append(f"# ScatterMoE LoRA — MXFP4 benchmark")
    lines.append("")
    lines.append(f"- **GPU**: {gpu_name()}")
    lines.append(
        f"- **Shape**: E={E}, K={K}, N={N}, top_k={top_k}, M={M}, rank={rank} "
        f"(active experts = {e_active})"
    )
    lines.append(
        f"- **Iters**: {args.warmup} warmup + {args.iters} timed, fwd+bwd per iter"
    )
    if peak_bw:
        lines.append(f"- **HBM peak (datasheet)**: {peak_bw:.0f} GB/s")
    lines.append("")
    lines.append("| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        hbm_pct = f"{r['hbm_pct']:.1f}" if not math.isnan(r["hbm_pct"]) else "N/A"
        lines.append(
            f"| {r['name']} | {r['ms_per_iter']:.2f} | {r['tokens_per_s']:.0f} | "
            f"{r['peak_mem_mb']:.1f} | {r['hbm_gbps']:.1f} | {hbm_pct} |"
        )

    md = "\n".join(lines) + "\n"
    print(md)

    out_path = Path(__file__).resolve().parent / "bench_mxfp4_results.md"
    out_path.write_text(md)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
