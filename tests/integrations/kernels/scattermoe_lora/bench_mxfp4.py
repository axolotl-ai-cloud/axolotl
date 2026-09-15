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
import subprocess
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
    W = torch.randn(E, N, K, device=DEVICE, dtype=DTYPE) * (1.0 / K**0.5)
    W_kernel = W.transpose(2, 1).contiguous()  # [E, K, N]
    return W, W_kernel


def _setup_mx(W_natural, chunk: int = 8):
    """Make a torchao MXFP4 ``MXTensor`` from the bf16 weight.

    ``MXTensor.to_mx`` materializes an fp32 working tensor internally; for
    large [E, N, K] weights this transient can spike setup-time GPU memory
    well beyond the final quantized footprint. To keep the bench runnable
    on a shared GPU, quantize ``chunk`` experts at a time and stitch the
    qdata/scale shards into a single MXTensor.
    """
    if W_natural.shape[0] <= chunk:
        return MXTensor.to_mx(
            W_natural, elem_dtype=torch.float4_e2m1fn_x2, block_size=32
        )
    qdata_parts = []
    scale_parts = []
    template = None
    for i in range(0, W_natural.shape[0], chunk):
        piece = W_natural[i : i + chunk].contiguous()
        mx_chunk = MXTensor.to_mx(
            piece, elem_dtype=torch.float4_e2m1fn_x2, block_size=32
        )
        qdata_parts.append(mx_chunk.qdata)
        scale_parts.append(mx_chunk.scale)
        if template is None:
            template = mx_chunk
    qdata = torch.cat(qdata_parts, dim=0)
    scale = torch.cat(scale_parts, dim=0)
    assert template is not None  # set on the first loop iter (loop body runs ≥ once)
    return MXTensor(
        qdata,
        scale,
        template.elem_dtype,
        template.block_size,
        template.orig_dtype,
        template.kernel_preference,
        template.act_quant_kwargs,
        template.is_swizzled_scales,
    )


def _routing(M, E, top_k, seed=1, mode="dense"):
    """Generate token→expert routing.

    ``mode="dense"`` uses per-token random logits; for moderate E and top_k
    this leaves nearly every expert active and exercises the kernels' full-load
    case. ``mode="sparse"`` injects a strong shared bias so the same handful of
    experts dominates the topk across all tokens — modelling realistic MoE
    routing where only a small fraction of experts is active per step.
    ``mode="balanced"`` models a load-balance-regularized router (aux-loss /
    z-loss trained): per-token logits = N(0, 1) noise + small per-expert
    bias N(0, 0.5). At large M this yields approximately balanced expert
    usage; at small M only a fraction of experts gets hit and which experts
    are active varies with seed/M — i.e. the seqlen → active-expert-count
    curve that drives the A-vs-B crossover.
    """
    torch.manual_seed(seed)
    if mode == "dense":
        logits = torch.randn(M, E, device=DEVICE)
    elif mode == "sparse":
        shared = torch.randn(E, device=DEVICE) * 5.0
        noise = torch.randn(M, E, device=DEVICE) * 0.1
        logits = shared.unsqueeze(0) + noise
    elif mode == "balanced":
        bias = torch.randn(E, device=DEVICE) * 0.5
        noise = torch.randn(M, E, device=DEVICE)
        logits = noise + bias.unsqueeze(0)
    else:
        raise ValueError(f"unknown routing mode: {mode}")
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


# Runners reuse the same leaf tensors across timed iters (x, lora A/B) and
# zero ``.grad`` to None at the top of each call. The previous per-iter
# ``.clone()`` + ``requires_grad_(True)`` was setup cost — not kernel cost —
# and biased the timing especially on small shapes. Gradient accumulation
# is avoided by setting ``.grad = None`` (faster than ``.zero_()``), so the
# autograd graph each iter is fresh but the leaf buffers are not reallocated.


def make_runner_bf16(W_kernel, lora_A, lora_B, sei, ssi, eo, top_k, scaling):
    A = lora_A.detach().clone().requires_grad_(True)
    B = lora_B.detach().clone().requires_grad_(True)

    def run(x):
        x.grad = None
        A.grad = None
        B.grad = None
        out = parallel_linear_lora(
            x,
            W_kernel,
            top_k,
            sei,
            ssi,
            eo,
            lora_A=A,
            lora_B=B,
            scaling=scaling,
            use_fused_dX=True,
            use_fused_gather=True,
        )
        out.sum().backward()
        return out

    return run


def make_runner_strategy_a(mx, lora_A, lora_B, sei, ssi, eo, top_k, scaling, E):
    experts = _MockExperts(mx)
    A = lora_A.detach().clone().requires_grad_(True)
    B = lora_B.detach().clone().requires_grad_(True)

    def run(x):
        x.grad = None
        A.grad = None
        B.grad = None
        active = get_active_experts(sei, E)
        remapped, compact_off = remap_expert_indices(sei, eo, active, E)
        W_compact = (
            selective_expert_weights(experts, "gate_up_proj", active)
            .transpose(2, 1)
            .contiguous()
        )
        A_c, B_c = selective_lora_weights(A, B, active, E)
        out = parallel_linear_lora(
            x,
            W_compact,
            top_k,
            remapped,
            ssi,
            compact_off,
            lora_A=A_c,
            lora_B=B_c,
            scaling=scaling,
            use_fused_dX=True,
            use_fused_gather=True,
        )
        out.sum().backward()
        return out

    return run


def make_runner_strategy_b(mx, lora_A, lora_B, sei, ssi, eo, top_k, scaling, E):
    A = lora_A.detach().clone().requires_grad_(True)
    B = lora_B.detach().clone().requires_grad_(True)

    def run(x):
        x.grad = None
        A.grad = None
        B.grad = None
        active = get_active_experts(sei, E)
        remapped, compact_off = remap_expert_indices(sei, eo, active, E)
        mx_active = selective_mx_weights_fwd(mx, active)
        A_c, B_c = selective_lora_weights(A, B, active, E)
        out = parallel_linear_lora(
            x,
            mx_active,
            top_k,
            remapped,
            ssi,
            compact_off,
            lora_A=A_c,
            lora_B=B_c,
            scaling=scaling,
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
    # Allocate the input leaf tensor once outside the timed window. Runners
    # reset ``.grad = None`` per iter; the underlying buffer is reused.
    x = x_template.detach().clone().requires_grad_(True)
    try:
        # Warmup
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(x)
        end.record()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"ms_per_iter": float("nan"), "peak_mem_mb": float("nan"), "oom": True}
    elapsed_ms = start.elapsed_time(end)
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return {
        "ms_per_iter": elapsed_ms / iters,
        "peak_mem_mb": peak_mem_mb,
        "oom": False,
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
    parser.add_argument(
        "--routing-mode",
        choices=("dense", "sparse", "balanced"),
        default="dense",
        help=(
            "dense: per-token random logits (~all experts active). "
            "sparse: shared bias + small per-token noise so the same ~top_k "
            "experts dominate routing across all tokens. "
            "balanced: per-token N(0,1) noise + small N(0,0.5) per-expert bias "
            "— mimics a load-balance-regularized router; active-expert count "
            "grows with M."
        ),
    )
    parser.add_argument(
        "--M-sweep",
        dest="M_sweep",
        default=None,
        help=(
            "Comma-separated list of M values, e.g. '256,1024,4096,16384'. "
            "When set, --M is ignored; the bench runs once per M with the "
            "selected routing mode and emits a single combined section."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append the new table to bench_mxfp4_results.md instead of overwriting it.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device, e.g. 'cuda', 'cuda:0', 'cuda:1'.",
    )
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device
    torch.cuda.set_device(torch.device(DEVICE))

    E, K, N, top_k, rank = args.E, args.K, args.N, args.top_k, args.rank

    if args.M_sweep:
        M_values = [int(s.strip()) for s in args.M_sweep.split(",") if s.strip()]
    else:
        M_values = [args.M]

    print(f"GPU: {gpu_name()}")
    print(
        f"Shape: E={E}, K={K}, N={N}, top_k={top_k}, rank={rank}, "
        f"M={M_values if args.M_sweep else M_values[0]}"
    )
    print(f"Iters: {args.warmup} warmup + {args.iters} timed")
    print()

    # Build dense bf16 weights + MX-quantize once — weights are independent of M.
    W_natural, W_kernel = _setup_bf16(E, K, N, top_k, M_values[0], rank)
    mx = _setup_mx(W_natural)
    # W_natural is only used to build the MX tensor; W_kernel feeds bf16 paths.
    # Free it eagerly so the dequant transient fits on memory-constrained GPUs.
    del W_natural
    torch.cuda.empty_cache()
    lora_A, lora_B = _lora(E, K, N, rank)
    scaling = 0.5
    peak_bw = gpu_hbm_bandwidth_gbps()

    per_M = []  # list of (M, e_active, results)
    for M in M_values:
        sei, ssi, eo, _top_idx = _routing(M, E, top_k, mode=args.routing_mode)
        x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)

        # Estimate bytes read per iter for HBM BW utilization:
        #   bf16 W:     E_active * K * N * 2
        #   MX  W:      E_active * (K*N/2 + K*N/32)
        # X is M*K*2 bytes. We ignore LoRA traffic (tiny relative to W).
        num_tokens = M * top_k
        e_active = int(get_active_experts(sei, E).numel())
        bytes_bf16 = e_active * K * N * 2 + M * K * 2
        bytes_mx = e_active * (K * N // 2 + K * N // 32) + M * K * 2

        runners = {
            "bf16 baseline": (
                make_runner_bf16(
                    W_kernel, lora_A, lora_B, sei, ssi, eo, top_k, scaling
                ),
                bytes_bf16,
            ),
            "Strategy A (selective dequant)": (
                make_runner_strategy_a(
                    mx, lora_A, lora_B, sei, ssi, eo, top_k, scaling, E
                ),
                bytes_bf16,  # post-dequant the kernel still reads bf16
            ),
            "Strategy B (fused MX)": (
                make_runner_strategy_b(
                    mx, lora_A, lora_B, sei, ssi, eo, top_k, scaling, E
                ),
                bytes_mx,
            ),
        }

        results = []
        for name, (fn, bytes_per_iter) in runners.items():
            r = bench(fn, x, args.warmup, args.iters)
            if r.get("oom"):
                tps = float("nan")
                bw = float("nan")
                bw_pct = float("nan")
            else:
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
                    oom=r.get("oom", False),
                )
            )
        per_M.append((M, e_active, results))

    section_lines = []
    if args.M_sweep:
        section_lines.append(
            f"## Routing mode: {args.routing_mode} — M sweep — {gpu_name()}"
        )
        section_lines.append("")
        section_lines.append(f"- **GPU**: {gpu_name()}")
        section_lines.append(
            f"- **Base shape**: E={E}, K={K}, N={N}, top_k={top_k}, rank={rank}"
        )
        section_lines.append(f"- **M values**: {', '.join(str(m) for m in M_values)}")
        section_lines.append(
            f"- **Iters**: {args.warmup} warmup + {args.iters} timed, fwd+bwd per iter"
        )
        if peak_bw:
            section_lines.append(f"- **HBM peak (datasheet)**: {peak_bw:.0f} GB/s")
        section_lines.append("")
        section_lines.append("### Summary (ms/iter, fwd+bwd)")
        section_lines.append("")
        section_lines.append(
            "| M | active / E | bf16 ms | Strategy A ms | Strategy B ms | winner (A vs B) |"
        )
        section_lines.append("| ---: | ---: | ---: | ---: | ---: | :---: |")

        def _fmt_ms(r):
            return "OOM" if r["oom"] else f"{r['ms_per_iter']:.2f}"

        for M, e_active, results in per_M:
            by_name = {r["name"]: r for r in results}
            a, b, bf = (
                by_name["Strategy A (selective dequant)"],
                by_name["Strategy B (fused MX)"],
                by_name["bf16 baseline"],
            )
            if a["oom"] and b["oom"]:
                winner = "—"
            elif a["oom"]:
                winner = "B"
            elif b["oom"]:
                winner = "A"
            else:
                winner = "A" if a["ms_per_iter"] < b["ms_per_iter"] else "B"
            section_lines.append(
                f"| {M} | {e_active}/{E} ({e_active / E:.2f}) | "
                f"{_fmt_ms(bf)} | {_fmt_ms(a)} | {_fmt_ms(b)} | {winner} |"
            )
        section_lines.append("")
        for M, e_active, results in per_M:
            section_lines.append(
                f"### M={M} (active experts = {e_active} / {E}, "
                f"num_active/E = {e_active / E:.3f})"
            )
            section_lines.append("")
            section_lines.append(
                "| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |"
            )
            section_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for r in results:
                if r["oom"]:
                    section_lines.append(
                        f"| {r['name']} | OOM | OOM | OOM | OOM | OOM |"
                    )
                    continue
                hbm_pct = (
                    f"{r['hbm_pct']:.1f}" if not math.isnan(r["hbm_pct"]) else "N/A"
                )
                section_lines.append(
                    f"| {r['name']} | {r['ms_per_iter']:.2f} | "
                    f"{r['tokens_per_s']:.0f} | {r['peak_mem_mb']:.1f} | "
                    f"{r['hbm_gbps']:.1f} | {hbm_pct} |"
                )
            section_lines.append("")
    else:
        M, e_active, results = per_M[0]
        section_lines.append(f"## Routing mode: {args.routing_mode} — {gpu_name()}")
        section_lines.append("")
        section_lines.append(f"- **GPU**: {gpu_name()}")
        section_lines.append(
            f"- **Shape**: E={E}, K={K}, N={N}, top_k={top_k}, M={M}, rank={rank} "
            f"(active experts = {e_active})"
        )
        section_lines.append(
            f"- **Iters**: {args.warmup} warmup + {args.iters} timed, fwd+bwd per iter"
        )
        if peak_bw:
            section_lines.append(f"- **HBM peak (datasheet)**: {peak_bw:.0f} GB/s")
        section_lines.append("")
        section_lines.append(
            "| Config | ms/iter | tokens/s | peak mem (MB) | HBM GB/s | HBM % |"
        )
        section_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for r in results:
            hbm_pct = f"{r['hbm_pct']:.1f}" if not math.isnan(r["hbm_pct"]) else "N/A"
            section_lines.append(
                f"| {r['name']} | {r['ms_per_iter']:.2f} | {r['tokens_per_s']:.0f} | "
                f"{r['peak_mem_mb']:.1f} | {r['hbm_gbps']:.1f} | {hbm_pct} |"
            )
    section_md = "\n".join(section_lines).rstrip() + "\n"

    out_path = Path(__file__).resolve().parent / "bench_mxfp4_results.md"
    if args.append and out_path.exists():
        existing = out_path.read_text().rstrip() + "\n\n"
        md = existing + section_md
    else:
        md = "# ScatterMoE LoRA — MXFP4 benchmark\n\n" + section_md

    print(section_md)
    out_path.write_text(md)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
