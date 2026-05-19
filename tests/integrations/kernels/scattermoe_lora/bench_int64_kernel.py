#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
ScatterMoE — INT64_INDICES vs INT32 dense scatter2scatter benchmark.

Times the dense ``kernels.ops.scatter2scatter`` at three representative
shapes and reports ms/iter for both ``INT64_INDICES=False`` (int32 fast
path) and ``INT64_INDICES=True`` (int64 safe path). The third shape is
the previously-failing seq=512K / 16-shard config; at that scale the
int32 path is incorrect (silent overflow corruption) so the int32 row
is gated by the ``_SCATTER2SCATTER_INT32_LIMIT`` and reported as the
chunked workaround's wall-clock instead (also for comparison against
the chunking baseline that PR #3667 shipped).

Run from the repo root:

  python tests/integrations/kernels/scattermoe_lora/bench_int64_kernel.py

A markdown summary is printed to stdout and written to
``bench_int64_kernel_results.md`` next to this script.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
from pathlib import Path
from typing import Callable

import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops import (
    scatter2scatter,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    _SCATTER2SCATTER_INT32_LIMIT,
    _scatter2scatter_int32_safe,
    flatten_sort_count,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def gpu_name() -> str:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
        first = out.splitlines()[0]
        if ":" in first:
            after_colon = first.split(":", 1)[1].strip()
            return after_colon.split("(", 1)[0].strip()
        return first
    except Exception:
        return torch.cuda.get_device_name(0)


def _time_ms(fn: Callable[[], torch.Tensor], iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _build_inputs(*, T: int, hidden: int, top_k: int, n: int, num_experts: int, seed: int):
    torch.manual_seed(seed)
    x = torch.randn(T, hidden, device=DEVICE, dtype=DTYPE)
    W = torch.randn(num_experts, hidden, n, device=DEVICE, dtype=DTYPE) * 0.02
    logits = torch.randn(T, num_experts, device=DEVICE)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    sei, ssi, _ = flatten_sort_count(top_idx, num_experts)
    return x, W, sei, ssi


def _run_shape(name: str, *, T: int, hidden: int, top_k: int, n: int, num_experts: int):
    x, W, sei, ssi = _build_inputs(
        T=T, hidden=hidden, top_k=top_k, n=n, num_experts=num_experts, seed=42
    )
    L_scattered = sei.size(0)
    out_elements = L_scattered * n
    overflow = out_elements >= _SCATTER2SCATTER_INT32_LIMIT
    auto_int64 = overflow  # the wrapper's auto-dispatch verdict

    def call(int64_indices: bool):
        return scatter2scatter(
            X=x,
            W=W,
            sorted_expert_idxs=sei,
            sorted_scattered_idxs=ssi,
            k=top_k,
            x_grouped=False,
            y_grouped=True,
            int64_indices=int64_indices,
        )

    # Warm both Triton variants (separate JITs per constexpr).
    _ = call(False) if not overflow else None
    if overflow:
        # int32 path is unsafe at overflow shapes (silent corruption); skip.
        ms_i32 = None
    else:
        ms_i32 = _time_ms(lambda: call(False))
    ms_i64 = _time_ms(lambda: call(True))

    # Chunked workaround timing (only meaningful at overflow shapes; below
    # threshold the wrapper short-circuits to the same kernel call).
    if overflow:
        ms_chunk = _time_ms(
            lambda: _scatter2scatter_int32_safe(
                X=x,
                W=W,
                sorted_expert_idxs=sei,
                sorted_scattered_idxs=ssi,
                k=top_k,
                x_grouped=False,
                y_grouped=True,
                int64_indices=False,
            )
        )
    else:
        ms_chunk = None

    return {
        "name": name,
        "T": T,
        "hidden": hidden,
        "top_k": top_k,
        "n": n,
        "num_experts": num_experts,
        "L_scattered": L_scattered,
        "out_elements": out_elements,
        "overflow": overflow,
        "auto_int64": auto_int64,
        "ms_i32": ms_i32,
        "ms_i64": ms_i64,
        "ms_chunk": ms_chunk,
    }


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.3f}"


def _markdown(rows, gpu_label: str) -> str:
    lines = []
    lines.append("# scatter2scatter INT64_INDICES bench")
    lines.append("")
    lines.append(f"GPU: **{gpu_label}**")
    lines.append("")
    lines.append("Median of 10 iters, 3 warmup. `top_k=8`, dtype=bf16, 128 experts.")
    lines.append("")
    lines.append("`auto_int64` is the wrapper's auto-dispatch verdict from "
                 "`_needs_int64_indices`. At overflow shapes the int32 path "
                 "is silently incorrect, so the int32 column shows the "
                 "chunked workaround's wall-clock from PR #3667 as the "
                 "apples-to-apples baseline.")
    lines.append("")
    lines.append("| Shape | T | L_scattered | out elems | auto_int64 | int32 ms | int64 ms | chunked ms | int64 vs fast (%) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        # Penalty: int64 vs the fastest legal int32 path
        # (raw int32 when below overflow, chunked workaround otherwise).
        baseline = r["ms_i32"] if r["ms_i32"] is not None else r["ms_chunk"]
        if baseline is not None and r["ms_i64"] is not None:
            pen = 100.0 * (r["ms_i64"] - baseline) / baseline
            pen_s = f"{pen:+.1f}"
        else:
            pen_s = "—"
        lines.append(
            f"| {r['name']} | {r['T']} | {r['L_scattered']} | "
            f"{r['out_elements']:.2e} | {str(r['auto_int64'])} | "
            f"{_fmt(r['ms_i32'])} | {_fmt(r['ms_i64'])} | "
            f"{_fmt(r['ms_chunk'])} | {pen_s} |"
        )
    lines.append("")
    lines.append("Acceptance: ≤5% regression on the int32 fast path at "
                 "small/medium shapes, ≤25% regression on the int64 path "
                 "vs the chunked workaround at overflow shapes.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None,
                        help="Output markdown path (default: alongside script)")
    args = parser.parse_args()

    # Three representative shapes per the task spec.
    shapes = [
        # name              T (tokens before top_k expansion), hidden, top_k, N (out), num_experts
        dict(name="small",  T=8_192,   hidden=2048, top_k=8, n=2048, num_experts=128),
        dict(name="medium", T=128_000, hidden=2048, top_k=8, n=2048, num_experts=128),
        # Overflow shape: 524288 / 16 shards = 32768 tokens, top_k=8 -> L=262144,
        # N=16384 (= 2*intermediate at the bench config) -> 2**32 elements.
        dict(name="overflow_524k_s16",
             T=32_768, hidden=2048, top_k=8, n=16_384, num_experts=128),
    ]

    rows = []
    for s in shapes:
        print(f"running {s['name']} ...", flush=True)
        rows.append(_run_shape(**s))
        torch.cuda.empty_cache()

    label = gpu_name()
    md = _markdown(rows, label)
    print(md)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).with_name("bench_int64_kernel_results.md")
    out_path.write_text(md)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
