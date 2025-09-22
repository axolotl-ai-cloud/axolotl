#!/usr/bin/env python
"""Reproduce TorchTitan CG GEMM timings for selected problem sizes."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

import torch

from axolotl.kernels.moe import (
    cg_grouped_gemm_forward,
    cg_grouped_gemm_forward_dynamic,
)


@dataclass
class Scenario:
    num_groups: int
    m: int
    n: int
    k: int


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(num_groups=4, m=8192, n=4096, k=7168),
    Scenario(num_groups=4, m=8192, n=7168, k=2048),
    Scenario(num_groups=8, m=4096, n=4096, k=7168),
    Scenario(num_groups=8, m=4096, n=7168, k=2048),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda"], help="Execution device")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Computation dtype")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="GROUP_SIZE_M expected by the kernel",
    )
    return parser.parse_args()


def pick_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def make_indices(num_groups: int, group_size: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(num_groups, device=device, dtype=torch.int32)
    return indices.repeat_interleave(group_size)


def timed_call(fn, *args, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def run_scenario(
    scenario: Scenario,
    *,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    group_size_m: int,
) -> dict:
    if scenario.m % scenario.num_groups != 0:
        raise ValueError(f"M ({scenario.m}) not divisible by groups ({scenario.num_groups})")
    group_size = scenario.m // scenario.num_groups
    if group_size % group_size_m != 0:
        raise ValueError(
            f"Group size {group_size} must be a multiple of GROUP_SIZE_M ({group_size_m}) for the Triton kernel"
        )

    inputs = torch.randn(scenario.m, scenario.k, device=device, dtype=dtype)
    weights = torch.randn(scenario.num_groups, scenario.n, scenario.k, device=device, dtype=dtype)
    indices = make_indices(scenario.num_groups, group_size, device)

    def persistent():
        return cg_grouped_gemm_forward(inputs, weights, indices, group_size_m)

    def baseline():
        return cg_grouped_gemm_forward_dynamic(inputs, weights, indices, group_size_m)

    persistent_ms = timed_call(persistent, warmup=warmup, iters=iters)
    baseline_ms = timed_call(baseline, warmup=warmup, iters=iters)

    return {
        "scenario": scenario,
        "persistent_ms": persistent_ms,
        "baseline_ms": baseline_ms,
        "speedup": baseline_ms / persistent_ms if persistent_ms > 0 else float("nan"),
    }


def main() -> None:  # pragma: no cover - utility script
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA device required for this benchmark")

    dtype = pick_dtype(args.dtype)
    device = torch.device(args.device)

    print(
        f"device={device} dtype={dtype} warmup={args.warmup} iters={args.iters} group_size={args.group_size}"
    )
    print(
        f"{'groups':>7} {'m':>7} {'n':>7} {'k':>7} {'persistent':>12} {'baseline':>12} {'speedup':>8}"
    )
    for result in run_all(
        SCENARIOS,
        dtype=dtype,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        group_size_m=args.group_size,
    ):
        scen = result["scenario"]
        print(
            f"{scen.num_groups:>7} {scen.m:>7} {scen.n:>7} {scen.k:>7}"
            f" {result['persistent_ms']:>11.3f} ms {result['baseline_ms']:>11.3f} ms {result['speedup']:>7.2f}x"
        )


def run_all(
    scenarios: Iterable[Scenario],
    *,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
    group_size_m: int,
) -> Iterable[dict]:
    for scenario in scenarios:
        yield run_scenario(
            scenario,
            dtype=dtype,
            device=device,
            warmup=warmup,
            iters=iters,
            group_size_m=group_size_m,
        )


if __name__ == "__main__":
    main()
