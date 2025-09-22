#!/usr/bin/env python
# mypy: ignore-errors
"""Sweep a set of DeepSeek V3 MoE benchmark configurations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmarks.deepseek_v3_moe import (
    DTYPE_MAP,
    benchmark_deepseek_v3,
)

DEFAULT_SWEEP = [
    {
        "batch": 4,
        "seq_len": 1024,
        "hidden_size": 2048,
        "moe_intermediate_size": 4096,
        "n_experts": 64,
        "top_k": 4,
        "groups": 4,
    },
    {
        "batch": 8,
        "seq_len": 2048,
        "hidden_size": 2048,
        "moe_intermediate_size": 4096,
        "n_experts": 64,
        "top_k": 4,
        "groups": 4,
    },
    {
        "batch": 8,
        "seq_len": 2048,
        "hidden_size": 4096,
        "moe_intermediate_size": 8192,
        "n_experts": 128,
        "top_k": 8,
        "groups": 8,
    },
    {
        "batch": 8,
        "seq_len": 2048,
        "hidden_size": 4096,
        "moe_intermediate_size": 8192,
        "n_experts": 256,
        "top_k": 8,
        "groups": 8,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        default="bf16",
        help="Computation dtype for all benchmarks",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=15, help="Benchmark iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="GROUP_SIZE_M used by the Triton kernel",
    )
    parser.add_argument(
        "--uniform-routing",
        action="store_true",
        help="Force uniform routing for every configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to store results",
    )
    return parser.parse_args()


def make_namespace(base: dict, args: argparse.Namespace) -> SimpleNamespace:
    combined = dict(base)
    combined.update(
        {
            "dtype": args.dtype,
            "device": args.device,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "group_size": args.group_size,
            "uniform_routing": args.uniform_routing,
        }
    )
    return SimpleNamespace(**combined)


def main() -> None:  # pragma: no cover - utility script
    args = parse_args()

    header = (
        "batch",
        "seq_len",
        "hidden_size",
        "moe_intermediate",
        "n_experts",
        "top_k",
        "baseline_ms",
        "patched_ms",
        "speedup",
        "min_tokens",
        "max_tokens",
        "max_diff",
    )
    rows = []

    print(
        f"Running sweep on device={args.device} dtype={args.dtype} uniform_routing={args.uniform_routing}"
    )
    print(
        f"{'batch':>5} {'seq':>5} {'hidden':>7} {'experts':>7} {'topk':>4} {'baseline':>12} {'patched':>12} {'speedup':>8}"
    )

    for cfg in DEFAULT_SWEEP:
        ns = make_namespace(cfg, args)
        result = benchmark_deepseek_v3(ns)
        rows.append(
            (
                cfg["batch"],
                cfg["seq_len"],
                cfg["hidden_size"],
                cfg["moe_intermediate_size"],
                cfg["n_experts"],
                cfg["top_k"],
                result["baseline_ms"],
                result["patched_ms"],
                result["speedup"],
                result["min_tokens"],
                result["max_tokens"],
                result["max_diff"],
            )
        )
        print(
            f"{cfg['batch']:>5} {cfg['seq_len']:>5} {cfg['hidden_size']:>7} {cfg['n_experts']:>7} {cfg['top_k']:>4}"
            f" {result['baseline_ms']:>11.3f} ms {result['patched_ms']:>11.3f} ms {result['speedup']:>7.2f}x"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
