#!/usr/bin/env python
# mypy: ignore-errors
"""Sweep a set of DeepSeek V3 MoE benchmark configurations."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

CURRENT_DIR = Path(__file__).resolve().parent
for candidate in [CURRENT_DIR, *CURRENT_DIR.parents]:
    repo_root = candidate / "axolotl"
    if repo_root.exists():
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        break
else:  # pragma: no cover - defensive guard
    raise SystemExit("Unable to locate axolotl repository root for imports")

from scripts.benchmarks.deepseek_v3_moe import (
    ACCURACY_TOLERANCE,
    DTYPE_MAP,
    benchmark_deepseek_v3,
)


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
        "--batches",
        default="4,8",
        help="Comma separated list of batch sizes",
    )
    parser.add_argument(
        "--seq-lens",
        default="1024,2048",
        help="Comma separated list of sequence lengths",
    )
    parser.add_argument(
        "--hidden-sizes",
        default="2048,4096",
        help="Comma separated list of hidden sizes",
    )
    parser.add_argument(
        "--moe-intermediates",
        default="4096,8192",
        help="Comma separated list of MoE intermediate sizes",
    )
    parser.add_argument(
        "--n-experts-list",
        default="64,128,256",
        help="Comma separated list of expert counts",
    )
    parser.add_argument(
        "--top-ks",
        default="4,8",
        help="Comma separated list of top-k values",
    )
    parser.add_argument(
        "--groups-list",
        default="4,8",
        help="Comma separated list of router group counts",
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

    def _parse_list(text: str) -> list[int]:
        return [int(item.strip()) for item in text.split(",") if item.strip()]

    batch_values = _parse_list(args.batches)
    seq_values = _parse_list(args.seq_lens)
    hidden_values = _parse_list(args.hidden_sizes)
    moe_values = _parse_list(args.moe_intermediates)
    expert_values = _parse_list(args.n_experts_list)
    topk_values = _parse_list(args.top_ks)
    group_values = _parse_list(args.groups_list)

    grid = []
    for batch, seq_len, hidden, moe, n_experts, top_k, groups in itertools.product(
        batch_values,
        seq_values,
        hidden_values,
        moe_values,
        expert_values,
        topk_values,
        group_values,
    ):
        if n_experts % groups != 0 or top_k > n_experts:
            continue
        grid.append(
            {
                "batch": batch,
                "seq_len": seq_len,
                "hidden_size": hidden,
                "moe_intermediate_size": moe,
                "n_experts": n_experts,
                "top_k": top_k,
                "groups": groups,
            }
        )

    if not grid:
        raise SystemExit("No valid parameter combinations produced")

    header = (
        "batch",
        "seq_len",
        "hidden_size",
        "moe_intermediate",
        "n_experts",
        "top_k",
        "groups",
        "baseline_ms",
        "patched_ms",
        "speedup",
        "baseline_vram_mib",
        "patched_vram_mib",
        "min_tokens",
        "max_tokens",
        "max_diff",
        "accuracy_ok",
    )
    rows = []

    print(
        f"Running sweep on device={args.device} dtype={args.dtype} uniform_routing={args.uniform_routing}"
    )
    print(
        f"{'batch':>5} {'seq':>5} {'hidden':>7} {'experts':>7} {'topk':>4} {'groups':>6}"
        f" {'baseline':>12} {'patched':>12} {'speedup':>8} {'b_vram':>8} {'p_vram':>8} {'acc':>5}"
    )

    for cfg in grid:
        ns = make_namespace(cfg, args)
        result = benchmark_deepseek_v3(ns)
        baseline_vram_mib = (
            result["baseline_vram"] / (1024**2)
            if result["baseline_vram"] is not None
            else float("nan")
        )
        patched_vram_mib = (
            result["patched_vram"] / (1024**2)
            if result["patched_vram"] is not None
            else float("nan")
        )
        rows.append(
            (
                cfg["batch"],
                cfg["seq_len"],
                cfg["hidden_size"],
                cfg["moe_intermediate_size"],
                cfg["n_experts"],
                cfg["top_k"],
                cfg["groups"],
                result["baseline_ms"],
                result["patched_ms"],
                result["speedup"],
                baseline_vram_mib,
                patched_vram_mib,
                result["min_tokens"],
                result["max_tokens"],
                result["max_diff"],
                result["accuracy_ok"],
            )
        )
        status = "OK" if result["accuracy_ok"] else "FAIL"
        print(
            f"{cfg['batch']:>5} {cfg['seq_len']:>5} {cfg['hidden_size']:>7} {cfg['n_experts']:>7} {cfg['top_k']:>4} {cfg['groups']:>6}"
            f" {result['baseline_ms']:>11.3f} ms {result['patched_ms']:>11.3f} ms {result['speedup']:>7.2f}x"
            f" {baseline_vram_mib:>8.1f} {patched_vram_mib:>8.1f} {status:>5}"
        )
        if not result["accuracy_ok"]:
            raise RuntimeError(
                f"Accuracy check failed for config {cfg}: max diff {result['max_diff']:.3e} exceeds tolerance {ACCURACY_TOLERANCE:.1e}"
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
