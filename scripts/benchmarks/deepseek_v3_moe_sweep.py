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
for candidate in [CURRENT_DIR, *CURRENT_DIR.parents]:
    repo_root = candidate / "axolotl"
    if repo_root.exists():
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        break
else:  # pragma: no cover - defensive guard
    raise SystemExit("Unable to locate axolotl repository root for imports")

from scripts.benchmarks.deepseek_v3_moe import (  # noqa: E402
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
        help="Override GROUP_SIZE_M for every configuration",
    )
    parser.add_argument(
        "--no-uniform-routing",
        action="store_true",
        help="Disable uniform routing for every configuration",
    )
    parser.add_argument(
        "--include-mixtral-long",
        action="store_true",
        help="Add an 8×8192 Mixtral-style run to the sweep",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to store results",
    )
    return parser.parse_args()


def make_namespace(
    base: dict, args: argparse.Namespace, backend: str
) -> SimpleNamespace:
    combined = dict(base)
    combined.update(
        {
            "dtype": args.dtype,
            "device": args.device,
            "backend": backend,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "uniform_routing": not args.no_uniform_routing,
        }
    )
    if args.group_size is not None:
        combined["group_size"] = args.group_size
    return SimpleNamespace(**combined)


ARCHETYPES = (
    (
        "mixtral",
        {
            "hidden_size": 4096,
            "moe_intermediate_size": 14336,
            "n_experts": 8,
            "top_k": 2,
            "groups": 1,
            "group_size": 128,
        },
        [(4, 2048), (8, 4096)],
    ),
    (
        "qwen",
        {
            "hidden_size": 6144,
            "moe_intermediate_size": 24576,
            "n_experts": 16,
            "top_k": 4,
            "groups": 8,
            "group_size": 128,
        },
        [(4, 4096), (8, 8192)],
    ),
    (
        "deepseek_v3",
        {
            "hidden_size": 12288,
            "moe_intermediate_size": 49152,
            "n_experts": 128,
            "top_k": 8,
            "groups": 16,
            "group_size": 128,
        },
        [(4, 4096), (8, 8192)],
    ),
)

MIXTRAL_LONG_SHAPES = [(8, 8192)]

BACKENDS = ("cg", "mg")


def main() -> None:  # pragma: no cover - utility script
    args = parse_args()

    grid = []
    for label, base_cfg, shapes in ARCHETYPES:
        for batch, seq_len in shapes:
            cfg = {
                "label": label,
                "batch": batch,
                "seq_len": seq_len,
                **base_cfg,
            }
            if cfg["n_experts"] % cfg["groups"] != 0 or cfg["top_k"] > cfg["n_experts"]:
                continue
            grid.append(cfg)

    if args.include_mixtral_long:
        base_cfg = ARCHETYPES[0][1]
        for batch, seq_len in MIXTRAL_LONG_SHAPES:
            grid.append(
                {
                    "label": "mixtral_long",
                    "batch": batch,
                    "seq_len": seq_len,
                    **base_cfg,
                }
            )

    if not grid:
        raise SystemExit("No valid parameter combinations produced")

    header = (
        "model",
        "batch",
        "seq_len",
        "hidden_size",
        "moe_intermediate",
        "n_experts",
        "top_k",
        "groups",
        "backend",
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

    uniform_flag = not args.no_uniform_routing
    print(
        f"Running sweep on device={args.device} dtype={args.dtype} backends={BACKENDS} uniform_routing={uniform_flag}"
    )
    print(
        f"{'model':>10} {'batch':>5} {'seq':>5} {'hidden':>7} {'experts':>7} {'topk':>4} {'groups':>6} {'backend':>8}"
        f" {'baseline':>12} {'patched':>12} {'speedup':>8} {'b_vram':>8} {'p_vram':>8} {'acc':>5}"
    )

    for cfg in grid:
        for backend in BACKENDS:
            ns = make_namespace(cfg, args, backend)
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
                    cfg["label"],
                    cfg["batch"],
                    cfg["seq_len"],
                    cfg["hidden_size"],
                    cfg["moe_intermediate_size"],
                    cfg["n_experts"],
                    cfg["top_k"],
                    cfg["groups"],
                    backend,
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
                f"{cfg['label']:>10} {cfg['batch']:>5} {cfg['seq_len']:>5} {cfg['hidden_size']:>7} {cfg['n_experts']:>7} {cfg['top_k']:>4} {cfg['groups']:>6} {backend:>8}"
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
