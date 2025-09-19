#!/usr/bin/env python
"""Sweep Torchtitan MoE grouped vs naive configurations and report performance."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TITAN_PATH = _PROJECT_ROOT / "torchtitan"
if str(_TITAN_PATH) not in sys.path:
    sys.path.append(str(_TITAN_PATH))

from torchtitan.models.moe import MoE, MoEArgs


def _parse_int_list(value: str) -> List[int]:
    return [int(v) for v in value.split(",") if v]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Torchtitan MoE grouped vs naive sweep")
    p.add_argument(
        "--batch-sizes", default="4,8,16", help="Comma separated batch sizes"
    )
    p.add_argument(
        "--seq-lens", default="1024,2048", help="Comma separated sequence lengths"
    )
    p.add_argument(
        "--experts", default="8,16,32,64", help="Comma separated expert counts"
    )
    p.add_argument("--top-ks", default="1,2,4", help="Comma separated top_k choices")
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=14336)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--init-std", type=float, default=0.02)
    p.add_argument("--score-before", action="store_true")
    p.add_argument("--score-func", choices=["softmax", "sigmoid"], default="softmax")
    p.add_argument("--route-norm", action="store_true")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    return p.parse_args()


def _map_dtype(arg: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[arg]


def _estimate_flops(tokens: int, hidden: int, inter: int, top_k: int) -> float:
    return 6.0 * tokens * top_k * hidden * inter


def _prepare_module(module: MoE, *, device: torch.device, dtype: torch.dtype) -> MoE:
    module = module.to(device=device)
    for param in module.parameters():
        param.data = param.data.to(dtype)
        if param.grad is not None:
            param.grad = None
    for name, buf in module.named_buffers():
        if name == "tokens_per_expert":
            module._buffers[name] = torch.zeros_like(
                buf, dtype=torch.float32, device=device
            )
        elif name == "expert_bias" and buf is not None:
            module._buffers[name] = torch.zeros_like(
                buf, dtype=torch.float32, device=device
            )
        else:
            module._buffers[name] = buf.to(device=device, dtype=dtype)
    module.eval()
    return module


@torch.inference_mode()
def _forward(module: MoE, x: torch.Tensor) -> torch.Tensor:
    return module(x)


def _bench(callable_, *, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        callable_()
        if device.type == "cuda":
            torch.cuda.synchronize()
    timings: List[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        callable_()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return sum(timings) / len(timings)


@dataclass
class SweepResult:
    bsz: int
    seq: int
    experts: int
    top_k: int
    dtype: str
    naive_ms: float
    grouped_ms: float
    speedup: float
    naive_tflops: float
    grouped_tflops: float
    max_abs: float
    mean_abs: float
    rel_l2: float


def _run_case(
    *,
    bsz: int,
    seq: int,
    experts: int,
    top_k: int,
    hidden: int,
    inter: int,
    dtype: torch.dtype,
    device: torch.device,
    iters: int,
    warmup: int,
    init_std: float,
    score_before: bool,
    score_func: str,
    route_norm: bool,
) -> SweepResult:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed(0)

    moe_args_grouped = MoEArgs(
        num_experts=experts,
        num_shared_experts=0,
        score_func=score_func,
        route_norm=route_norm,
        top_k=top_k,
        use_grouped_mm=True,
        score_before_experts=score_before,
        load_balance_coeff=None,
    )
    moe_grouped = MoE(moe_args_grouped, dim=hidden, hidden_dim=inter)
    moe_grouped.init_weights(init_std, buffer_device=device)

    moe_args_naive = MoEArgs(
        num_experts=experts,
        num_shared_experts=0,
        score_func=score_func,
        route_norm=route_norm,
        top_k=top_k,
        use_grouped_mm=False,
        score_before_experts=score_before,
        load_balance_coeff=None,
    )
    moe_naive = MoE(moe_args_naive, dim=hidden, hidden_dim=inter)
    moe_naive.load_state_dict(moe_grouped.state_dict(), strict=True)

    moe_grouped = _prepare_module(moe_grouped, device=device, dtype=dtype)
    moe_naive = _prepare_module(moe_naive, device=device, dtype=dtype)

    x = torch.randn(bsz, seq, hidden, device=device, dtype=dtype)

    def run_naive():
        if hasattr(moe_naive, "tokens_per_expert"):
            moe_naive.tokens_per_expert.zero_()
        return _forward(moe_naive, x)

    def run_grouped():
        if hasattr(moe_grouped, "tokens_per_expert"):
            moe_grouped.tokens_per_expert.zero_()
        return _forward(moe_grouped, x)

    naive_ms = _bench(run_naive, iters=iters, warmup=warmup, device=device)
    y_naive = run_naive()

    grouped_ms = _bench(run_grouped, iters=iters, warmup=warmup, device=device)
    y_grouped = run_grouped()

    diff = (y_naive.float() - y_grouped.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_l2 = (diff.pow(2).sum() / (y_naive.float().pow(2).sum() + 1e-12)).sqrt().item()

    tokens = bsz * seq
    flops = _estimate_flops(tokens, hidden, inter, top_k)
    naive_tflops = flops / ((naive_ms / 1000.0) * 1e12)
    grouped_tflops = flops / ((grouped_ms / 1000.0) * 1e12)
    speedup = naive_ms / grouped_ms if grouped_ms > 0 else float("nan")

    return SweepResult(
        bsz=bsz,
        seq=seq,
        experts=experts,
        top_k=top_k,
        dtype=str(dtype),
        naive_ms=naive_ms,
        grouped_ms=grouped_ms,
        speedup=speedup,
        naive_tflops=naive_tflops,
        grouped_tflops=grouped_tflops,
        max_abs=max_abs,
        mean_abs=mean_abs,
        rel_l2=rel_l2,
    )


def _print_header(
    hidden: int, inter: int, dtype: torch.dtype, device: torch.device
) -> None:
    print(f"Device={device} dtype={dtype} hidden={hidden} inter={inter}")
    print(
        "bsz\tseq\texperts\ttop_k\tnaive(ms)\tgrouped(ms)\tspeedup\t"
        "naive TF/s\tgrouped TF/s\tmax_abs\tmean_abs\trel_l2"
    )


def _print_result(res: SweepResult) -> None:
    print(
        f"{res.bsz}\t{res.seq}\t{res.experts}\t{res.top_k}\t"
        f"{res.naive_ms:.2f}\t{res.grouped_ms:.2f}\t{res.speedup:.2f}\t"
        f"{res.naive_tflops:.2f}\t{res.grouped_tflops:.2f}\t"
        f"{res.max_abs:.2e}\t{res.mean_abs:.2e}\t{res.rel_l2:.2e}"
    )


def _write_csv(path: Path, results: Iterable[SweepResult]) -> None:
    fieldnames = [
        "batch_size",
        "seq_len",
        "experts",
        "top_k",
        "dtype",
        "naive_ms",
        "grouped_ms",
        "speedup",
        "naive_tflops",
        "grouped_tflops",
        "max_abs",
        "mean_abs",
        "rel_l2",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "batch_size": r.bsz,
                    "seq_len": r.seq,
                    "experts": r.experts,
                    "top_k": r.top_k,
                    "dtype": r.dtype,
                    "naive_ms": f"{r.naive_ms:.4f}",
                    "grouped_ms": f"{r.grouped_ms:.4f}",
                    "speedup": f"{r.speedup:.4f}",
                    "naive_tflops": f"{r.naive_tflops:.4f}",
                    "grouped_tflops": f"{r.grouped_tflops:.4f}",
                    "max_abs": f"{r.max_abs:.6e}",
                    "mean_abs": f"{r.mean_abs:.6e}",
                    "rel_l2": f"{r.rel_l2:.6e}",
                }
            )


def main() -> None:
    args = _parse_args()
    dtype = _map_dtype(args.dtype)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_sizes = _parse_int_list(args.batch_sizes)
    seq_lens = _parse_int_list(args.seq_lens)
    experts_list = _parse_int_list(args.experts)
    top_ks = _parse_int_list(args.top_ks)

    results: List[SweepResult] = []
    _print_header(args.hidden, args.inter, dtype, device)

    for bsz in batch_sizes:
        for seq in seq_lens:
            for experts in experts_list:
                for top_k in top_ks:
                    try:
                        res = _run_case(
                            bsz=bsz,
                            seq=seq,
                            experts=experts,
                            top_k=top_k,
                            hidden=args.hidden,
                            inter=args.inter,
                            dtype=dtype,
                            device=device,
                            iters=args.iters,
                            warmup=args.warmup,
                            init_std=args.init_std,
                            score_before=args.score_before,
                            score_func=args.score_func,
                            route_norm=args.route_norm,
                        )
                    except RuntimeError as err:
                        print(
                            f"{bsz}\t{seq}\t{experts}\t{top_k}\tERROR: {err}",
                            file=sys.stderr,
                        )
                        continue
                    results.append(res)
                    _print_result(res)

    if args.csv and results:
        _write_csv(args.csv, results)
        print(f"Wrote {len(results)} rows to {args.csv}")


if __name__ == "__main__":
    main()
