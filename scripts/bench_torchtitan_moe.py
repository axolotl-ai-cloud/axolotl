#!/usr/bin/env python
"""Benchmark Torchtitan MoE grouped vs naive expert execution."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Ensure torchtitan is importable when running from the axolotl tree
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TITAN_PATH = _PROJECT_ROOT / "torchtitan"
if str(_TITAN_PATH) not in sys.path:
    sys.path.append(str(_TITAN_PATH))

from torchtitan.models.moe import MoE, MoEArgs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Torchtitan MoE microbenchmark")
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=14336)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--init-std", type=float, default=0.02)
    p.add_argument(
        "--score-before",
        action="store_true",
        help="Apply routing scores before expert computation (default: after)",
    )
    p.add_argument(
        "--score-func",
        choices=["softmax", "sigmoid"],
        default="softmax",
    )
    p.add_argument(
        "--route-norm",
        action="store_true",
        help="Enable Torchtitan router normalization when using sigmoid scores.",
    )
    return p.parse_args()


def _map_dtype(arg: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[arg]


def _estimate_moe_flops(tokens: int, hidden: int, inter: int, top_k: int) -> float:
    # Two up projections + one down projection per expert/token combination.
    return 6.0 * tokens * top_k * hidden * inter


def _prepare_module(
    moe: MoE,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> MoE:
    moe = moe.to(device=device)
    for param in moe.parameters():
        param.data = param.data.to(dtype)
        if param.grad is not None:
            param.grad = None

    buffers = dict(moe.named_buffers())
    for name, buf in buffers.items():
        if name == "tokens_per_expert":
            moe._buffers[name] = torch.zeros_like(
                buf, dtype=torch.float32, device=device
            )
        elif name == "expert_bias" and buf is not None:
            moe._buffers[name] = torch.zeros_like(
                buf, dtype=torch.float32, device=device
            )
        else:
            moe._buffers[name] = buf.to(device=device, dtype=dtype)
    moe.eval()
    return moe


@torch.inference_mode()
def _forward_fn(module: MoE, x: torch.Tensor) -> torch.Tensor:
    return module(x)


def _bench(fn, *, iters: int, warmup: int, sync: bool = True) -> float:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for _ in range(warmup):
        fn()
        if sync and device.type == "cuda":
            torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if sync and device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync and device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return sum(times) / len(times)


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = _map_dtype(args.dtype)

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed(0)

    moe_args_grouped = MoEArgs(
        num_experts=args.experts,
        num_shared_experts=0,
        score_func=args.score_func,
        route_norm=args.route_norm,
        top_k=args.top_k,
        use_grouped_mm=True,
        score_before_experts=args.score_before,
        load_balance_coeff=None,
    )
    moe_grouped = MoE(moe_args_grouped, dim=args.hidden, hidden_dim=args.inter)
    moe_grouped.init_weights(args.init_std, buffer_device=device)

    moe_args_naive = MoEArgs(
        num_experts=args.experts,
        num_shared_experts=0,
        score_func=args.score_func,
        route_norm=args.route_norm,
        top_k=args.top_k,
        use_grouped_mm=False,
        score_before_experts=args.score_before,
        load_balance_coeff=None,
    )
    moe_naive = MoE(moe_args_naive, dim=args.hidden, hidden_dim=args.inter)
    moe_naive.load_state_dict(moe_grouped.state_dict(), strict=True)

    moe_grouped = _prepare_module(moe_grouped, device=device, dtype=dtype)
    moe_naive = _prepare_module(moe_naive, device=device, dtype=dtype)

    x = torch.randn(args.bsz, args.seq, args.hidden, device=device, dtype=dtype)

    tokens = args.bsz * args.seq
    print(
        f"Device={device} dtype={dtype} tokens={tokens} hidden={args.hidden} "
        f"inter={args.inter} experts={args.experts} top_k={args.top_k}"
    )

    def run_naive():
        return _forward_fn(moe_naive, x)

    def run_grouped():
        return _forward_fn(moe_grouped, x)

    if hasattr(moe_naive, "tokens_per_expert"):
        moe_naive.tokens_per_expert.zero_()
    if hasattr(moe_grouped, "tokens_per_expert"):
        moe_grouped.tokens_per_expert.zero_()

    t_naive = _bench(run_naive, iters=args.iters, warmup=args.warmup)
    flops = _estimate_moe_flops(tokens, args.hidden, args.inter, args.top_k)
    tflops_naive = flops / ((t_naive / 1000.0) * 1e12)
    print(
        f"naive\t{t_naive:.2f} ms\t{tokens / (t_naive / 1000.0):.1f} tok/s\t"
        f"{tflops_naive:.2f} TFLOP/s"
    )

    y_naive = run_naive()

    if hasattr(moe_grouped, "tokens_per_expert"):
        moe_grouped.tokens_per_expert.zero_()

    t_grouped = _bench(run_grouped, iters=args.iters, warmup=args.warmup)
    tflops_grouped = flops / ((t_grouped / 1000.0) * 1e12)
    speedup = t_naive / t_grouped if t_grouped > 0 else float("nan")
    print(
        f"grouped\t{t_grouped:.2f} ms\t{tokens / (t_grouped / 1000.0):.1f} tok/s\t"
        f"{tflops_grouped:.2f} TFLOP/s\t{speedup:.2f}×"
    )

    y_grouped = run_grouped()
    diff = (y_naive.float() - y_grouped.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_l2 = (diff.pow(2).sum() / (y_naive.float().pow(2).sum() + 1e-12)).sqrt().item()
    print(
        f"grouped_check: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_l2={rel_l2:.3e}"
    )


if __name__ == "__main__":
    main()
