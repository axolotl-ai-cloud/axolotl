#!/usr/bin/env python
# mypy: disable-error-code="operator"
"""Microbenchmark NVFP4 lm_head + CE variants.

This is not a training-throughput benchmark. Use scripts/bench_nvfp4.sh for
end-to-end numbers.
"""

from __future__ import annotations

import argparse
import contextlib
import os

import torch
import torch.nn.functional as F
from torch import nn

from axolotl.integrations.nvfp4.kernels.nvfp4_fused_ce import fused_fp4_cross_entropy
from axolotl.integrations.nvfp4.nvfp4_training import (
    NVFP4FastComputeBaseLinear,
    NVFP4FastFrozenBaseLinear,
    NVFP4FrozenBaseLinear,
    NVFP4Recipe,
)


def _build_head(kind: str, linear: nn.Linear, recipe: NVFP4Recipe) -> nn.Module:
    if kind == "torchao":
        return NVFP4FrozenBaseLinear.from_linear(linear, recipe)
    if kind == "fast-frozen":
        return NVFP4FastFrozenBaseLinear.from_linear(linear, recipe)
    if kind == "fast-compute":
        return NVFP4FastComputeBaseLinear.from_linear(linear, recipe)
    raise ValueError(f"unknown head kind: {kind}")


def _maybe_liger():
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        return LigerFusedLinearCrossEntropyLoss(ignore_index=-100), None
    except Exception as exc:
        return None, exc


def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _finite_float(x: torch.Tensor | None) -> str:
    if x is None:
        return "n/a"
    return f"{float(x.detach()):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=32768)
    parser.add_argument(
        "--head",
        choices=("torchao", "fast-frozen", "fast-compute"),
        default="torchao",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.hidden % 16 != 0 or args.vocab % 32 != 0:
        raise SystemExit("--hidden must be divisible by 16 and --vocab by 32")

    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(
        "device=",
        torch.cuda.get_device_name(device),
        "capability=",
        torch.cuda.get_device_capability(device),
    )
    print(
        f"shape=tokens:{args.tokens} hidden:{args.hidden} vocab:{args.vocab} "
        f"head:{args.head} backward:{args.backward}"
    )

    linear = nn.Linear(args.hidden, args.vocab, bias=False, device=device, dtype=dtype)
    linear.weight.requires_grad_(False)
    head = _build_head(args.head, linear, NVFP4Recipe())
    hidden = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)
    labels = torch.randint(0, args.vocab, (args.tokens,), device=device)
    labels[::97] = -100

    dense_weight = head.weight.detach()
    liger, liger_exc = _maybe_liger()
    if liger_exc is not None:
        print(f"liger=unavailable ({type(liger_exc).__name__}: {liger_exc})")

    def materialized_loss(x):
        return F.cross_entropy(
            (x @ dense_weight.t()).float(), labels, ignore_index=-100
        )

    def old_loss(x):
        return fused_fp4_cross_entropy(x, head, labels, shift=False, fp4_matmul=False)

    def new_loss(x):
        return fused_fp4_cross_entropy(x, head, labels, shift=False, fp4_matmul=True)

    with torch.no_grad():
        ref = materialized_loss(hidden)
        old = old_loss(hidden)
        new = new_loss(hidden)
        print(
            "loss materialized=",
            _finite_float(ref),
            "existing_fp4_ce=",
            _finite_float(old),
            "fp4_scaled_mm_ce=",
            _finite_float(new),
        )
        print(
            "abs_delta existing=",
            "n/a" if old is None else f"{float((old - ref).abs()):.6f}",
            "fp4_scaled_mm=",
            f"{float((new - ref).abs()):.6f}",
        )

    def wrap_loss(loss_fn):
        if not args.backward:

            def run():
                with torch.no_grad():
                    loss_fn(hidden)

            return run

        def run():
            x = hidden.detach().clone().requires_grad_(True)
            loss = loss_fn(x)
            if loss is None:
                return
            loss.backward()

        return run

    timings: list[tuple[str, float | None]] = []
    timings.append(
        (
            "materialized_bf16_ce",
            _time_cuda(wrap_loss(materialized_loss), args.warmup, args.iters),
        )
    )
    if liger is not None:

        def liger_loss(x):
            return liger(dense_weight, x, labels)

        timings.append(
            (
                "liger_fused_linear_ce",
                _time_cuda(wrap_loss(liger_loss), args.warmup, args.iters),
            )
        )
    old_probe = old_loss(hidden)
    if old_probe is not None:
        timings.append(
            (
                "existing_nvfp4_fused_ce",
                _time_cuda(wrap_loss(old_loss), args.warmup, args.iters),
            )
        )
    timings.append(
        (
            "fp4_scaled_mm_ce",
            _time_cuda(wrap_loss(new_loss), args.warmup, args.iters),
        )
    )

    for name, ms in timings:
        tok_s = args.tokens / (ms / 1000.0)
        print(f"{name}: {ms:.3f} ms ({tok_s:.1f} tok/s)")

    del linear, head, dense_weight, hidden, labels
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("AXOLOTL_NVFP4_FUSED_CE_FP4_MM", "1")
    main()
