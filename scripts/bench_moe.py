#!/usr/bin/env python
"""Benchmark Hugging Face Qwen2 MoE block with and without grouped_mm."""

from __future__ import annotations

import argparse
import sys
import time
import weakref
from pathlib import Path

import torch

try:
    from axolotl.kernels.moe import torch_grouped as tg
except Exception:  # pragma: no cover
    tg = None


def bench(run, *, iters: int, warmup: int, sync: bool = True) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(warmup):
        run()
        if sync and device.type == "cuda":
            torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if sync and device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        if sync and device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return sum(times) / len(times)


def estimate_moe_flops(tokens: int, hidden: int, inter: int, top_k: int) -> float:
    return 6.0 * tokens * top_k * hidden * inter


def load_hf_block(
    hidden: int,
    inter: int,
    experts: int,
    top_k: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    project_root = Path(__file__).resolve().parents[2]
    transformers_src = project_root / "transformers" / "src"
    if transformers_src.exists() and str(transformers_src) not in sys.path:
        sys.path.append(str(transformers_src))

    from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

    cfg = Qwen2MoeConfig(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter,
        num_experts=experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
        qkv_bias=True,
    )

    block = Qwen2MoeSparseMoeBlock(cfg).to(device=device, dtype=dtype)
    block_grouped = Qwen2MoeSparseMoeBlock(cfg).to(device=device, dtype=dtype)
    block_grouped.load_state_dict(block.state_dict())
    return block, block_grouped


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen2 MoE grouped_mm benchmark")
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--inter", type=int, default=2048)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--compile",
        action="store_true",
        help="Torch.compile both paths before benchmarking",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed(0)

    block_naive, block_grouped = load_hf_block(
        args.hidden,
        args.inter,
        args.experts,
        args.top_k,
        device=device,
        dtype=dtype,
    )

    tokens = args.bsz * args.seq
    flops_total = estimate_moe_flops(tokens, args.hidden, args.inter, args.top_k)
    print(
        f"Device={device} dtype={dtype} tokens={tokens} hidden={args.hidden} inter={args.inter} "
        f"experts={args.experts} top_k={args.top_k}"
    )

    x = torch.randn(args.bsz, args.seq, args.hidden, device=device, dtype=dtype)

    # Optional torch.compile
    run_grouped_impl = None
    if args.compile:
        try:
            block_naive = torch.compile(block_naive)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print(f"torch.compile naive failed ({exc}); using eager")
        else:

            def grouped_forward(inp, *, block=block_grouped):
                block.experts._ax_parent_block_ref = weakref.ref(block)  # type: ignore[attr-defined]
                y, _ = tg.moe_ffn_forward_grouped(
                    inp, block.gate, block.experts, block.top_k
                )
                return y

            try:
                run_grouped_impl = torch.compile(grouped_forward)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                print(f"torch.compile grouped failed ({exc}); using eager")
                run_grouped_impl = None

    def run_naive(block=block_naive, data=x):
        y, _ = block(data)
        return y

    def run_grouped(block=block_grouped, data=x, impl=run_grouped_impl):
        if impl is not None:
            return impl(data)
        if tg is None or not tg.available():
            return torch.empty(0)
        block.experts._ax_parent_block_ref = weakref.ref(block)  # type: ignore[attr-defined]
        y, _ = tg.moe_ffn_forward_grouped(data, block.gate, block.experts, block.top_k)
        return y if y is not None else torch.empty(0)

    t_naive = bench(run_naive, iters=args.iters, warmup=args.warmup)
    tflops_naive = flops_total / ((t_naive / 1000.0) * 1e12)
    print(
        f"naive\t{t_naive:.2f} ms\t{tokens / (t_naive / 1000.0):.1f} tok/s\t{tflops_naive:.2f} TFLOP/s"
    )

    with torch.no_grad():
        y_ref = run_naive()

    if tg is None or not tg.available():
        print("torch_grouped\tN/A (unavailable)")
        return

    y_grouped = run_grouped()
    if y_grouped.numel() == 0:
        print("torch_grouped\tN/A (op not callable)")
        return

    t_grouped = bench(run_grouped, iters=args.iters, warmup=args.warmup)
    tflops_grouped = flops_total / ((t_grouped / 1000.0) * 1e12)
    speedup = t_naive / t_grouped
    print(
        f"torch_grouped\t{t_grouped:.2f} ms\t{tokens / (t_grouped / 1000.0):.1f} tok/s\t"
        f"{tflops_grouped:.2f} TFLOP/s\t{speedup:.2f}×"
    )

    diff = (y_ref.float() - y_grouped.float()).abs()
    print(
        "torch_grouped_check: "
        f"max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e} "
        f"rel_l2={(diff.pow(2).sum() / (y_ref.float().pow(2).sum() + 1e-12)).sqrt().item():.3e}"
    )

    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            run_naive()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            run_grouped()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
