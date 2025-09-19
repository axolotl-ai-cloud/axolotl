#!/usr/bin/env python
"""Sweep grouped_mm vs naive performance for Qwen2 MoE block."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch._dynamo as dynamo

try:
    from axolotl.kernels.moe import torch_grouped as tg
except Exception:  # pragma: no cover
    tg = None


def _parse_list(arg: str) -> List[int]:
    return [int(v) for v in arg.split(",") if v]


def _bench(run, *, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        run()
        if device.type == "cuda":
            torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return sum(times) / len(times)


def _estimate_flops(tokens: int, hidden: int, inter: int, top_k: int) -> float:
    return 6.0 * tokens * top_k * hidden * inter


def _load_block(
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


@dataclass
class Result:
    bsz: int
    seq: int
    hidden: int
    inter: int
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


def main() -> None:
    p = argparse.ArgumentParser(description="Grouped MoE sweep")
    p.add_argument("--batch-sizes", default="4,8,16")
    p.add_argument("--seq-lens", default="512,1024,2048")
    p.add_argument("--hidden", default="2048,4096")
    p.add_argument("--inter", default="5632,8192,14336")
    p.add_argument("--experts", default="8,16,32")
    p.add_argument("--top-k", default="1,2,4")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    if tg is None or not tg.available():
        print("torch_grouped unavailable; sweep aborted")
        return

    bs_list = _parse_list(args.batch_sizes)
    seq_list = _parse_list(args.seq_lens)
    hidden_list = _parse_list(args.hidden)
    inter_list = _parse_list(args.inter)
    expert_list = _parse_list(args.experts)
    topk_list = _parse_list(args.top_k)

    results: List[Result] = []

    print(
        "bsz\tseq\thidden\tinter\texperts\ttop_k\tnaive(ms)\tgrouped(ms)\tspeedup\t"
        "naive TF/s\tgrouped TF/s\tmax_abs\tmean_abs\trel_l2"
    )

    for bsz in bs_list:
        for seq in seq_list:
            tokens = bsz * seq
            for hidden in hidden_list:
                for inter in inter_list:
                    for experts in expert_list:
                        for top_k in topk_list:
                            torch.manual_seed(0)
                            if device.type == "cuda":
                                torch.cuda.manual_seed(0)

                            block_naive, block_grouped = _load_block(
                                hidden,
                                inter,
                                experts,
                                top_k,
                                device=device,
                                dtype=dtype,
                            )

                            x = torch.randn(
                                bsz, seq, hidden, device=device, dtype=dtype
                            )

                            compiled_impl = None
                            if args.compile:
                                dynamo.config.capture_scalar_outputs = True
                                dynamo.config.allow_unspec_int_on_nn_module = True
                                try:
                                    block_naive = torch.compile(block_naive)  # type: ignore[arg-type]
                                except Exception as exc:
                                    print(
                                        f"torch.compile naive failed ({exc}); using eager"
                                    )
                                else:

                                    def grouped_forward(inp, *, block=block_grouped):
                                        block.experts._ax_parent_block_ref = (
                                            weakref.ref(block)
                                        )  # type: ignore[attr-defined]
                                        y, _ = tg.moe_ffn_forward_grouped(
                                            inp,
                                            block.gate,
                                            block.experts,
                                            block.top_k,
                                        )
                                        return y

                                    try:
                                        compiled_impl = torch.compile(grouped_forward)  # type: ignore[arg-type]
                                    except Exception as exc:
                                        print(
                                            f"torch.compile grouped failed ({exc}); using eager"
                                        )
                                        compiled_impl = None

                            def run_naive(block=block_naive, data=x):
                                y, _ = block(data)
                                return y

                            def run_grouped(
                                block=block_grouped, data=x, impl=compiled_impl
                            ):
                                if impl is not None:
                                    return impl(data)
                                block.experts._ax_parent_block_ref = weakref.ref(block)  # type: ignore[attr-defined]
                                y, _ = tg.moe_ffn_forward_grouped(
                                    data,
                                    block.gate,
                                    block.experts,
                                    block.top_k,
                                )
                                return y

                            naive_ms = _bench(
                                run_naive,
                                iters=args.iters,
                                warmup=args.warmup,
                                device=device,
                            )
                            y_naive = run_naive()

                            grouped_ms = _bench(
                                run_grouped,
                                iters=args.iters,
                                warmup=args.warmup,
                                device=device,
                            )
                            y_grouped = run_grouped()

                            diff = (y_naive.float() - y_grouped.float()).abs()
                            res = Result(
                                bsz,
                                seq,
                                hidden,
                                inter,
                                experts,
                                top_k,
                                args.dtype,
                                naive_ms,
                                grouped_ms,
                                naive_ms / grouped_ms,
                                _estimate_flops(tokens, hidden, inter, top_k)
                                / ((naive_ms / 1000.0) * 1e12),
                                _estimate_flops(tokens, hidden, inter, top_k)
                                / ((grouped_ms / 1000.0) * 1e12),
                                diff.max().item(),
                                diff.mean().item(),
                                (
                                    (
                                        diff.pow(2).sum()
                                        / (y_naive.float().pow(2).sum() + 1e-12)
                                    )
                                    .sqrt()
                                    .item()
                                ),
                            )
                            results.append(res)
                            print(
                                f"{bsz}\t{seq}\t{hidden}\t{inter}\t{experts}\t{top_k}\t{res.naive_ms:.2f}\t"
                                f"{res.grouped_ms:.2f}\t{res.speedup:.2f}\t{res.naive_tflops:.2f}\t"
                                f"{res.grouped_tflops:.2f}\t{res.max_abs:.2e}\t{res.mean_abs:.2e}\t{res.rel_l2:.2e}"
                            )

    if args.csv:
        fieldnames = [
            "bsz",
            "seq",
            "hidden",
            "inter",
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
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "bsz": r.bsz,
                        "seq": r.seq,
                        "hidden": r.hidden,
                        "inter": r.inter,
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


if __name__ == "__main__":
    import weakref

    main()
