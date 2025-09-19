#!/usr/bin/env python
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from axolotl.kernels.moe import torch_grouped as tg
except Exception:  # pragma: no cover - fallback when torch_grouped unavailable
    tg = None


class SwiGLUMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(x)) * self.w3(x))


class Experts(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            SwiGLUMlp(hidden_size, intermediate_size) for _ in range(num_experts)
        )
        self.num_experts = num_experts

    def __getitem__(self, idx):
        return self.layers[idx]


def forward_naive(
    hidden_states: torch.Tensor, gate: nn.Linear, experts: Experts, top_k: int
):
    bsz, seqlen, hdim = hidden_states.shape
    x = hidden_states.view(-1, hdim)
    router_logits = gate(x)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, top_k, dim=-1, sorted=False)
    topk_weight = (topk_weight / topk_weight.sum(dim=-1, keepdim=True)).to(x.dtype)
    x_rep = x.repeat_interleave(top_k, dim=0)
    y = torch.empty_like(x_rep)
    flat_idx = topk_idx.view(-1)
    for i in range(experts.num_experts):
        sel = flat_idx == i
        if sel.any():
            y[sel] = experts[i](x_rep[sel])
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    return y.view(bsz, seqlen, hdim)


def bench(fn, iters=50, warmup=10, sync=True):
    # warmup
    for _ in range(warmup):
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    # measure
    times = []
    for _ in range(iters):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
    return sum(times) / len(times)


def estimate_moe_flops(tokens: int, hidden: int, inter: int, top_k: int) -> float:
    """Estimate GEMM FLOPs for a SwiGLU MoE MLP.

    Two up projections (w1,w3) + one down (w2), each token processed by top_k experts.
    FLOPs ≈ 6 * (tokens * top_k) * hidden * inter (2*m*k*n per GEMM).
    """
    m_rep = tokens * top_k
    return 6.0 * m_rep * hidden * inter


def main():
    p = argparse.ArgumentParser(description="MoE microbenchmark")
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=14336)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]
    )
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--hf-block",
        type=str,
        default="none",
        choices=["none", "qwen2_moe"],
        help="Use a Hugging Face MoE block for benchmarking instead of the toy SwiGLU layer.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Capture CUDA profiler tables for naive and grouped runs.",
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    torch.manual_seed(0)
    if device == "cuda":
        torch.cuda.manual_seed(0)

    # data
    x = torch.randn(args.bsz, args.seq, args.hidden, device=device, dtype=dtype)

    tokens = args.bsz * args.seq

    use_hf = args.hf_block != "none"

    if use_hf:
        project_root = Path(__file__).resolve().parents[2]
        transformers_src = project_root / "transformers" / "src"
        if transformers_src.exists() and str(transformers_src) not in sys.path:
            sys.path.append(str(transformers_src))

        if args.hf_block == "qwen2_moe":
            from transformers.models.qwen2_moe.configuration_qwen2_moe import (
                Qwen2MoeConfig,
            )
            from transformers.models.qwen2_moe.modeling_qwen2_moe import (
                Qwen2MoeSparseMoeBlock,
            )

            cfg = Qwen2MoeConfig(
                hidden_size=args.hidden,
                moe_intermediate_size=args.inter,
                shared_expert_intermediate_size=args.inter,
                num_experts=args.experts,
                num_experts_per_tok=args.top_k,
                norm_topk_prob=True,
                qkv_bias=True,
            )

            block_naive = Qwen2MoeSparseMoeBlock(cfg).to(device=device, dtype=dtype)
            block_grouped = Qwen2MoeSparseMoeBlock(cfg).to(device=device, dtype=dtype)
            block_grouped.load_state_dict(block_naive.state_dict())

        def run_naive_model(inp: torch.Tensor) -> torch.Tensor:
            out, _ = block_naive(inp)
            return out

        def run_grouped_model(inp: torch.Tensor) -> torch.Tensor:
            if tg is None or not tg.available():
                return torch.empty(0)
            block_grouped.experts._ax_parent_block = block_grouped
            y, _ = tg.moe_ffn_forward_grouped(
                inp, block_grouped.gate, block_grouped.experts, block_grouped.top_k
            )
            return y if y is not None else torch.empty(0)

        flops_total = estimate_moe_flops(tokens, args.hidden, args.inter, args.top_k)
        print(
            f"Device={device} dtype={dtype} tokens={tokens} hidden={args.hidden} inter={args.inter} "
            f"experts={args.experts} top_k={args.top_k} hf_block={args.hf_block}"
        )

    else:
        experts = Experts(args.experts, args.hidden, args.inter).to(
            device=device, dtype=dtype
        )
        gate = nn.Linear(args.hidden, args.experts, bias=False).to(
            device=device, dtype=dtype
        )

        def run_naive_model(inp: torch.Tensor) -> torch.Tensor:
            return forward_naive(inp, gate, experts, args.top_k)

        def run_grouped_model(inp: torch.Tensor) -> torch.Tensor:
            if tg is None or not tg.available():
                return torch.empty(0)
            y, _ = tg.moe_ffn_forward_grouped(inp, gate, experts, args.top_k)
            return y if y is not None else torch.empty(0)

        flops_total = estimate_moe_flops(tokens, args.hidden, args.inter, args.top_k)
        print(
            f"Device={device} dtype={dtype} tokens={tokens} hidden={args.hidden} inter={args.inter} "
            f"experts={args.experts} top_k={args.top_k}"
        )

    # Benchmark naive
    t_naive = bench(lambda: run_naive_model(x), iters=args.iters, warmup=args.warmup)
    tflops_naive = flops_total / ((t_naive / 1000.0) * 1e12)
    print(
        f"naive\t{t_naive:.2f} ms\t{tokens / (t_naive / 1000):.1f} tok/s\t{tflops_naive:.2f} TFLOP/s"
    )

    with torch.no_grad():
        y_ref = run_naive_model(x)

    # Benchmark grouped
    if tg is not None and tg.available():
        y_grouped = run_grouped_model(x)
        if y_grouped.numel() == 0:
            print("torch_grouped\tN/A (op not callable)")
        else:
            t_grouped = bench(
                lambda: run_grouped_model(x),
                iters=args.iters,
                warmup=args.warmup,
            )
            tflops_grouped = flops_total / ((t_grouped / 1000.0) * 1e12)
            speedup = t_naive / t_grouped
            print(
                f"torch_grouped\t{t_grouped:.2f} ms\t{tokens / (t_grouped / 1000):.1f} tok/s\t"
                f"{tflops_grouped:.2f} TFLOP/s\t{speedup:.2f}×"
            )
            diff = (y_ref.float() - y_grouped.float()).abs()
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            rel_l2 = (
                (diff.pow(2).sum() / (y_ref.float().pow(2).sum() + 1e-12)).sqrt().item()
            )
            print(
                f"torch_grouped_check: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_l2={rel_l2:.3e}"
            )
    else:
        print("torch_grouped\tN/A (unavailable)")

    if args.profile and tg is not None and tg.available():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            run_naive_model(x)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            run_grouped_model(x)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
