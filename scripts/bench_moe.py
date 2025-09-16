#!/usr/bin/env python
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def bench(fn, *args, iters=50, warmup=10, sync=True):
    # warmup
    for _ in range(warmup):
        out = fn(*args)
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
    # measure
    times = []
    for _ in range(iters):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
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

    # Model
    experts = Experts(args.experts, args.hidden, args.inter).to(
        device=device, dtype=dtype
    )
    gate = nn.Linear(args.hidden, args.experts, bias=False).to(
        device=device, dtype=dtype
    )

    # data
    x = torch.randn(args.bsz, args.seq, args.hidden, device=device, dtype=dtype)

    # Report config
    tokens = args.bsz * args.seq
    print(
        f"Device={device} dtype={dtype} tokens={tokens} hidden={args.hidden} inter={args.inter} experts={args.experts} top_k={args.top_k}"
    )

    # Naive baseline
    t_naive = bench(
        forward_naive,
        x,
        gate,
        experts,
        args.top_k,
        iters=args.iters,
        warmup=args.warmup,
    )
    flops_total = estimate_moe_flops(tokens, args.hidden, args.inter, args.top_k)
    tflops_naive = flops_total / ((t_naive / 1000.0) * 1e12)
    print(
        f"naive\t{t_naive:.2f} ms\t{tokens / (t_naive / 1000):.1f} tok/s\t{tflops_naive:.2f} TFLOP/s"
    )

    # Prepare reference output once for checks
    with torch.no_grad():
        y_ref = forward_naive(x, gate, experts, args.top_k)

    # torch_grouped backend (PyTorch 2.8+)
    try:
        from axolotl.kernels.moe import torch_grouped as tg
    except Exception:
        tg = None
    if tg is not None and tg.available():

        def forward_tg(a, g, ex, topk):
            y, _ = tg.moe_ffn_forward_grouped(a, g, ex, topk)
            return y

        y_tg = forward_tg(x, gate, experts, args.top_k)
        if y_tg is not None:
            t_ms = bench(
                forward_tg,
                x,
                gate,
                experts,
                args.top_k,
                iters=args.iters,
                warmup=args.warmup,
            )
            tflops = flops_total / ((t_ms / 1000.0) * 1e12)
            speedup = t_naive / t_ms
            print(
                f"torch_grouped\t{t_ms:.2f} ms\t{tokens / (t_ms / 1000):.1f} tok/s\t{tflops:.2f} TFLOP/s\t{speedup:.2f}×"
            )
            with torch.no_grad():
                y_fast = y_tg
                y_ref32 = y_ref.float()
                y_fast32 = y_fast.float()
                diff = (y_ref32 - y_fast32).abs()
                max_abs = diff.max().item()
                mean_abs = diff.mean().item()
                rel_l2 = (
                    (diff.pow(2).sum() / (y_ref32.pow(2).sum() + 1e-12)).sqrt().item()
                )
                print(
                    f"torch_grouped_check: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_l2={rel_l2:.3e}"
                )
        else:
            try:
                from axolotl.kernels.moe.torch_grouped import LAST_ERROR as _TG_ERR
            except Exception:
                _TG_ERR = None
            reason = f" (reason: {_TG_ERR})" if _TG_ERR else ""
            print(f"torch_grouped\tN/A (op not callable){reason}")
    else:
        print("torch_grouped\tN/A (unavailable)")


if __name__ == "__main__":
    main()
