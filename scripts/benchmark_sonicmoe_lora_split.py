"""Benchmark SonicMoE expert LoRA: materialized ``W + s * B @ A`` vs the LoRA split path.

One bf16 MoE layer at a Beam EP=8 rank's shape (32 local experts, H=5120, I=2560, LoRA r=64 on
gate_up_proj and down_proj, compacted pairs with 4 local experts per token), timed over a sweep
of routed-pair counts. Reports ms per forward+backward, the peak memory above the inputs, and the
expert-GEMM TFLOPs each path issues (from the shapes, not measured). Optionally also times the
rank-r GEMMs on quack against ``torch._grouped_mm`` (which needs the gathered rows materialized).

    python scripts/benchmark_sonicmoe_lora_split.py --pairs 65536 131072 262144 360448
"""

import argparse
import time
from types import SimpleNamespace

import torch


def expert_gemm_tflops(pairs, num_experts, hidden, inter, rank, recompute):
    """(materialized, split) TFLOPs per layer per micro-step for the expert GEMMs."""
    unit = 2 * pairs * hidden * inter
    forward_base = 3 * unit
    backward_base_mat = 6 * unit
    backward_base_split = 3 * unit
    merge = 2 * num_experts * rank * (2 * inter * hidden + hidden * inter)
    side_fwd = 2 * pairs * rank * (2 * hidden + 3 * inter)
    side_bwd = 2 * pairs * rank * (4 * hidden + 6 * inter)
    fwd_passes = 2 if recompute else 1
    mat = fwd_passes * (forward_base + merge) + backward_base_mat + 2 * merge
    split = fwd_passes * (forward_base + side_fwd) + backward_base_split + side_bwd
    return mat / 1e12, split / 1e12


def build(pairs, num_experts, hidden, inter, rank, per_token, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    num_tokens = pairs // per_token
    w1 = (
        torch.randn(num_experts, 2 * inter, hidden, device="cuda", generator=gen)
        * hidden**-0.5
    ).bfloat16()
    w2 = (
        torch.randn(num_experts, hidden, inter, device="cuda", generator=gen)
        * inter**-0.5
    ).bfloat16()

    def lora(weight):
        e, dim1, dim2 = weight.shape
        a = torch.randn(rank * e, dim2, device="cuda", generator=gen) * 0.01
        b = torch.randn(dim1, rank * e, device="cuda", generator=gen) * 0.01
        return a.bfloat16().requires_grad_(True), b.bfloat16().requires_grad_(True)

    tokens = torch.arange(
        num_tokens, device="cuda", dtype=torch.int32
    ).repeat_interleave(per_token)
    expert_ids = torch.rand(num_tokens, num_experts, device="cuda", generator=gen)
    expert_ids = expert_ids.argsort(dim=1)[:, :per_token].reshape(-1).int()
    scores = torch.rand(pairs, device="cuda", generator=gen).bfloat16()
    x = torch.randn(num_tokens, hidden, device="cuda", generator=gen).bfloat16()
    facade = SimpleNamespace(
        has_post_expert_norm=False,
        config=SimpleNamespace(hidden_act="silu"),
        is_transposed=False,
        is_concatenated=True,
        gate_up_proj=w1,
        down_proj=w2,
        gate_up_proj_bias=None,
        down_proj_bias=None,
        has_bias=False,
        num_experts=num_experts,
    )
    return (
        facade,
        x.requires_grad_(True),
        tokens,
        expert_ids,
        scores,
        lora(w1),
        lora(w2),
    )


def materialized(facade, x, tokens, expert_ids, scores, lora1, lora2, scaling):
    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        _sonicmoe_pairs_forward,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize

    merged = SimpleNamespace(**vars(facade))
    merged.gate_up_proj = MoELoRAMaterialize.apply(facade.gate_up_proj, *lora1, scaling)
    merged.down_proj = MoELoRAMaterialize.apply(facade.down_proj, *lora2, scaling)
    return _sonicmoe_pairs_forward(merged, x, tokens, expert_ids, scores)


def split(facade, x, tokens, expert_ids, scores, lora1, lora2, scaling):
    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import (
        sonicmoe_lora_split_forward,
    )

    return sonicmoe_lora_split_forward(
        facade,
        x,
        tokens,
        expert_ids,
        scores,
        (*lora1, scaling),
        (*lora2, scaling),
    )


def time_step(fn, case, scaling, warmup, iters):
    facade, x, tokens, expert_ids, scores, lora1, lora2 = case
    scores = scores.clone().requires_grad_(True)
    leaves = (x, scores, *lora1, *lora2)

    def step():
        out = fn(facade, x, tokens, expert_ids, scores, lora1, lora2, scaling)
        torch.autograd.grad(out.float().square().mean(), leaves)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1e3
    return ms, (torch.cuda.max_memory_allocated() - base) / 2**30


def bench_rank_r(case, warmup, iters):
    """u = x[gather] @ P per expert: quack with gather vs ``torch._grouped_mm`` on x[gather]."""
    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import sonic_ops

    facade, x, tokens, expert_ids, _scores, (lora_A, _), _ = case
    ops = sonic_ops()
    num_experts = facade.num_experts
    rank = lora_A.shape[0] // num_experts
    meta = ops.routing_metadata(tokens, expert_ids, x.shape[0], num_experts)
    p = lora_A.detach().reshape(num_experts, rank, -1).transpose(1, 2).contiguous()
    out = torch.empty(tokens.numel(), rank, dtype=x.dtype, device="cuda")
    xd = x.detach()

    def quack():
        ops.gemm(xd, p, out, meta.cu_seqlens, meta.x_gather_idx)

    offs = meta.cu_seqlens[1:].to(torch.int32)

    def grouped_mm():
        rows = xd[meta.x_gather_idx.long()]
        torch._grouped_mm(rows, p, offs=offs)

    results = {}
    for name, fn in (("quack(gather)", quack), ("torch._grouped_mm", grouped_mm)):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            results[name] = (time.perf_counter() - start) / iters * 1e3
        except Exception as exc:  # noqa: BLE001
            results[name] = f"failed: {type(exc).__name__}: {exc}"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pairs", type=int, nargs="+", default=[65536, 131072, 262144, 360448]
    )
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--inter", type=int, default=2560)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--per-token", type=int, default=4)
    parser.add_argument("--scaling", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rank-r-backends", action="store_true")
    args = parser.parse_args()

    from axolotl.integrations.kernels.plugin import _shim_nvvm_mbarrier_arrive

    _shim_nvvm_mbarrier_arrive()
    print(
        f"E={args.experts} H={args.hidden} I={args.inter} r={args.rank} "
        f"on {torch.cuda.get_device_name()}"
    )
    header = (
        "pairs",
        "mat ms",
        "split ms",
        "mat GiB",
        "split GiB",
        "mat TF",
        "split TF",
        "mat TF+AC",
        "split TF+AC",
    )
    print(" | ".join(f"{h:>10}" for h in header))
    for pairs in args.pairs:
        case = build(
            pairs, args.experts, args.hidden, args.inter, args.rank, args.per_token
        )
        mat_ms, mat_gib = time_step(
            materialized, case, args.scaling, args.warmup, args.iters
        )
        split_ms, split_gib = time_step(
            split, case, args.scaling, args.warmup, args.iters
        )
        flops = expert_gemm_tflops(
            pairs, args.experts, args.hidden, args.inter, args.rank, False
        )
        flops_ac = expert_gemm_tflops(
            pairs, args.experts, args.hidden, args.inter, args.rank, True
        )
        row = (pairs, mat_ms, split_ms, mat_gib, split_gib, *flops, *flops_ac)
        print(
            " | ".join(
                f"{v:>10d}" if isinstance(v, int) else f"{v:>10.2f}" for v in row
            )
        )
        if args.rank_r_backends:
            print(
                "   rank-r u = x P (ms):", bench_rank_r(case, args.warmup, args.iters)
            )
        del case
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
