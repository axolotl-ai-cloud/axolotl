"""Benchmark for ScatterMoE LoRA Triton kernels.

Measures forward, backward dX, and backward dA/dB kernels at common MoE
model shapes. Reports per-kernel timings, LoRA overhead vs base scatter2scatter,
and full fwd+bwd autograd throughput.

Usage:
  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_scattermoe_lora.py
  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_scattermoe_lora.py --ranks 16 64
  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_scattermoe_lora.py --models Qwen/Qwen3.5-35B-A3B
"""

import argparse
import gc
import time
from functools import partial

import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.kernels import (
    lora_ops,
    ops as base_ops,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    flatten_sort_count,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (
    ScatterMoELoRA,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP = 5
ITERS = 20

# ─── Model configs ──────────────────────────────────────────────────────────

BUILTIN_CONFIGS = {
    "Qwen3.5-35B-A3B": (256, 2048, 512, 8),  # E, H, I, k
    "Qwen3-30B-A3B": (128, 2048, 768, 8),
    "OLMoE-1B-7B": (64, 2048, 1024, 8),
    "Mixtral-8x7B": (8, 4096, 14336, 2),
}


def _resolve_config(spec):
    """Resolve a model spec to (E, H, I, k). Accepts builtin names or HF IDs."""
    key = spec.lower().replace("/", "-")
    for name, cfg in BUILTIN_CONFIGS.items():
        if key in name.lower() or name.lower() in key:
            return name, cfg

    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(spec, trust_remote_code=True)
    if callable(getattr(hf_cfg, "get_text_config", None)):
        tc = hf_cfg.get_text_config()
        if hasattr(tc, "model_type") and tc.model_type != hf_cfg.model_type:
            hf_cfg = tc
    hidden = hf_cfg.hidden_size
    inter = getattr(hf_cfg, "moe_intermediate_size", None) or hf_cfg.intermediate_size
    experts = (
        getattr(hf_cfg, "num_experts", None)
        or getattr(hf_cfg, "num_local_experts", None)
        or getattr(hf_cfg, "n_routed_experts", None)
    )
    top_k = (
        getattr(hf_cfg, "num_experts_per_tok", None)
        or getattr(hf_cfg, "num_experts_per_token", None)
        or 2
    )
    name = spec.split("/")[-1]
    return name, (experts, hidden, inter, top_k)


# ─── Benchmark helpers ──────────────────────────────────────────────────────


def _clean():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def _setup(num_experts, K, N, T, top_k, R):
    torch.manual_seed(42)
    x = torch.randn(T, K, device=DEVICE, dtype=DTYPE)
    W = torch.randn(num_experts, K, N, device=DEVICE, dtype=DTYPE) * 0.02
    lora_A = torch.randn(R * num_experts, K, device=DEVICE, dtype=DTYPE) * 0.01
    lora_B = torch.randn(N, R * num_experts, device=DEVICE, dtype=DTYPE) * 0.01
    logits = torch.randn(T, num_experts, device=DEVICE)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    sei, ssi, eo = flatten_sort_count(top_idx, num_experts)
    gx = base_ops.group(x, ssi, fan_out=top_k)
    dy = torch.randn(gx.size(0), N, device=DEVICE, dtype=DTYPE)
    return x, W, lora_A, lora_B, sei, ssi, eo, gx, dy


# ─── Kernel wrappers (avoid B023 loop-variable capture) ──────────────────────


def _call_fwd(x, W, sei, ssi, top_k, lA, lB):
    return lora_ops.scatter2scatter_lora(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=top_k,
        lora_A=lA,
        lora_B=lB,
        scaling=2.0,
    )


def _call_base(x, W, sei, ssi, top_k):
    return base_ops.scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=top_k,
    )


def _call_dx(dy, W, sei, ssi, lA, lB):
    return lora_ops.scatter2scatter_lora_dX(
        DY=dy,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=1,
        lora_A=lA,
        lora_B=lB,
        scaling=2.0,
        dy_grouped=True,
        dx_grouped=False,
    )


def _call_bwd(dy, gx, lA, lB, eo, num_experts):
    return lora_ops.group_bwd_lora(
        DY=dy,
        X=gx,
        lora_A=lA,
        lora_B=lB,
        expert_offsets=eo,
        E=num_experts,
        scaling=2.0,
    )


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="ScatterMoE LoRA kernel benchmark")
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        help="Model names or HF IDs (default: all builtins)",
    )
    parser.add_argument("--ranks", "-r", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--seq-len", "-T", type=int, default=2048)
    args = parser.parse_args()

    T = args.seq_len
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"T={T}, ranks={args.ranks}\n")

    if args.models:
        configs = [_resolve_config(m) for m in args.models]
    else:
        configs = list(BUILTIN_CONFIGS.items())

    for model_name, (num_experts, hidden, inter, top_k) in configs:
        print(f"{'=' * 70}")
        print(f"  {model_name}: E={num_experts}, H={hidden}, I={inter}, k={top_k}")
        print(f"{'=' * 70}")

        for R in args.ranks:
            for proj, K, N in [("gate_up", hidden, 2 * inter), ("down", inter, hidden)]:
                _clean()
                x, W, lA, lB, sei, ssi, eo, gx, dy = _setup(
                    num_experts, K, N, T, top_k, R
                )

                # Forward with LoRA (auto-dispatched: fused or split)
                dispatch = (
                    "split"
                    if (
                        num_experts <= lora_ops._SPLIT_LORA_FWD_MAX_EXPERTS
                        and K * N >= lora_ops._SPLIT_LORA_FWD_THRESHOLD
                    )
                    else "fused"
                )
                t_fwd = _bench(partial(_call_fwd, x, W, sei, ssi, top_k, lA, lB))
                t_base = _bench(partial(_call_base, x, W, sei, ssi, top_k))
                t_dx = _bench(partial(_call_dx, dy, W, sei, ssi, lA, lB))
                t_bwd = _bench(partial(_call_bwd, dy, gx, lA, lB, eo, num_experts))

                total = t_fwd + t_dx + t_bwd
                overhead = t_fwd / t_base - 1 if t_base > 0 else 0

                print(
                    f"  R={R:>2} {proj:<8}  "
                    f"fwd={t_fwd:>6.2f}ms [{dispatch}]  "
                    f"base={t_base:>6.2f}ms "
                    f"(+{overhead * 100:.0f}%)  "
                    f"dx={t_dx:>6.2f}ms  bwd={t_bwd:>6.2f}ms  "
                    f"total={total:>6.2f}ms"
                )

                # Full autograd fwd+bwd with memory measurement
                x_ag = x.clone().requires_grad_(True)
                lA_ag = lA.clone().requires_grad_(True)
                lB_ag = lB.clone().requires_grad_(True)

                def _run_autograd(
                    _x=x_ag,
                    _W=W,
                    _k=top_k,
                    _sei=sei,
                    _ssi=ssi,
                    _eo=eo,
                    _lA=lA_ag,
                    _lB=lB_ag,
                ):
                    out = ScatterMoELoRA.apply(
                        _x,
                        _W,
                        _k,
                        _sei,
                        _ssi,
                        _eo,
                        _lA,
                        _lB,
                        2.0,
                        None,
                        None,
                        False,
                        False,
                        True,
                        False,
                    )
                    out.sum().backward()
                    _x.grad = None
                    _lA.grad = None
                    _lB.grad = None

                t_full = _bench(_run_autograd)

                _clean()
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()
                _run_autograd()
                torch.cuda.synchronize()
                mem_peak = torch.cuda.max_memory_allocated() - mem_before

                print(
                    f"         full_fwd_bwd={t_full:>6.2f}ms  "
                    f"peak_delta={mem_peak / 1e6:>6.1f}MB"
                )

        print()


if __name__ == "__main__":
    main()
