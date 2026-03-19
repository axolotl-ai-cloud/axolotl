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
import statistics
import time

import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.kernels import (
    ops as base_ops,
    lora_ops,
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
    "Qwen3.5-35B-A3B": (256, 2048, 512, 8),    # E, H, I, k
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

    # Try HuggingFace AutoConfig
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(spec, trust_remote_code=True)
    if callable(getattr(hf_cfg, "get_text_config", None)):
        tc = hf_cfg.get_text_config()
        if hasattr(tc, "model_type") and tc.model_type != hf_cfg.model_type:
            hf_cfg = tc
    H = hf_cfg.hidden_size
    I = getattr(hf_cfg, "moe_intermediate_size", None) or hf_cfg.intermediate_size
    E = (getattr(hf_cfg, "num_experts", None)
         or getattr(hf_cfg, "num_local_experts", None)
         or getattr(hf_cfg, "n_routed_experts", None))
    k = (getattr(hf_cfg, "num_experts_per_tok", None)
         or getattr(hf_cfg, "num_experts_per_token", None) or 2)
    name = spec.split("/")[-1]
    return name, (E, H, I, k)


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
    return statistics.median(times)


def _setup(E, K, N, T, top_k, R):
    torch.manual_seed(42)
    x = torch.randn(T, K, device=DEVICE, dtype=DTYPE)
    W = torch.randn(E, K, N, device=DEVICE, dtype=DTYPE) * 0.02
    lora_A = torch.randn(R * E, K, device=DEVICE, dtype=DTYPE) * 0.01
    lora_B = torch.randn(N, R * E, device=DEVICE, dtype=DTYPE) * 0.01
    logits = torch.randn(T, E, device=DEVICE)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    sei, ssi, eo = flatten_sort_count(top_idx, E)
    gx = base_ops.group(x, ssi, fan_out=top_k)
    dy = torch.randn(gx.size(0), N, device=DEVICE, dtype=DTYPE)
    return x, W, lora_A, lora_B, sei, ssi, eo, gx, dy


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ScatterMoE LoRA kernel benchmark")
    parser.add_argument("--models", "-m", nargs="+",
                        help="Model names or HF IDs (default: all builtins)")
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
        configs = [(n, c) for n, c in configs]

    for model_name, (E, H, I, k) in configs:
        print(f"{'=' * 70}")
        print(f"  {model_name}: E={E}, H={H}, I={I}, k={k}")
        print(f"{'=' * 70}")

        for R in args.ranks:
            for proj, K, N in [("gate_up", H, 2 * I), ("down", I, H)]:
                _clean()
                x, W, lA, lB, sei, ssi, eo, gx, dy = _setup(E, K, N, T, k, R)

                # Forward with LoRA
                t_fwd = _bench(lambda: lora_ops.scatter2scatter_lora(
                    X=x, W=W, sorted_expert_idxs=sei, sorted_scattered_idxs=ssi,
                    k=k, lora_A=lA, lora_B=lB, scaling=2.0,
                ))

                # Forward without LoRA (base)
                t_base = _bench(lambda: base_ops.scatter2scatter(
                    X=x, W=W, sorted_expert_idxs=sei, sorted_scattered_idxs=ssi, k=k,
                ))

                # Backward dX
                t_dx = _bench(lambda: lora_ops.scatter2scatter_lora_dX(
                    DY=dy, W=W, sorted_expert_idxs=sei, sorted_scattered_idxs=ssi,
                    k=1, lora_A=lA, lora_B=lB, scaling=2.0,
                    dy_grouped=True, dx_grouped=False,
                ))

                # Backward dA/dB
                t_bwd = _bench(lambda: lora_ops.group_bwd_lora(
                    DY=dy, X=gx, lora_A=lA, lora_B=lB,
                    expert_offsets=eo, E=E, scaling=2.0,
                ))

                total = t_fwd + t_dx + t_bwd
                overhead = t_fwd / t_base - 1 if t_base > 0 else 0

                print(f"  R={R:>2} {proj:<8}  "
                      f"fwd={t_fwd:>6.2f}ms  base={t_base:>6.2f}ms "
                      f"(+{overhead*100:.0f}%)  "
                      f"dx={t_dx:>6.2f}ms  bwd={t_bwd:>6.2f}ms  "
                      f"total={total:>6.2f}ms")

                # Full autograd fwd+bwd
                x_ag = x.clone().requires_grad_(True)
                lA_ag = lA.clone().requires_grad_(True)
                lB_ag = lB.clone().requires_grad_(True)

                def _run_autograd():
                    out = ScatterMoELoRA.apply(
                        x_ag, W, k, sei, ssi, eo,
                        lA_ag, lB_ag, 2.0,
                        None, None, False, False, True, False,
                    )
                    out.sum().backward()
                    x_ag.grad = None
                    lA_ag.grad = None
                    lB_ag.grad = None

                t_full = _bench(_run_autograd)
                print(f"         full_fwd_bwd={t_full:>6.2f}ms")

        print()


if __name__ == "__main__":
    main()
