"""Benchmark for selective_log_softmax Triton kernel vs original implementation.

Usage: CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_selective_logsoftmax.py
"""

import gc
import statistics

import torch

from axolotl.monkeypatch.trainer.utils import (
    selective_log_softmax,
    selective_log_softmax_original,
)

V = 151936  # Qwen vocab
WARMUP = 5
BENCH_ITERS = 20
MEM_ITERS = 10


def _clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.synchronize()


def profile_time(fn, args, n_iters=BENCH_ITERS):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def profile_memory(fn, args, n_iters=MEM_ITERS):
    for _ in range(WARMUP):
        out = fn(*args)
        del out
    torch.cuda.synchronize()

    peaks = []
    for _ in range(n_iters):
        _clean_gpu()
        base = torch.cuda.max_memory_allocated()
        out = fn(*args)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - base)
        del out
    return [p / 1e6 for p in peaks]


def fmt(values, unit=""):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:8.2f} ± {std:5.2f} {unit}  [min={min(values):.2f}, max={max(values):.2f}]"


def benchmark_forward():
    print("=" * 60)
    print(f"FORWARD BENCHMARK  (warmup={WARMUP}, time={BENCH_ITERS}, mem={MEM_ITERS})")
    print("=" * 60)

    configs = [
        (1, 2048),
        (1, 8192),
        (4, 4096),
        (8, 2048),
        (16, 2048),
        (16, 4096),
    ]

    for B, L in configs:
        mem_gb = B * L * V * 2 / 1e9
        if mem_gb > 28:
            print(f"\n  skip B={B}, L={L} ({mem_gb:.1f} GB)")
            continue

        N = B * L
        print(f"\n{'─' * 60}")
        print(f"B={B:2d}, L={L:5d}  ({N:6d} rows, logits {mem_gb:.2f} GB)")
        print(f"{'─' * 60}")

        torch.manual_seed(42)
        logits = torch.randn(B, L, V, device="cuda", dtype=torch.bfloat16)
        index = torch.randint(0, V, (B, L), device="cuda")

        t_orig = profile_time(selective_log_softmax_original, (logits, index))
        t_triton = profile_time(selective_log_softmax, (logits, index))
        orig_mean = statistics.mean(t_orig)
        triton_mean = statistics.mean(t_triton)

        print("  TIME (ms):")
        print(f"    original: {fmt(t_orig, 'ms')}")
        print(f"    triton:   {fmt(t_triton, 'ms')}")
        print(f"    speedup:  {orig_mean / triton_mean:.2f}x")

        m_orig = profile_memory(selective_log_softmax_original, (logits, index))
        m_triton = profile_memory(selective_log_softmax, (logits, index))
        orig_peak = statistics.mean(m_orig)
        triton_peak = statistics.mean(m_triton)

        print("  MEMORY (peak overhead):")
        print(f"    original: {fmt(m_orig, 'MB')}")
        print(f"    triton:   {fmt(m_triton, 'MB')}")
        print(f"    saved:    {orig_peak - triton_peak:.1f} MB")

        del logits, index
        _clean_gpu()


def benchmark_backward():
    print("\n" + "=" * 60)
    print(f"FWD+BWD BENCHMARK  (warmup={WARMUP}, time={BENCH_ITERS}, mem={MEM_ITERS})")
    print("=" * 60)

    configs = [
        (1, 2048),
        (1, 8192),
        (4, 4096),
        (8, 2048),
        (16, 2048),
        (16, 4096),
    ]

    def fwd_bwd_original(logits, index):
        logits.grad = None
        out = selective_log_softmax_original(logits, index)
        out.sum().backward()

    def fwd_bwd_triton(logits, index):
        logits.grad = None
        out = selective_log_softmax(logits, index)
        out.sum().backward()

    for B, L in configs:
        mem_gb = B * L * V * 2 / 1e9
        if mem_gb > 20:
            print(f"\n  skip B={B}, L={L} ({mem_gb:.1f} GB, need room for grads)")
            continue

        N = B * L
        print(f"\n{'─' * 60}")
        print(f"B={B:2d}, L={L:5d}  ({N:6d} rows, logits {mem_gb:.2f} GB)")
        print(f"{'─' * 60}")

        torch.manual_seed(42)
        logits_orig = torch.randn(
            B, L, V, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        logits_tri = logits_orig.detach().clone().requires_grad_(True)
        index = torch.randint(0, V, (B, L), device="cuda")

        t_orig = profile_time(fwd_bwd_original, (logits_orig, index))
        t_triton = profile_time(fwd_bwd_triton, (logits_tri, index))
        orig_mean = statistics.mean(t_orig)
        triton_mean = statistics.mean(t_triton)

        print("  FWD+BWD TIME (ms):")
        print(f"    original: {fmt(t_orig, 'ms')}")
        print(f"    triton:   {fmt(t_triton, 'ms')}")
        print(f"    speedup:  {orig_mean / triton_mean:.2f}x")

        m_orig = profile_memory(fwd_bwd_original, (logits_orig, index))
        m_triton = profile_memory(fwd_bwd_triton, (logits_tri, index))
        orig_peak = statistics.mean(m_orig)
        triton_peak = statistics.mean(m_triton)

        print("  FWD+BWD MEMORY (peak overhead):")
        print(f"    original: {fmt(m_orig, 'MB')}")
        print(f"    triton:   {fmt(m_triton, 'MB')}")
        print(f"    saved:    {orig_peak - triton_peak:.1f} MB")

        del logits_orig, logits_tri, index
        _clean_gpu()


if __name__ == "__main__":
    benchmark_forward()
    benchmark_backward()
