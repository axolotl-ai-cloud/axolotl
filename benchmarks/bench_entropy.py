"""Benchmark for entropy_from_logits Triton kernel vs original chunked implementation.

Usage: CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_entropy.py
"""

import gc
import statistics

import torch
import torch.nn.functional as F

from axolotl.monkeypatch.trainer.utils import entropy_from_logits

V = 151936  # Qwen vocab
WARMUP = 5
BENCH_ITERS = 20
MEM_ITERS = 10


def entropy_from_logits_original(logits: torch.Tensor, chunk_size: int = 128):
    """Original chunked implementation (reference)."""
    original_shape = logits.shape[:-1]
    num_classes = logits.shape[-1]
    flat_logits = logits.reshape(-1, num_classes)
    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)
    return torch.cat(entropies, dim=0).reshape(original_shape)


def _clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.synchronize()


def profile_time(fn, logits, n_iters=BENCH_ITERS):
    for _ in range(WARMUP):
        out = fn(logits, chunk_size=128)
        del out
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(logits, chunk_size=128)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
        del out
    return times


def profile_memory(fn, logits, n_iters=MEM_ITERS):
    for _ in range(WARMUP):
        out = fn(logits, chunk_size=128)
        del out
    torch.cuda.synchronize()

    peaks = []
    for _ in range(n_iters):
        _clean_gpu()
        base = torch.cuda.max_memory_allocated()
        out = fn(logits, chunk_size=128)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - base)
        del out
    return [p / 1e6 for p in peaks]


def fmt(values, unit=""):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:8.2f} ± {std:5.2f} {unit}  [min={min(values):.2f}, max={max(values):.2f}]"


def benchmark_contiguous():
    print("=" * 60)
    print(
        f"CONTIGUOUS BENCHMARK  (warmup={WARMUP}, time={BENCH_ITERS}, mem={MEM_ITERS})"
    )
    print("=" * 60)

    configs = [
        (1, 2048),
        (1, 8192),
        (1, 16384),
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

        t_orig = profile_time(entropy_from_logits_original, logits)
        t_triton = profile_time(entropy_from_logits, logits)
        orig_mean = statistics.mean(t_orig)
        triton_mean = statistics.mean(t_triton)

        print("  TIME (ms):")
        print(f"    original: {fmt(t_orig, 'ms')}")
        print(f"    triton:   {fmt(t_triton, 'ms')}")
        print(f"    speedup:  {orig_mean / triton_mean:.2f}x")

        m_orig = profile_memory(entropy_from_logits_original, logits)
        m_triton = profile_memory(entropy_from_logits, logits)
        orig_peak = statistics.mean(m_orig)
        triton_peak = statistics.mean(m_triton)

        print("  MEMORY (peak overhead):")
        print(f"    original: {fmt(m_orig, 'MB')}")
        print(f"    triton:   {fmt(m_triton, 'MB')}")
        print(f"    saved:    {orig_peak - triton_peak:.1f} MB")

        del logits
        _clean_gpu()


def benchmark_noncontiguous():
    print("\n" + "=" * 60)
    print(
        f"NON-CONTIGUOUS BENCHMARK  (warmup={WARMUP}, time={BENCH_ITERS}, mem={MEM_ITERS})"
    )
    print("=" * 60)

    configs = [
        (4, 2048, "transpose"),
        (4, 8192, "transpose"),
        (8, 2048, "transpose"),
        (4, 4096, "slice_batch"),
    ]

    for B, L, method in configs:
        torch.manual_seed(42)

        if method == "transpose":
            raw = torch.randn(L, B, V, device="cuda", dtype=torch.bfloat16)
            logits_nc = raw.transpose(0, 1)
            raw_gb = L * B * V * 2 / 1e9
        elif method == "slice_batch":
            raw = torch.randn(B * 2, L, V, device="cuda", dtype=torch.bfloat16)
            logits_nc = raw[::2]
            raw_gb = B * 2 * L * V * 2 / 1e9
        else:
            continue

        if raw_gb > 28:
            print(f"\n  skip B={B}, L={L}, {method} ({raw_gb:.1f} GB)")
            del raw, logits_nc
            torch.cuda.empty_cache()
            continue

        N = B * L
        print(f"\n{'─' * 60}")
        print(f"B={B}, L={L}  {method}  ({N} rows, raw {raw_gb:.2f} GB)")
        print(f"{'─' * 60}")

        def original_with_copy(logits, chunk_size=128):
            return entropy_from_logits_original(
                logits.contiguous(), chunk_size=chunk_size
            )

        t_orig = profile_time(original_with_copy, logits_nc)
        t_triton = profile_time(entropy_from_logits, logits_nc)
        orig_mean = statistics.mean(t_orig)
        triton_mean = statistics.mean(t_triton)

        print("  TIME (ms):")
        print(f"    orig+copy:     {fmt(t_orig, 'ms')}")
        print(f"    triton-strided:{fmt(t_triton, 'ms')}")
        print(f"    speedup:       {orig_mean / triton_mean:.2f}x")

        m_orig = profile_memory(original_with_copy, logits_nc)
        m_triton = profile_memory(entropy_from_logits, logits_nc)
        orig_peak = statistics.mean(m_orig)
        triton_peak = statistics.mean(m_triton)

        print("  MEMORY (peak overhead):")
        print(f"    orig+copy:     {fmt(m_orig, 'MB')}")
        print(f"    triton-strided:{fmt(m_triton, 'MB')}")
        print(f"    saved:         {orig_peak - triton_peak:.1f} MB")

        del raw, logits_nc
        _clean_gpu()


if __name__ == "__main__":
    benchmark_contiguous()
    benchmark_noncontiguous()
