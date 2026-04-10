#!/usr/bin/env python3
"""
Axolotl Training Profiler Analyzer
===================================

Analyzes PyTorch profiler output from axolotl training runs (profiler_steps config).
Produces breakdowns by CUDA kernel category, identifies bottlenecks, and optionally
compares two traces (e.g. before/after optimization).

Supports both:
  - profiler_trace.json (torch.profiler Chrome trace -- timing analysis)
  - snapshot.pickle (torch.cuda.memory._snapshot -- memory analysis)

Usage:
  # Analyze a single trace
  python analyze_profile.py outputs/qwen35_moe_profile/

  # Compare before vs after
  python analyze_profile.py outputs/before/ --compare outputs/after/

  # Include step 0 (warmup/compilation) in analysis
  python analyze_profile.py outputs/run/ --include-warmup

  # Memory-only analysis
  python analyze_profile.py outputs/run/ --memory-only

  # Quick mode (first 2M events only, for large traces)
  python analyze_profile.py outputs/run/ --quick
"""

import argparse
import json
import pickle  # nosec B403
import time
from collections import defaultdict
from pathlib import Path

# ---- Kernel categorization ------------------------------------------------

KERNEL_CATEGORIES = [
    # ScatterMoE -- ordered most specific first
    (
        "ScatterMoE bwd LoRA (split)",
        ["_group_bwd_lora_split", "_group_bwd_da", "_group_bwd_db"],
    ),
    ("ScatterMoE bwd LoRA (fused)", ["_group_bwd_lora"]),
    ("ScatterMoE bwd dX", ["_scatter2scatter_lora_dx"]),
    ("ScatterMoE fwd", ["_scatter2scatter_lora"]),
    # Quantization
    ("BnB Dequantization", ["dequantize", "kDequantizeBlockwise"]),
    # Attention
    ("Flash Attention", ["flash", "fmha"]),
    # Loss
    ("CCE Loss", ["_cce_"]),
    # LoRA fused kernels (autograd.Function based)
    ("LoRA QKV Kernel", ["lora_qkv"]),
    ("LoRA O Kernel", ["lora_o"]),
    ("LoRA MLP Kernel", ["lora_mlp"]),
    # LoRA activation kernels (SwiGLU/GEGLU Triton kernels)
    ("LoRA Activation (SwiGLU/GEGLU)", ["swiglu", "geglu"]),
    # DoRA weight norm
    ("DoRA Weight Norm", ["linalg_norm", "dora_scale"]),
    # Compute
    ("GEMM/CUTLASS", ["cutlass", "gemm", "gemv", "cublas"]),
    ("Triton (norms etc)", ["triton"]),
    ("Conv1d", ["conv1d", "causal_conv"]),
    # Optimizer
    ("Optimizer", ["adam", "optim"]),
    # Dtype conversion (fp32→bf16 LoRA matrix casts etc)
    ("Dtype Conversion", ["_to_copy", "to_copy"]),
    # Memory
    ("Elementwise/Fill", ["fill", "elementwise", "cast", "copy_kernel"]),
    ("Memory ops", ["memcpy", "memset"]),
    # Routing
    ("TopK/Sort", ["topk", "sort"]),
    ("Index/Gather/Scatter", ["index", "gather", "scatter"]),
]

# Categories to keep when pre-filtering during streaming load
_KEEP_CATS = {"kernel", "gpu_memcpy", "cpu_op", "python_function", "ac2g", "Runtime"}


def categorize_kernel(name):
    nl = name.lower()
    for cat_name, patterns in KERNEL_CATEGORIES:
        if any(p in nl for p in patterns):
            return cat_name
    return "Other"


# ---- Trace loading ---------------------------------------------------------


def _try_ijson_load(trace_file, quick=False, max_events=2_000_000):
    """Stream-parse trace JSON with ijson. Returns filtered events list."""
    try:
        import ijson
    except ImportError:
        return None

    events = []
    count = 0
    with open(trace_file, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            count += 1
            cat = ev.get("cat", "")
            # Pre-filter: only keep categories we care about
            if cat in _KEEP_CATS or ev.get("ph") == "M":
                events.append(ev)
            if quick and count >= max_events:
                print(f"  --quick: stopped after {count:,} events")
                break
    return events, count


def load_trace(path, quick=False):
    trace_file = (
        Path(path) / "profiler_trace.json" if Path(path).is_dir() else Path(path)
    )
    if not trace_file.exists():
        return None
    size_gb = trace_file.stat().st_size / 1e9
    print(f"Loading {trace_file.name} ({size_gb:.1f} GB)...")
    t0 = time.monotonic()

    # Try streaming parser first for large files
    if size_gb > 0.5:
        result = _try_ijson_load(trace_file, quick=quick)
        if result is not None:
            events, total_count = result
            elapsed = time.monotonic() - t0
            print(
                f"  {total_count:,} total events, {len(events):,} kept "
                f"(streamed in {elapsed:.1f}s)"
            )
            return events

    # Fallback: standard json.load
    with open(trace_file) as f:
        data = json.load(f)
    all_events = data.get("traceEvents", [])
    elapsed = time.monotonic() - t0

    if quick:
        all_events = all_events[:2_000_000]
        print(f"  --quick: limited to first {len(all_events):,} events")

    # Pre-filter
    events = [
        ev
        for ev in all_events
        if ev.get("cat", "") in _KEEP_CATS or ev.get("ph") == "M"
    ]
    print(
        f"  {len(all_events):,} total events, {len(events):,} kept "
        f"(loaded in {elapsed:.1f}s)"
    )
    return events


# ---- Trace analysis -------------------------------------------------------


def _estimate_n_steps(cuda_events):
    """Estimate the number of training steps from CUDA event timestamps.

    Detects step boundaries by looking for large gaps (>2x median gap) in the
    sorted timestamp sequence of CUDA kernels.
    """
    if len(cuda_events) < 100:
        return 1
    timestamps = sorted(float(ev.get("ts", 0)) for ev in cuda_events)
    # Compute gaps between consecutive events
    gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    if not gaps:
        return 1
    median_gap = sorted(gaps)[len(gaps) // 2]
    # A step boundary has a gap much larger than the median inter-kernel gap
    threshold = max(median_gap * 50, 100_000)  # at least 100ms
    n_boundaries = sum(1 for g in gaps if g > threshold)
    return max(n_boundaries + 1, 1)


def analyze_trace(events, skip_warmup=True):
    cuda_events = [
        ev
        for ev in events
        if ev.get("ph") == "X" and ev.get("cat") in ("kernel", "gpu_memcpy")
    ]
    if not cuda_events:
        print("  No CUDA kernel events found!")
        return None

    cutoff_ts = None

    if skip_warmup and len(cuda_events) > 1000:
        timestamps = sorted(set(float(ev.get("ts", 0)) for ev in cuda_events))
        min_ts, max_ts = timestamps[0], timestamps[-1]
        total_span = max_ts - min_ts

        # Step 0 is warmup (Triton compilation + autotune). It's typically
        # the slowest step by far. Use 45% of wall-clock as cutoff -- step 0
        # usually takes >50% of total time when it includes compilation.
        cutoff_ts = min_ts + total_span * 0.45
        before = len(cuda_events)
        cuda_events = [ev for ev in cuda_events if float(ev.get("ts", 0)) > cutoff_ts]
        print(f"  Excluding step 0 (warmup): {before:,} -> {len(cuda_events):,} events")

    n_steps_profiled = _estimate_n_steps(cuda_events)

    # Aggregate by kernel (cast to float to handle ijson Decimal values)
    kernel_stats = defaultdict(lambda: {"total_us": 0.0, "count": 0, "max_us": 0.0})
    for ev in cuda_events:
        name = ev.get("name", "unknown")
        dur = float(ev.get("dur", 0))
        kernel_stats[name]["total_us"] += dur
        kernel_stats[name]["count"] += 1
        kernel_stats[name]["max_us"] = max(kernel_stats[name]["max_us"], dur)

    total_cuda = sum(v["total_us"] for v in kernel_stats.values())

    # Group by category
    cat_stats = defaultdict(lambda: {"total_us": 0, "count": 0})
    for name, info in kernel_stats.items():
        cat = categorize_kernel(name)
        cat_stats[cat]["total_us"] += info["total_us"]
        cat_stats[cat]["count"] += info["count"]

    # Fill/zero_ analysis: find FillFunctor kernels and group by tensor size
    fill_by_size = defaultdict(lambda: {"total_us": 0, "count": 0})
    for ev in cuda_events:
        name = ev.get("name", "")
        if "FillFunctor" not in name and "fill" not in name.lower():
            continue
        # Extract Input Dims from args if present
        args = ev.get("args", {})
        input_dims = args.get("Input Dims", args.get("input_dims", "unknown"))
        if isinstance(input_dims, list):
            input_dims = str(input_dims)
        dur = float(ev.get("dur", 0))
        fill_by_size[input_dims]["total_us"] += dur
        fill_by_size[input_dims]["count"] += 1

    # CPU op analysis for wall-clock estimation (apply same warmup cutoff)
    cpu_ops = [
        ev
        for ev in events
        if ev.get("ph") == "X" and ev.get("cat") in ("cpu_op", "python_function")
    ]
    if cutoff_ts is not None:
        cpu_ops = [ev for ev in cpu_ops if float(ev.get("ts", 0)) > cutoff_ts]
    wall_clock_us = 0
    if cpu_ops:
        ts_sorted = sorted(cpu_ops, key=lambda e: float(e.get("ts", 0)))
        min_cpu_ts = float(ts_sorted[0].get("ts", 0))
        max_cpu_end = max(
            float(e.get("ts", 0)) + float(e.get("dur", 0)) for e in ts_sorted
        )
        wall_clock_us = max_cpu_end - min_cpu_ts

    return {
        "total_cuda_us": total_cuda,
        "n_steps": n_steps_profiled,
        "categories": dict(cat_stats),
        "kernel_stats": dict(kernel_stats),
        "n_events": len(cuda_events),
        "fill_by_size": dict(fill_by_size),
        "wall_clock_us": wall_clock_us,
    }


def print_trace_analysis(result, label=""):
    total = result["total_cuda_us"]
    n = result["n_steps"]

    if label:
        print(f"\n{'=' * 75}")
        print(f"  {label}")
        print(f"{'=' * 75}")

    print(
        f"\n  CUDA kernel time: {total / 1e6:.2f}s over {n} steps "
        f"(~{total / n / 1e6:.2f}s/step)"
    )

    if result.get("wall_clock_us"):
        wc = result["wall_clock_us"]
        print(
            f"  Wall clock span:  {wc / 1e6:.2f}s over {n} steps "
            f"(~{wc / n / 1e6:.2f}s/step)"
        )

    print(f"\n  {'Category':<40} {'Total':>9} {'%':>6} {'Count':>7} {'Per step':>9}")
    print(f"  {'-' * 75}")
    for cat, info in sorted(
        result["categories"].items(), key=lambda x: x[1]["total_us"], reverse=True
    ):
        pct = info["total_us"] / total * 100
        ps = info["total_us"] / n / 1000
        print(
            f"  {cat:<40} {info['total_us'] / 1000:>8.1f}ms {pct:>5.1f}% "
            f"{info['count']:>7} {ps:>7.1f}ms"
        )

    print("\n  Top 15 individual kernels:")
    for name, info in sorted(
        result["kernel_stats"].items(), key=lambda x: x[1]["total_us"], reverse=True
    )[:15]:
        pct = info["total_us"] / total * 100
        avg = info["total_us"] / info["count"] / 1000
        print(
            f"    {name[:62]:<62} {info['total_us'] / 1000:>7.1f}ms "
            f"({pct:>4.1f}%) x{info['count']:<5} avg={avg:.3f}ms"
        )

    # Fill/zero_ breakdown
    fill_data = result.get("fill_by_size", {})
    if fill_data:
        total_fill_us = sum(v["total_us"] for v in fill_data.values())
        if total_fill_us > 0:
            print(
                f"\n  Fill/zero_ kernel breakdown by tensor size "
                f"(total: {total_fill_us / 1000:.1f}ms, "
                f"{total_fill_us / total * 100:.1f}% of CUDA time):"
            )
            print(f"    {'Input Dims':<50} {'Time':>9} {'Count':>7}")
            print(f"    {'-' * 70}")
            for dims, info in sorted(
                fill_data.items(), key=lambda x: x[1]["total_us"], reverse=True
            )[:10]:
                dims_str = str(dims)[:50]
                print(
                    f"    {dims_str:<50} {info['total_us'] / 1000:>7.1f}ms "
                    f"{info['count']:>7}"
                )


def print_summary(result, mem_result=None):
    """Print optimization recommendations and summary."""
    total = result["total_cuda_us"]
    n = result["n_steps"]

    print(f"\n{'=' * 75}")
    print("  SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 75}")

    # Estimated per-step wall clock
    wc = result.get("wall_clock_us", 0)
    if wc > 0:
        print(f"\n  Estimated per-step wall clock: {wc / n / 1e6:.2f}s")
    else:
        print(f"\n  Estimated per-step CUDA time:  {total / n / 1e6:.2f}s")

    # Memory utilization
    if mem_result:
        reserved = mem_result["total_reserved"]
        allocated = mem_result["total_allocated"]
        if reserved > 0:
            util_pct = allocated / reserved * 100
            print(
                f"  Memory utilization: {util_pct:.1f}% "
                f"({allocated / 1e9:.2f} / {reserved / 1e9:.2f} GB)"
            )

    # Build recommendations
    recommendations = []

    cats_sorted = sorted(
        result["categories"].items(), key=lambda x: x[1]["total_us"], reverse=True
    )

    # Check top category
    if cats_sorted:
        top_cat, top_info = cats_sorted[0]
        top_pct = top_info["total_us"] / total * 100
        if top_pct > 30:
            if "GEMM" in top_cat or "CUTLASS" in top_cat:
                recommendations.append(
                    f"GEMM/matmul dominates ({top_pct:.0f}%). "
                    f"Consider FP8 training, LoRA (fewer params), "
                    f"or smaller batch size to reduce compute."
                )
            elif "Attention" in top_cat:
                recommendations.append(
                    f"Attention dominates ({top_pct:.0f}%). "
                    f"Ensure FlashAttention v2/v3 is active. "
                    f"Consider reducing sequence length or using sliding window."
                )
            elif "Fill" in top_cat or "Elementwise" in top_cat:
                recommendations.append(
                    f"Elementwise/Fill ops dominate ({top_pct:.0f}%). "
                    f"Enable kernel fusion (Liger, torch.compile) to reduce "
                    f"memory-bound elementwise operations."
                )
            elif "ScatterMoE" in top_cat:
                recommendations.append(
                    f"MoE routing/scatter dominates ({top_pct:.0f}%). "
                    f"Check expert count and capacity factor. "
                    f"Verify ScatterMoE kernels are using optimal block sizes."
                )
            else:
                recommendations.append(
                    f"'{top_cat}' dominates ({top_pct:.0f}%). "
                    f"Focus optimization efforts here first."
                )

    # Check memory ops
    for cat, info in cats_sorted:
        pct = info["total_us"] / total * 100
        if "Memory" in cat and pct > 10:
            recommendations.append(
                f"Memory ops are {pct:.0f}% of CUDA time. "
                f"Consider gradient checkpointing, reducing activation "
                f"recomputation, or pinned memory for data loading."
            )
            break

    # Check fill overhead
    fill_data = result.get("fill_by_size", {})
    total_fill_us = sum(v["total_us"] for v in fill_data.values())
    fill_pct = total_fill_us / total * 100 if total > 0 else 0
    if fill_pct > 5:
        recommendations.append(
            f"Fill/zero_ kernels consume {fill_pct:.1f}% of CUDA time. "
            f"Large zero-fills suggest excessive tensor allocation. "
            f"Consider reusing buffers or lazy initialization."
        )

    # Check fragmentation
    if mem_result and mem_result.get("fragmentation_pct", 0) > 20:
        frag = mem_result["fragmentation_pct"]
        recommendations.append(
            f"Memory fragmentation is {frag:.0f}%. "
            f"Use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"or max_split_size_mb to reduce fragmentation."
        )

    # GPU utilization hint from wall clock vs CUDA time
    if wc > 0 and total > 0:
        gpu_util = total / wc * 100
        if gpu_util < 50:
            recommendations.append(
                f"GPU utilization is ~{gpu_util:.0f}% (CUDA time vs wall clock). "
                f"CPU-side bottleneck likely. Profile data loading, "
                f"reward computation, or weight sync overhead."
            )

    # Print top 3
    print("\n  Top optimization recommendations:")
    if not recommendations:
        print("    No major issues detected.")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"    {i}. {rec}")

    print()


def compare_traces(before, after):
    print(f"\n{'=' * 75}")
    print("  COMPARISON")
    print(f"{'=' * 75}")

    tb = before["total_cuda_us"]
    ta = after["total_cuda_us"]
    nb = before["n_steps"]
    na = after["n_steps"]

    print(
        f"\n  Per-step CUDA time: {tb / nb / 1e6:.2f}s -> {ta / na / 1e6:.2f}s "
        f"({(ta / na - tb / nb) / (tb / nb) * 100:+.1f}%)"
    )

    all_cats = sorted(
        set(list(before["categories"]) + list(after["categories"])),
        key=lambda c: before["categories"].get(c, {"total_us": 0})["total_us"],
        reverse=True,
    )

    print(
        f"\n  {'Category':<35} {'Before/step':>11} {'After/step':>11} "
        f"{'Delta':>9} {'Speedup':>8}"
    )
    print(f"  {'-' * 78}")
    for cat in all_cats:
        b = before["categories"].get(cat, {"total_us": 0})["total_us"] / nb
        a = after["categories"].get(cat, {"total_us": 0})["total_us"] / na
        delta = a - b
        speedup = b / a if a > 0 else float("inf")
        if b > tb / nb * 0.003 or a > ta / na * 0.003:
            print(
                f"  {cat:<35} {b / 1000:>9.1f}ms {a / 1000:>9.1f}ms "
                f"{delta / 1000:>+8.1f}ms {speedup:>7.2f}x"
            )


# ---- Allocation churn attribution -----------------------------------------


def _short_path(p):
    """Shorten a file path for display."""
    if "/site-packages/" in p:
        return "..." + p.split("/site-packages/")[1]
    if "/axolotl/src/" in p:
        return "axolotl/" + p.split("/axolotl/src/")[1]
    if "/axolotl/" in p:
        return "axolotl/" + p.split("/axolotl/")[1]
    if len(p) > 60:
        parts = p.split("/")
        return ".../" + "/".join(parts[-3:])
    return p


def _get_top_python_frame(frames):
    """Get the first meaningful Python frame from a trace frame list.

    Skips internal torch/cuda/autograd frames to find the user-level code
    that triggered the allocation.  Falls back to first Python frame.
    """
    skip_patterns = [
        "torch/autograd/",
        "torch/utils/checkpoint",
        "torch/nn/modules/module.py",
        "torch/_dynamo/",
        "torch/_compile",
        "torch/_ops",
        "torch/_functorch/",
        "torch/_inductor/",
        "torch/library",
        "torch/cuda",
        "torch/_C",
        "<frozen",
        "<unknown>",
        "fire/core.py",
        "runpy",
    ]
    py_frames = [
        f
        for f in frames
        if isinstance(f, dict) and f.get("filename", "").endswith(".py")
    ]
    for fr in py_frames:
        fname = fr.get("filename", "")
        if not any(p in fname for p in skip_patterns):
            return fr
    return py_frames[0] if py_frames else None


def _categorize_source(fname, funcname):
    """Assign a human-readable category to an allocation source."""
    if "checkpoint" in fname.lower() or "backward" in funcname:
        return "Gradient checkpoint recompute"
    if "bitsandbytes" in fname:
        return "BnB dequantization"
    if "scattermoe" in fname or "scatter" in funcname.lower():
        return "ScatterMoE LoRA"
    if "adam" in funcname.lower() or "optim" in fname.lower() or "inductor" in fname:
        return "Optimizer"
    if "norm" in funcname.lower():
        return "LayerNorm"
    if "fla/" in fname or "gated_delta" in fname:
        return "FLA linear attention"
    return "Other"


def analyze_allocation_churn(snapshot):
    """Group allocation churn by Python source attribution.

    For each major churn size, identifies which Python code creates
    the tensors (gradient checkpointing, BnB dequant, LoRA conversion, etc).
    """
    traces = snapshot.get("device_traces", [[]])[0]
    if not traces:
        return None

    # Find the top churn sizes
    size_counts = defaultdict(int)
    for ev in traces:
        if ev.get("action") == "alloc":
            size_counts[ev["size"]] += 1

    # Top 5 sizes by total churn (count × size)
    top_sizes = sorted(size_counts.items(), key=lambda x: x[0] * x[1], reverse=True)[:5]

    results = {}
    for target_sz, total_count in top_sizes:
        mb = target_sz / 1e6
        if mb < 1.0:
            continue

        frame_groups = defaultdict(lambda: {"count": 0})
        for ev in traces:
            if ev.get("action") == "alloc" and ev.get("size") == target_sz:
                top = _get_top_python_frame(ev.get("frames", []))
                if top:
                    key = (top["filename"], top["name"], top.get("line", 0))
                else:
                    key = ("<unknown>", "<no attribution>", 0)
                frame_groups[key]["count"] += 1

        # Group into categories
        categories = defaultdict(lambda: {"count": 0, "sources": []})
        for (fname, funcname, lineno), info in frame_groups.items():
            cat = _categorize_source(fname, funcname)
            categories[cat]["count"] += info["count"]
            categories[cat]["sources"].append(
                (funcname, _short_path(fname), lineno, info["count"])
            )

        results[target_sz] = {
            "total_count": total_count,
            "total_churn_gb": total_count * target_sz / 1e9,
            "categories": dict(categories),
        }

    return results


def print_allocation_churn(results, label=""):
    if not results:
        return

    if label:
        print(f"\n{'=' * 75}")
        print(f"  {label}")
        print(f"{'=' * 75}")
    else:
        print("\n  ALLOCATION CHURN BY SOURCE")

    for target_sz in sorted(
        results, key=lambda s: results[s]["total_churn_gb"], reverse=True
    ):
        info = results[target_sz]
        mb = target_sz / 1e6
        print(
            f"\n  {mb:.1f}MB × {info['total_count']:,} = {info['total_churn_gb']:.0f}GB churn:"
        )
        for cat, cinfo in sorted(
            info["categories"].items(), key=lambda x: x[1]["count"], reverse=True
        ):
            pct = cinfo["count"] / info["total_count"] * 100
            print(f"    {pct:>4.0f}%  {cat} ({cinfo['count']} allocs)")
            for funcname, short, lineno, cnt in sorted(
                cinfo["sources"], key=lambda x: x[3], reverse=True
            )[:3]:
                print(f"           {funcname:35s} {short}:{lineno} (x{cnt})")


# ---- CPU overhead analysis ------------------------------------------------


def analyze_cpu_overhead(events):
    """Analyze memcpy, checkpoint recomputation, and GPU utilization from trace events."""
    cuda_total_us = 0
    cuda_count = 0
    memcpy_stats = defaultdict(lambda: {"dur_us": 0, "count": 0, "bytes": 0})
    checkpoint_us = 0
    checkpoint_count = 0
    min_ts = float("inf")
    max_ts_end = 0

    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur = float(ev.get("dur", 0))
        ts = float(ev.get("ts", 0))
        end = ts + dur
        if ts < min_ts:
            min_ts = ts
        if end > max_ts_end:
            max_ts_end = end

        if cat == "kernel":
            cuda_total_us += dur
            cuda_count += 1
        elif cat == "gpu_memcpy":
            if "DtoH" in name:
                direction = "GPU→CPU (offload)"
            elif "HtoD" in name:
                direction = "CPU→GPU (reload)"
            elif "DtoD" in name:
                direction = "GPU→GPU"
            else:
                direction = name
            memcpy_stats[direction]["dur_us"] += dur
            memcpy_stats[direction]["count"] += 1
            nbytes = ev.get("args", {}).get("Bytes", 0)
            if nbytes:
                memcpy_stats[direction]["bytes"] += int(nbytes)
        elif cat in ("cpu_op", "python_function"):
            nl = name.lower()
            if "checkpoint" in nl or "recompute" in nl:
                checkpoint_us += dur
                checkpoint_count += 1

    wall_us = max_ts_end - min_ts if max_ts_end > min_ts else 0

    return {
        "wall_clock_us": wall_us,
        "cuda_total_us": cuda_total_us,
        "cuda_kernel_count": cuda_count,
        "memcpy_stats": dict(memcpy_stats),
        "checkpoint_us": checkpoint_us,
        "checkpoint_count": checkpoint_count,
    }


def print_cpu_overhead(result, n_steps=2, label=""):
    if not result:
        return
    if label:
        print(f"\n{'=' * 75}")
        print(f"  {label}")
        print(f"{'=' * 75}")
    else:
        print("\n  CPU OVERHEAD ANALYSIS")

    wall = result["wall_clock_us"]
    cuda = result["cuda_total_us"]
    count = result["cuda_kernel_count"]
    gpu_util = cuda / wall * 100 if wall > 0 else 0
    cpu_gap = wall - cuda

    print(
        f"\n  Wall clock:       {wall / 1e6:.2f}s (~{wall / n_steps / 1e6:.2f}s/step)"
    )
    print(f"  CUDA kernel time: {cuda / 1e6:.2f}s ({count:,} kernels)")
    print(f"  GPU utilization:  {gpu_util:.1f}%")
    print(f"  CPU overhead:     {cpu_gap / 1e6:.2f}s ({100 - gpu_util:.1f}%)")

    memcpy = result["memcpy_stats"]
    if memcpy:
        total_memcpy = sum(v["dur_us"] for v in memcpy.values())
        total_bytes = sum(v["bytes"] for v in memcpy.values())
        print(
            f"\n  Memory transfers: {total_memcpy / 1e6:.3f}s "
            f"({total_bytes / 1e9:.2f}GB, {total_memcpy / wall * 100:.1f}% of wall)"
        )
        for direction, info in sorted(
            memcpy.items(), key=lambda x: x[1]["dur_us"], reverse=True
        ):
            gb = info["bytes"] / 1e9
            print(
                f"    {direction:30s} {info['dur_us'] / 1e6:.3f}s  "
                f"x{info['count']:>5}  {gb:.2f}GB"
            )

    if result["checkpoint_count"] > 0:
        print(
            f"\n  Gradient checkpoint CPU ops: {result['checkpoint_us'] / 1e6:.3f}s "
            f"(x{result['checkpoint_count']})"
        )


# ---- Memory snapshot analysis ---------------------------------------------


def load_snapshot(path):
    """Load a PyTorch CUDA memory snapshot from a pickle file.

    WARNING: This uses pickle.load() which can execute arbitrary code.
    Only load snapshot files that you generated yourself from trusted
    training runs. Never load snapshots from untrusted sources.
    """
    snap_file = Path(path) / "snapshot.pickle" if Path(path).is_dir() else Path(path)
    if not snap_file.exists():
        return None
    print(f"Loading {snap_file.name} ({snap_file.stat().st_size / 1e6:.0f} MB)...")
    with open(snap_file, "rb") as f:
        return pickle.load(f)  # nosec B301


def _extract_python_frames(snapshot):
    """Extract Python source attribution from snapshot blocks with stacks='all'.

    The snapshot structure (when stacks='all') stores frames in:
      segments[i].blocks[j].history[k].frames = [(filename, lineno, name), ...]

    Returns a dict mapping (filename, function_name) -> {"bytes": int, "count": int}
    """
    source_allocs = defaultdict(lambda: {"bytes": 0, "count": 0})

    for seg in snapshot.get("segments", []):
        for block in seg.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            size = block.get("size", 0)
            history = block.get("history", [])
            if not history:
                continue

            # Use the most recent allocation history entry
            last_hist = history[-1]
            frames = last_hist.get("frames", [])

            # Find the first Python frame (skip C++ frames)
            # Frames are tuples: (filename, lineno, name)
            attributed = False
            for frame in frames:
                if not isinstance(frame, (list, tuple)) or len(frame) < 3:
                    continue
                filename, lineno, funcname = frame[0], frame[1], frame[2]
                # Skip internal torch/cuda frames to find user-level attribution
                fname_str = str(filename)
                if any(
                    skip in fname_str
                    for skip in [
                        "torch/cuda",
                        "torch/_C",
                        "torch/utils",
                        "cuda/memory.py",
                        "<unknown>",
                    ]
                ):
                    continue
                key = (fname_str, funcname, lineno)
                source_allocs[key]["bytes"] += size
                source_allocs[key]["count"] += 1
                attributed = True
                break

            # If no user frame found, use first available frame
            if not attributed and frames:
                frame = frames[0]
                if isinstance(frame, (list, tuple)) and len(frame) >= 3:
                    key = (str(frame[0]), str(frame[2]), frame[1])
                    source_allocs[key]["bytes"] += size
                    source_allocs[key]["count"] += 1

    return dict(source_allocs)


def _extract_source_file_summary(source_allocs):
    """Aggregate per-frame allocations to per-file level."""
    file_allocs = defaultdict(lambda: {"bytes": 0, "count": 0, "functions": set()})
    for (filename, funcname, _lineno), info in source_allocs.items():
        file_allocs[filename]["bytes"] += info["bytes"]
        file_allocs[filename]["count"] += info["count"]
        file_allocs[filename]["functions"].add(funcname)
    return dict(file_allocs)


def analyze_snapshot(snapshot):
    segments = snapshot.get("segments", [])
    total_reserved = sum(s.get("total_size", 0) for s in segments)
    total_allocated = sum(s.get("allocated_size", 0) for s in segments)

    # Active blocks
    active_blocks = []
    for seg in segments:
        for block in seg.get("blocks", []):
            if block.get("state") == "active_allocated":
                active_blocks.append(block.get("size", 0))

    # Allocation churn from trace
    trace = snapshot.get("device_traces", [[]])[0]
    size_counts = defaultdict(lambda: {"count": 0, "total": 0})
    for ev in trace:
        if ev.get("action") == "alloc":
            sz = ev.get("size", 0)
            size_counts[sz]["count"] += 1
            size_counts[sz]["total"] += sz

    # Python frame attribution
    source_allocs = _extract_python_frames(snapshot)
    file_summary = _extract_source_file_summary(source_allocs)

    return {
        "total_reserved": total_reserved,
        "total_allocated": total_allocated,
        "fragmentation_pct": (total_reserved - total_allocated) / total_reserved * 100
        if total_reserved > 0
        else 0,
        "n_segments": len(segments),
        "n_active_blocks": len(active_blocks),
        "active_bytes": sum(active_blocks),
        "largest_active": sorted(active_blocks, reverse=True)[:10],
        "alloc_churn": dict(
            sorted(size_counts.items(), key=lambda x: x[1]["total"], reverse=True)[:15]
        ),
        "n_trace_events": len(trace),
        "source_allocs": source_allocs,
        "file_summary": file_summary,
    }


def print_memory_analysis(result, label=""):
    if label:
        print(f"\n{'=' * 75}")
        print(f"  {label}")
        print(f"{'=' * 75}")

    reserved = result["total_reserved"]
    allocated = result["total_allocated"]

    print(f"\n  Reserved:    {reserved / 1e9:.2f} GB")
    print(f"  Allocated:   {allocated / 1e9:.2f} GB")
    print(f"  Utilization: {allocated / reserved * 100:.1f}%" if reserved > 0 else "")
    print(f"  Fragmentation: {result['fragmentation_pct']:.1f}%")
    print(
        f"  Segments: {result['n_segments']}, Active blocks: {result['n_active_blocks']}"
    )

    print("\n  Largest active allocations:")
    for sz in result["largest_active"]:
        print(f"    {sz / 1e6:>10.1f} MB")

    # Python source file attribution
    file_summary = result.get("file_summary", {})
    if file_summary:
        print("\n  Top allocations by source file:")
        print(f"    {'Source file':<55} {'Alloc':>10} {'Count':>7}")
        print(f"    {'-' * 75}")
        for fname, info in sorted(
            file_summary.items(), key=lambda x: x[1]["bytes"], reverse=True
        )[:15]:
            # Shorten path for display
            short = fname
            if len(short) > 55:
                parts = short.split("/")
                # Keep last 3 path components
                short = ".../" + "/".join(parts[-3:])
                if len(short) > 55:
                    short = short[:52] + "..."
            funcs = ", ".join(sorted(info["functions"])[:3])
            sz = info["bytes"]
            unit = "MB"
            val = sz / 1e6
            if val >= 1000:
                unit = "GB"
                val = sz / 1e9
            print(f"    {short:<55} {val:>8.1f}{unit} {info['count']:>7}")
            if funcs:
                print(f"      functions: {funcs[:70]}")

    # Top allocations by function (more granular)
    source_allocs = result.get("source_allocs", {})
    if source_allocs:
        print("\n  Top allocations by function (with line numbers):")
        print(f"    {'Function':<35} {'File:Line':<35} {'Size':>10}")
        print(f"    {'-' * 82}")
        for (fname, funcname, lineno), info in sorted(
            source_allocs.items(), key=lambda x: x[1]["bytes"], reverse=True
        )[:15]:
            # Shorten filename
            short_file = fname
            parts = short_file.split("/")
            if len(parts) > 2:
                short_file = "/".join(parts[-2:])
            loc = f"{short_file}:{lineno}"
            if len(loc) > 35:
                loc = "..." + loc[-32:]
            sz = info["bytes"]
            if sz >= 1e9:
                sz_str = f"{sz / 1e9:.2f}GB"
            else:
                sz_str = f"{sz / 1e6:.1f}MB"
            print(f"    {funcname:<35} {loc:<35} {sz_str:>10}")

    print("\n  Top allocation churn (alloc count x size):")
    print(f"    {'Size':>12} {'Count':>8} {'Total churned':>14}")
    print(f"    {'-' * 38}")
    for sz, info in result["alloc_churn"].items():
        if sz >= 1e6:
            print(
                f"    {sz / 1e6:>10.1f}MB {info['count']:>8} "
                f"{info['total'] / 1e9:>12.2f}GB"
            )


# ---- Peak memory timeline from trace events --------------------------------


def analyze_peak_memory(snapshot):
    """Walk through device_traces chronologically to find peak concurrent memory usage.

    The snapshot's segment data only captures end-of-step state.  The device_traces
    record every alloc/free, letting us reconstruct peak usage and identify which
    allocation sources were live at that moment.
    """
    traces = snapshot.get("device_traces", [[]])[0]
    if not traces:
        return None

    current = 0
    peak = 0
    peak_idx = 0
    live_allocs = {}  # addr -> (size, frames)
    peak_live = {}

    for i, ev in enumerate(traces):
        action = ev.get("action")
        addr = ev.get("addr", 0)
        size = ev.get("size", 0)

        if action == "alloc":
            current += size
            live_allocs[addr] = (size, ev.get("frames", []))
            if current > peak:
                peak = current
                peak_idx = i
                peak_live = dict(live_allocs)
        elif action == "free_requested":
            if addr in live_allocs:
                current -= live_allocs[addr][0]
                del live_allocs[addr]

    # Categorize allocations at peak
    peak_categories = defaultdict(lambda: {"bytes": 0, "count": 0})
    for _addr, (size, frames) in peak_live.items():
        top = _get_top_python_frame(frames)
        if top:
            cat = _categorize_source(top["filename"], top["name"])
        else:
            cat = "Unknown"
        peak_categories[cat]["bytes"] += size
        peak_categories[cat]["count"] += 1

    return {
        "peak_bytes": peak,
        "peak_event_idx": peak_idx,
        "total_events": len(traces),
        "end_bytes": current,
        "peak_categories": dict(peak_categories),
    }


def print_peak_memory(result, mem_result=None, label=""):
    if not result:
        return

    if label:
        print(f"\n{'=' * 75}")
        print(f"  {label}")
        print(f"{'=' * 75}")

    peak_gb = result["peak_bytes"] / 1e9
    end_gb = result["end_bytes"] / 1e9
    # The device_traces only record allocations AFTER profiling starts.
    # Model weights and other persistent allocations are not tracked.
    # We can estimate the persistent baseline from snapshot allocated - peak_traced.
    persistent_gb = 0
    if mem_result:
        persistent_gb = mem_result["total_allocated"] / 1e9 - end_gb
    total_peak_gb = persistent_gb + peak_gb

    print(
        f"\n  Profiled peak (transient):  {peak_gb:.2f} GB  "
        f"(at event {result['peak_event_idx']:,} / {result['total_events']:,})"
    )
    if persistent_gb > 0:
        print(
            f"  Persistent baseline:       {persistent_gb:.2f} GB  "
            f"(model + optimizer, allocated before profiling)"
        )
        print(f"  Estimated total peak:      {total_peak_gb:.2f} GB")
    print(f"  Transient headroom:        {peak_gb - end_gb:.2f} GB above end-of-trace")

    cats = result.get("peak_categories", {})
    if cats:
        print("\n  Allocations live at peak:")
        print(f"    {'Category':<35} {'Size':>10} {'Count':>7}")
        print(f"    {'-' * 55}")
        for cat, info in sorted(
            cats.items(), key=lambda x: x[1]["bytes"], reverse=True
        ):
            sz = info["bytes"]
            if sz >= 1e9:
                sz_str = f"{sz / 1e9:.2f} GB"
            else:
                sz_str = f"{sz / 1e6:.1f} MB"
            print(f"    {cat:<35} {sz_str:>10} {info['count']:>7}")


# ---- Fragmentation diagnosis -----------------------------------------------


def analyze_fragmentation(snapshot):
    """Analyze segment-level memory layout to explain fragmentation.

    Examines each CUDA segment for inactive (freed but unreturned) blocks,
    pinned small allocations that prevent segment merging, and the overall
    segment size distribution.
    """
    segments = snapshot.get("segments", [])
    if not segments:
        return None

    total_reserved = 0
    total_allocated = 0
    total_inactive = 0
    segment_sizes = []
    inactive_gaps = []  # (gap_size, segment_size, active_around)
    pinned_fragments = []  # small active blocks surrounded by inactive

    for seg in segments:
        seg_size = seg.get("total_size", 0)
        total_reserved += seg_size
        segment_sizes.append(seg_size)
        blocks = seg.get("blocks", [])

        seg_active = 0
        seg_inactive = 0
        for bi, block in enumerate(blocks):
            bsize = block.get("size", 0)
            if block.get("state") == "active_allocated":
                seg_active += bsize
                total_allocated += bsize
                # Check if this small block is surrounded by inactive
                if bsize < 2 * 1024 * 1024:  # < 2MB
                    prev_inactive = bi > 0 and blocks[bi - 1].get("state") == "inactive"
                    next_inactive = (
                        bi < len(blocks) - 1
                        and blocks[bi + 1].get("state") == "inactive"
                    )
                    if prev_inactive and next_inactive:
                        pinned_fragments.append((bsize, seg_size))
            elif block.get("state") == "inactive":
                seg_inactive += bsize
                total_inactive += bsize
                inactive_gaps.append((bsize, seg_size))

    # Classify segment sizes
    size_buckets = defaultdict(lambda: {"count": 0, "total": 0})
    for sz in segment_sizes:
        if sz >= 1024 * 1024 * 1024:
            bucket = ">=1 GB"
        elif sz >= 256 * 1024 * 1024:
            bucket = "256MB-1GB"
        elif sz >= 64 * 1024 * 1024:
            bucket = "64-256MB"
        elif sz >= 2 * 1024 * 1024:
            bucket = "2-64MB"
        else:
            bucket = "<2MB"
        size_buckets[bucket]["count"] += 1
        size_buckets[bucket]["total"] += sz

    # Large inactive gaps that could be reclaimed
    inactive_gaps.sort(key=lambda x: x[0], reverse=True)

    return {
        "total_reserved": total_reserved,
        "total_allocated": total_allocated,
        "total_inactive": total_inactive,
        "n_segments": len(segments),
        "segment_size_buckets": dict(size_buckets),
        "large_inactive_gaps": inactive_gaps[:20],
        "pinned_fragments": len(pinned_fragments),
        "expandable_segments_would_help": (
            total_inactive > 0.1 * total_reserved and len(segments) > 10
        ),
    }


def print_fragmentation(result, gpu_capacity_gb=None, label=""):
    if not result:
        return

    if label:
        print(f"\n{'=' * 75}")
        print(f"  {label}")
        print(f"{'=' * 75}")

    reserved = result["total_reserved"]
    allocated = result["total_allocated"]
    inactive = result["total_inactive"]
    frag_pct = inactive / reserved * 100 if reserved > 0 else 0

    print(
        f"\n  Reserved:     {reserved / 1e9:.2f} GB across {result['n_segments']} segments"
    )
    print(f"  Allocated:    {allocated / 1e9:.2f} GB")
    print(f"  Inactive:     {inactive / 1e9:.2f} GB ({frag_pct:.1f}% fragmentation)")

    if result["pinned_fragments"] > 0:
        print(
            f"  Pinned small blocks (<2MB between inactive): "
            f"{result['pinned_fragments']} (prevent segment merging)"
        )

    # Segment size distribution
    print("\n  Segment size distribution:")
    bucket_order = [">=1 GB", "256MB-1GB", "64-256MB", "2-64MB", "<2MB"]
    for bucket in bucket_order:
        info = result["segment_size_buckets"].get(bucket)
        if info:
            print(
                f"    {bucket:<12} {info['count']:>4} segments  "
                f"{info['total'] / 1e9:>6.2f} GB"
            )

    # Largest inactive gaps
    gaps = result.get("large_inactive_gaps", [])
    if gaps:
        print("\n  Largest inactive gaps (freed but unreclaimable):")
        shown = 0
        for gap_sz, seg_sz in gaps:
            if gap_sz >= 32 * 1024 * 1024 and shown < 10:
                print(
                    f"    {gap_sz / 1e6:>8.0f} MB gap in {seg_sz / 1e6:.0f} MB segment"
                )
                shown += 1

    # OOM risk assessment
    if gpu_capacity_gb:
        gpu_bytes = gpu_capacity_gb * 1e9
        usable = gpu_bytes - (reserved - allocated)  # capacity minus fragmented waste
        print(f"\n  OOM Risk Assessment (GPU: {gpu_capacity_gb:.1f} GB):")
        print(
            f"    Usable capacity: {usable / 1e9:.2f} GB "
            f"(GPU capacity minus {inactive / 1e9:.2f} GB fragmentation)"
        )
        headroom = gpu_bytes - reserved
        print(f"    Current headroom: {headroom / 1e9:.2f} GB")
        if headroom < 1.0e9:
            print("    ⚠ CRITICAL: <1 GB headroom — high OOM risk!")
        elif headroom < 2.0e9:
            print("    ⚠ WARNING: <2 GB headroom — moderate OOM risk")

    # Recommendation
    if result.get("expandable_segments_would_help"):
        print("\n  → FIX: Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("    This eliminates segment fragmentation by growing segments in-place,")
        print(
            f"    which would reclaim up to {inactive / 1e9:.1f} GB of wasted memory."
        )


# ---- Sequence-length scaling analysis --------------------------------------


def analyze_scaling(mem_a, mem_b, churn_a, churn_b):
    """Compare per-tensor allocation sizes between two runs.

    When two profiles differ only in sequence length, this shows which tensor
    categories scale with sequence length by comparing the dominant tensor sizes.
    Total churn may differ due to different profiling windows, so we focus on
    per-tensor size ratios instead.
    """
    if not churn_a or not churn_b:
        return None

    def _cat_sizes(churn):
        """Map category -> {largest_size, total_bytes, count}."""
        cat_data = defaultdict(lambda: {"max_size": 0, "sizes": [], "count": 0})
        for sz, info in churn.items():
            for cat, cinfo in info.get("categories", {}).items():
                cat_data[cat]["count"] += cinfo["count"]
                cat_data[cat]["sizes"].append((sz, cinfo["count"]))
                if sz > cat_data[cat]["max_size"]:
                    cat_data[cat]["max_size"] = sz
        return dict(cat_data)

    cats_a = _cat_sizes(churn_a)
    cats_b = _cat_sizes(churn_b)

    all_cats = sorted(
        set(list(cats_a) + list(cats_b)),
        key=lambda c: max(
            cats_a.get(c, {"max_size": 0})["max_size"],
            cats_b.get(c, {"max_size": 0})["max_size"],
        ),
        reverse=True,
    )

    scaling = []
    for cat in all_cats:
        a = cats_a.get(cat)
        b = cats_b.get(cat)
        if not a or not b:
            continue
        a_max = a["max_size"]
        b_max = b["max_size"]
        if a_max > 1e6 and b_max > 1e6:  # Only compare >1MB tensors
            tensor_ratio = b_max / a_max if a_max > 0 else None
            scaling.append(
                {
                    "category": cat,
                    "size_a_mb": a_max / 1e6,
                    "size_b_mb": b_max / 1e6,
                    "tensor_ratio": tensor_ratio,
                    "count_a": a["count"],
                    "count_b": b["count"],
                    "scales_with_seqlen": tensor_ratio is not None
                    and tensor_ratio > 1.05,
                }
            )

    scaling.sort(key=lambda x: x["size_b_mb"], reverse=True)
    return scaling


def print_scaling(scaling, label_a="Before", label_b="After", label=""):
    if not scaling:
        return

    if label:
        print(f"\n{'=' * 75}")
        print(f"  {label}")
        print(f"{'=' * 75}")

    print("\n  Per-tensor size comparison (largest tensor per category):")
    print(
        f"  {'Category':<35} {'A size':>10} {'B size':>10} {'Ratio':>7} {'Scales?':>8}"
    )
    print(f"  {'-' * 73}")
    for entry in scaling:
        ratio_str = f"{entry['tensor_ratio']:.2f}x" if entry["tensor_ratio"] else "N/A"
        scales = "YES" if entry["scales_with_seqlen"] else "no"
        print(
            f"  {entry['category']:<35} {entry['size_a_mb']:>8.1f}MB "
            f"{entry['size_b_mb']:>8.1f}MB {ratio_str:>7} {scales:>8}"
        )

    # Summary
    seq_scaling = [e for e in scaling if e["scales_with_seqlen"]]
    constant = [e for e in scaling if not e["scales_with_seqlen"]]
    if seq_scaling:
        ratios = [e["tensor_ratio"] for e in seq_scaling if e["tensor_ratio"]]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        print(f"\n  Sequence-length scaling detected ({avg_ratio:.2f}x avg):")
        for e in seq_scaling:
            print(
                f"    - {e['category']}: {e['size_a_mb']:.1f}MB -> "
                f"{e['size_b_mb']:.1f}MB ({e['tensor_ratio']:.2f}x)"
            )
    if constant:
        print("\n  Constant-size categories (do not scale with seq len):")
        for e in constant:
            print(f"    - {e['category']}: {e['size_a_mb']:.1f}MB")


# ---- Main -----------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze axolotl training profiler output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Path to output directory (containing profiler_trace.json and/or "
        "snapshot.pickle) or directly to a trace file. "
        "Security note: snapshot.pickle uses pickle deserialization — "
        "only use files from your own trusted training runs.",
    )
    parser.add_argument(
        "--compare",
        help="Path to second run for A/B comparison. "
        "Same security note as path: only use trusted snapshot files.",
    )
    parser.add_argument(
        "--include-warmup",
        action="store_true",
        help="Include step 0 (warmup/compilation) in timing analysis",
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Only analyze memory snapshot, skip trace",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only load first 2M events for rapid analysis of large traces",
    )
    parser.add_argument(
        "--gpu-gb",
        type=float,
        default=None,
        help="GPU total memory in GB (for OOM risk assessment). "
        "Auto-detected if not specified.",
    )
    args = parser.parse_args()

    # Auto-detect GPU capacity if not specified
    gpu_capacity_gb = args.gpu_gb
    if gpu_capacity_gb is None:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_capacity_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            pass

    skip = not args.include_warmup
    trace_result = None
    mem_result = None
    events = None

    # -- Trace analysis --
    if not args.memory_only:
        events = load_trace(args.path, quick=args.quick)
        if events:
            trace_result = analyze_trace(events, skip_warmup=skip)
            if trace_result:
                print_trace_analysis(trace_result, label=f"Trace: {args.path}")

                if args.compare:
                    events2 = load_trace(args.compare, quick=args.quick)
                    if events2:
                        result2 = analyze_trace(events2, skip_warmup=skip)
                        if result2:
                            print_trace_analysis(
                                result2, label=f"Trace: {args.compare}"
                            )
                            compare_traces(trace_result, result2)
        else:
            print(f"  No profiler_trace.json found in {args.path}")

    # -- CPU overhead analysis (from trace) --
    if events and trace_result:
        cpu_result = analyze_cpu_overhead(events)
        print_cpu_overhead(
            cpu_result,
            n_steps=trace_result["n_steps"],
            label=f"CPU Overhead: {args.path}",
        )

    # -- Memory analysis --
    snapshot = load_snapshot(args.path)
    churn_result = None
    churn2 = None
    if snapshot:
        mem_result = analyze_snapshot(snapshot)
        print_memory_analysis(mem_result, label=f"Memory: {args.path}")

        # Peak memory timeline
        peak_result = analyze_peak_memory(snapshot)
        print_peak_memory(
            peak_result, mem_result=mem_result, label=f"Peak Memory: {args.path}"
        )

        # Fragmentation diagnosis
        frag_result = analyze_fragmentation(snapshot)
        print_fragmentation(
            frag_result,
            gpu_capacity_gb=gpu_capacity_gb,
            label=f"Fragmentation: {args.path}",
        )

        # Allocation churn attribution
        churn_result = analyze_allocation_churn(snapshot)
        if churn_result:
            print_allocation_churn(churn_result, label=f"Allocation Churn: {args.path}")

        if args.compare:
            snapshot2 = load_snapshot(args.compare)
            if snapshot2:
                mem2 = analyze_snapshot(snapshot2)
                print_memory_analysis(mem2, label=f"Memory: {args.compare}")

                # Peak memory for comparison
                peak2 = analyze_peak_memory(snapshot2)
                print_peak_memory(
                    peak2, mem_result=mem2, label=f"Peak Memory: {args.compare}"
                )

                # Fragmentation for comparison
                frag2 = analyze_fragmentation(snapshot2)
                print_fragmentation(
                    frag2,
                    gpu_capacity_gb=gpu_capacity_gb,
                    label=f"Fragmentation: {args.compare}",
                )

                churn2 = analyze_allocation_churn(snapshot2)
                if churn2:
                    print_allocation_churn(
                        churn2, label=f"Allocation Churn: {args.compare}"
                    )

                # Memory comparison summary
                print("\n  Memory comparison:")
                print(
                    f"    Reserved:  {mem_result['total_reserved'] / 1e9:.2f} -> "
                    f"{mem2['total_reserved'] / 1e9:.2f} GB"
                )
                print(
                    f"    Allocated: {mem_result['total_allocated'] / 1e9:.2f} -> "
                    f"{mem2['total_allocated'] / 1e9:.2f} GB"
                )
                print(
                    f"    Frag:      {mem_result['fragmentation_pct']:.1f}% -> "
                    f"{mem2['fragmentation_pct']:.1f}%"
                )
                if peak_result and peak2:
                    print(
                        f"    Peak:      {peak_result['peak_bytes'] / 1e9:.2f} -> "
                        f"{peak2['peak_bytes'] / 1e9:.2f} GB"
                    )

                # Scaling analysis
                if churn_result and churn2:
                    scaling = analyze_scaling(mem_result, mem2, churn_result, churn2)
                    print_scaling(
                        scaling,
                        label_a=str(args.path),
                        label_b=str(args.compare),
                        label="Allocation Scaling Analysis",
                    )
    elif not args.memory_only:
        pass  # trace-only is fine
    else:
        print(f"  No snapshot.pickle found in {args.path}")

    # -- Summary --
    if trace_result:
        print_summary(trace_result, mem_result=mem_result)


if __name__ == "__main__":
    main()
