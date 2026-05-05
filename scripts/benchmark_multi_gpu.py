"""Multi-GPU mode throughput + memory benchmark for ProTrain on 4x RTX 3090.

Compares four training modes on an identical workload (fresh-init
Llama-3B + LoRA r=8, bs=2 per rank, seq=256, fp16) and emits both a
JSON file and a human-readable markdown table:

    1. single-rank (baseline)            — world_size=1, no protrain collectives
    2. DDP composition                   — world_size=4, force_all_persistent=True,
                                            outer DistributedDataParallel wrap
    3. replicated offload (ZeRO-2-ish)   — world_size=4, zero3_shard=False,
                                            force_all_persistent=False, no DDP wrap
                                            (per-param all_reduce owns grad sync)
    4. ZeRO-3 sharded                    — world_size=4, zero3_shard=True,
                                            force_all_persistent=False, no DDP wrap
                                            (reduce_scatter / all_gather own the path)

Per-rank GPU peak is measured via ``torch.cuda.max_memory_allocated``;
per-rank CPU pinned bytes come from the chunk manager:
    - ZeRO-3 mode: ``chunk_manager.per_rank_cpu_bytes()`` (sum over
      ``_ChunkShardState.shard_bytes``).
    - Replicated mode: sum of ``slot.cpu_data.numel() *
      slot.element_size`` over every ``_CpuParamSlot`` (full chunk on
      every rank).
    - DDP / single-rank: reported as 0 (chunks are fully persistent —
      nothing lives on CPU).

Throughput:
    throughput = world_size * bs / median_iter_s

Each mode runs 6 iterations; iterations 0..1 are warm-up and discarded;
the median of iterations 2..5 is used.

Intentional CUDA environment handling:
    - ``CUDA_VISIBLE_DEVICES=1,4,5,7`` (the 4 unused 3090s on this rig)
    - ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` — propagated into child
      subprocesses because torch's default FASTEST_FIRST re-orders the
      visible set by SM count on the mixed-SKU test host and would
      silently route ranks to non-3090 silicon.

Usage:
    CUDA_VISIBLE_DEVICES=1,4,5,7 CUDA_DEVICE_ORDER=PCI_BUS_ID \
        python scripts/benchmark_multi_gpu.py

Writes:
    scripts/multi_gpu_benchmark_results.json
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess  # nosec B404
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# The multi-rank worker script is a heredoc string so this file is
# self-contained and has no sibling module dependency. Environment
# variables carry the mode selector.
_WORKER_SCRIPT = textwrap.dedent(
    '''
    """Subprocess entry: spawns ``PROTRAIN_WORLD_SIZE`` ranks and
    writes per-rank stats to ``PROTRAIN_OUT_DIR/rank{r}.json``.

    Mode selector (``PROTRAIN_MODE``):
        "single"     — world_size=1, no protrain collectives
        "ddp"        — world_size=N, force_all_persistent=True, DDP wrap
        "replicated" — world_size=N, zero3_shard=False, no DDP
        "zero3"      — world_size=N, zero3_shard=True,  no DDP
    """
    import json
    import os
    import sys
    import time
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


    def _worker(rank, world_size, out_dir, mode, bs, seq, n_iters, n_warmup):
        # Set CUDA_DEVICE_ORDER in the child before any CUDA alloc —
        # torch reads it at init-time. Parent passed it through env,
        # but spawn inherits from the parent shell's env so we re-assert
        # it here for safety (the existing M6 test does the same).
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if world_size > 1:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = os.environ.get(
                "PROTRAIN_MASTER_PORT", "29542"
            )

        torch.cuda.set_device(rank)
        if world_size > 1:
            # Bound NCCL rendezvous so a stuck rank fails fast instead
            # of hanging the whole benchmark up to the parent's 30-min
            # subprocess timeout. 5 min is generous for a localhost
            # process group on this rig (typically completes in <2s).
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                device_id=torch.device("cuda", rank),
                timeout=timedelta(minutes=5),
            )
        try:
            _run(rank, world_size, out_dir, mode, bs, seq, n_iters, n_warmup)
            # Barrier ONLY on the success path. CodeRabbit R2-01: a
            # teardown barrier in ``finally`` blocks remaining workers
            # when one peer has already raised, turning a single-rank
            # failure into a full ``_launch_mode`` 30-min timeout. On
            # the failure path we skip the barrier and rely on
            # ``destroy_process_group`` alone.
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
        finally:
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


    def _run(rank, world_size, out_dir, mode, bs, seq, n_iters, n_warmup):
        from transformers import LlamaConfig, LlamaForCausalLM
        from peft import LoraConfig, get_peft_model

        from axolotl.integrations.protrain.api import (
            protrain_model_wrapper,
            protrain_optimizer_wrapper,
        )
        from axolotl.integrations.protrain.types import HardwareProfile

        # Use a shared seed across ranks for model init so the
        # ``replicated`` and ``zero3`` modes start from identical
        # weights on every rank — i.e. the cross-rank setup is a true
        # synchronized replica/shard, matching what DDP gives via
        # broadcast at wrap time. Without this, fresh-init RNG
        # divergence biases the mode-to-mode benchmark comparison.
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Llama-3B config — same as the M7 ZeRO-3 test so profiler cache
        # hits are shared across runs.
        cfg = LlamaConfig(
            hidden_size=2560,
            num_hidden_layers=26,
            num_attention_heads=20,
            num_key_value_heads=20,
            intermediate_size=6912,
            vocab_size=32000,
            use_cache=False,
        )

        device = torch.device("cuda", rank)
        # fp16 + LoRA matches the DDP-mode M6 workload. Fresh-init fp16
        # logits can overflow, but we only care about throughput /
        # memory — loss value is irrelevant here. LoRA r=8 keeps the
        # trainable-param set tiny so DDP's allreduce overhead is
        # negligible relative to the model compute.
        model = LlamaForCausalLM(cfg).half().to(device)

        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

        hw = HardwareProfile(
            gpu_sku=torch.cuda.get_device_name(rank),
            gpu_memory_bytes=torch.cuda.get_device_properties(rank).total_memory,
            gpu_count=world_size,
            pcie_h2d_bps=13e9,
            pcie_d2h_bps=13e9,
            has_nvlink=False,
        )

        # Build kwargs per mode.
        force_all_persistent = (mode == "ddp") or (mode == "single")
        if mode == "zero3":
            zero3_shard = True
        elif mode == "replicated":
            zero3_shard = False
        else:
            zero3_shard = None  # auto; ends up False for DDP / single

        # For replicated / zero3 modes we MUST drive the searcher away
        # from picking ``n_persist = N_chunk`` — otherwise the CPU pool
        # stays empty and the "offloaded replicated" mode is
        # indistinguishable from DDP.
        #
        # Round-3 R9 tightened the explicit-override path to reject
        # configs whose offloaded chunks land on non-CKPT blocks
        # (``block_map_runtime_admissible``). The previous hardcoded
        # tuple ``n_persist=2, n_checkpoint=0, n_swap=0`` is invalid for
        # any model whose chunks beyond the first 2 don't all map to
        # CKPT blocks — i.e. most realistic models. Computing
        # admissible overrides up front would require N_chunk / N_block,
        # which aren't known here (the layout is built inside
        # ``protrain_model_wrapper``). Instead we drive the searcher
        # via the capacity inputs: a tight ``capacity_bytes`` forces
        # ``n_persist < N_chunk`` so the searcher selects a feasible
        # offload config (with a CKPT-admissible block_map). DDP /
        # single keep the loose 20 GiB so the searcher lands at
        # ``n_persist = N_chunk`` (Mode A) naturally.
        if mode in ("replicated", "zero3"):
            # 4 GiB per rank — well below the Llama-3B fp16 param
            # footprint (~6 GB), guaranteeing the searcher CANNOT pick
            # a fully-persistent layout and must offload some chunks
            # to host RAM. The searcher picks n_buffer / n_checkpoint /
            # n_swap consistent with the resulting block_map.
            capacity = 4 * (1 << 30)
        else:
            capacity = 20 * (1 << 30)

        wrapper_kwargs = dict(
            model_config=cfg,
            hardware_profile=hw,
            batch_size=bs,
            seq_len=seq,
            capacity_bytes=capacity,
            auto_mode=False,
            force_all_persistent=force_all_persistent,
            zero3_shard=zero3_shard,
        )

        wrapped = protrain_model_wrapper(model, **wrapper_kwargs)
        optim = protrain_optimizer_wrapper(wrapped, lr=1e-4)

        use_ddp = (mode == "ddp")
        if use_ddp:
            # Per M6 test comments: force_all_persistent=True means
            # every chunk is resident on GPU at DDP-wrap time, so DDP
            # sees real shapes (zero-sized placeholders would break it).
            # Skip internal grad reduce — DDP owns cross-rank sync.
            wrapped.chunk_manager.skip_internal_grad_reduce = True
            ddp_module = torch.nn.parallel.DistributedDataParallel(
                wrapped.module,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        else:
            ddp_module = wrapped.module

        # Reseed per-rank AFTER model init so each rank gets a distinct
        # synthetic minibatch (model weights stay identical across ranks
        # — see the shared ``manual_seed(42)`` above).
        torch.manual_seed(42 + rank)
        input_ids = torch.randint(
            0, cfg.vocab_size, (bs, seq), device=device, dtype=torch.long
        )
        labels = input_ids.clone()

        iter_times = []
        # Reset CUDA peak so warm-up setup doesn't contribute.
        # We reset BEFORE the warm-up iterations to include their peak
        # in the max_memory_allocated reading as well — every iteration
        # touches the same path so the peak is stable across iters.
        torch.cuda.reset_peak_memory_stats(device)
        for i in range(n_iters):
            torch.cuda.synchronize()
            if world_size > 1:
                dist.barrier()
            t0 = time.perf_counter()

            out = ddp_module(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad()

            torch.cuda.synchronize()
            if world_size > 1:
                dist.barrier()
            iter_times.append(time.perf_counter() - t0)

        peak_gpu_bytes = torch.cuda.max_memory_allocated(device)

        # Per-rank CPU pinned bytes:
        #   - ZeRO-3: chunk_manager.per_rank_cpu_bytes() (shard sum)
        #   - replicated (offloaded, non-sharded): sum of cpu_data
        #     element bytes across every param slot on this rank
        #   - DDP / single: chunks are fully persistent -> 0 CPU bytes
        chunk_manager = wrapped.chunk_manager
        if mode == "zero3":
            cpu_pinned = int(chunk_manager.per_rank_cpu_bytes())
        elif mode == "replicated":
            # Replicated mode holds the full chunk on every rank.
            # Use the public accessor (mirrors per_rank_cpu_bytes for
            # ZeRO-3 sharded layout) instead of touching ``_cpu_slots``.
            cpu_pinned = int(chunk_manager.replicated_cpu_bytes())
        else:
            cpu_pinned = 0

        # Record the trainable parameter count (LoRA adapter set) for
        # sanity — same number across modes modulo ProTrain internal
        # param list differences.
        n_trainable = sum(
            p.numel() for _, p in wrapped.module.named_parameters()
            if p.requires_grad and p.numel() > 0
        )

        stats = {
            "rank": rank,
            "mode": mode,
            "world_size": world_size,
            "bs": bs,
            "seq": seq,
            "iter_times": iter_times,
            "peak_gpu_bytes": peak_gpu_bytes,
            "cpu_pinned_bytes": cpu_pinned,
            "n_trainable": n_trainable,
        }
        out_path = os.path.join(out_dir, f"rank{rank}.json")
        with open(out_path, "w") as f:
            json.dump(stats, f)
        print(
            f"[rank{rank}] mode={mode} ws={world_size} "
            f"iters={[round(t, 4) for t in iter_times]} "
            f"peak_gpu={peak_gpu_bytes/1e9:.2f}GB "
            f"cpu_pinned={cpu_pinned/1e9:.3f}GB",
            flush=True,
        )


    def main():
        world = int(os.environ["PROTRAIN_WORLD_SIZE"])
        bs = int(os.environ["PROTRAIN_BATCH_SIZE"])
        seq = int(os.environ["PROTRAIN_SEQ_LEN"])
        n_iters = int(os.environ["PROTRAIN_N_ITERS"])
        n_warmup = int(os.environ["PROTRAIN_N_WARMUP"])
        out_dir = os.environ["PROTRAIN_OUT_DIR"]
        mode = os.environ["PROTRAIN_MODE"]

        os.makedirs(out_dir, exist_ok=True)

        if world == 1:
            # Run inline (no spawn) — mirrors the M6 baseline pattern.
            _worker(0, 1, out_dir, mode, bs, seq, n_iters, n_warmup)
            return 0

        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(world):
            p = ctx.Process(
                target=_worker,
                args=(rank, world, out_dir, mode, bs, seq, n_iters, n_warmup),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        for p in procs:
            if p.exitcode != 0:
                print(f"worker pid={p.pid} exited with {p.exitcode}", flush=True)
                return p.exitcode
        return 0


    if __name__ == "__main__":
        sys.exit(main())
    '''
)


# ---- Orchestration ----------------------------------------------------


def _launch_mode(
    *,
    mode: str,
    world_size: int,
    cuda_visible: str,
    bs: int,
    seq: int,
    n_iters: int,
    n_warmup: int,
    work_dir: Path,
    master_port: str,
) -> list[dict]:
    """Run one mode in a subprocess, return the per-rank stats list."""
    out_dir = work_dir / f"stats_{mode}"
    # Clear stale per-rank stats from any prior failed/partial run so we
    # don't pick up rank*.json files that were never overwritten.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    # MUST propagate PCI_BUS_ID ordering into the child — see comment
    # on _launch in tests/protrain/test_multi_gpu_7b.py.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PROTRAIN_WORLD_SIZE"] = str(world_size)
    env["PROTRAIN_BATCH_SIZE"] = str(bs)
    env["PROTRAIN_SEQ_LEN"] = str(seq)
    env["PROTRAIN_N_ITERS"] = str(n_iters)
    env["PROTRAIN_N_WARMUP"] = str(n_warmup)
    env["PROTRAIN_OUT_DIR"] = str(out_dir)
    env["PROTRAIN_MODE"] = mode
    # Each mode gets its own port to avoid stale bind errors when a
    # prior subprocess leaks the rendezvous socket.
    env["PROTRAIN_MASTER_PORT"] = master_port
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_P2P_DISABLE", "0")

    script_path = work_dir / f"_worker_{mode}.py"
    script_path.write_text(_WORKER_SCRIPT)
    log_path = work_dir / f"worker_{mode}.log"
    with log_path.open("w") as log_f:
        proc = subprocess.Popen(  # nosec B603
            [sys.executable, str(script_path)],
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(
                f"mode={mode} worker timed out; killed process group; log tail:\n{tail}"
            ) from None
    if returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(
            f"mode={mode} worker failed (exit={returncode}); log tail:\n{tail}"
        )

    # Collect per-rank stats.
    stats = []
    for r in range(world_size):
        p = out_dir / f"rank{r}.json"
        if not p.exists():
            raise RuntimeError(f"mode={mode}: rank{r}.json missing; see {log_path}")
        with p.open() as f:
            stats.append(json.load(f))
    return stats


def _summarize(mode: str, per_rank: list[dict], n_warmup: int) -> dict:
    """Combine per-rank stats into one summary row."""
    world_size = per_rank[0]["world_size"]
    bs = per_rank[0]["bs"]
    # Use rank 0's iter times for throughput (all ranks barrier
    # together so rank-0 time is representative). Drop warm-up.
    rank0_times = per_rank[0]["iter_times"][n_warmup:]
    if not rank0_times:
        raise RuntimeError(
            f"mode={mode}: no non-warmup iters; iter_times={per_rank[0]['iter_times']}"
        )
    median_iter = statistics.median(rank0_times)
    throughput = world_size * bs / median_iter

    peaks_gpu = [r["peak_gpu_bytes"] for r in per_rank]
    cpu_pinned = [r["cpu_pinned_bytes"] for r in per_rank]

    return {
        "mode": mode,
        "world_size": world_size,
        "bs_per_rank": bs,
        "median_iter_s": median_iter,
        "throughput_samples_per_s": throughput,
        "peak_gpu_bytes_per_rank": peaks_gpu,
        "cpu_pinned_bytes_per_rank": cpu_pinned,
        "peak_gpu_bytes_max": max(peaks_gpu),
        "cpu_pinned_bytes_max": max(cpu_pinned) if cpu_pinned else 0,
        "iter_times_rank0": per_rank[0]["iter_times"],
    }


def _fmt_gb(b: int) -> str:
    return f"{b / 1e9:.2f} GB"


def _render_markdown(summaries: list[dict]) -> str:
    """Return a markdown table + qualitative summary."""
    baseline = next((s for s in summaries if s["mode"] == "single"), None)
    base_tp = baseline["throughput_samples_per_s"] if baseline else None

    lines = [
        "| Mode | World | Throughput (samples/s) | Scaling vs 1-GPU | Per-rank GPU peak | Per-rank CPU pinned |",
        "|---|---|---|---|---|---|",
    ]
    pretty = {
        "single": "Single-rank (baseline)",
        "ddp": "DDP (force_all_persistent=True)",
        "replicated": "Replicated offload (zero3_shard=False)",
        "zero3": "ZeRO-3 sharded (zero3_shard=True)",
    }
    order = ["single", "ddp", "replicated", "zero3"]
    for mode in order:
        row = next((s for s in summaries if s["mode"] == mode), None)
        if row is None:
            continue
        if base_tp:
            scaling = f"{row['throughput_samples_per_s'] / base_tp:.2f}x"
        else:
            scaling = "—"
        lines.append(
            f"| {pretty[mode]} | {row['world_size']} | "
            f"{row['throughput_samples_per_s']:.3f} | "
            f"{scaling} | "
            f"{_fmt_gb(row['peak_gpu_bytes_max'])} | "
            f"{_fmt_gb(row['cpu_pinned_bytes_max'])} |"
        )
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parent
    work_dir = Path(tempfile.mkdtemp(prefix="benchmark_multi_gpu_", dir=str(root)))

    # Cleanup escape hatch: set PROTRAIN_BENCHMARK_KEEP_TMP=1 to retain
    # the per-rank stats dir (rank{r}.json) after the run for debugging
    # a failed mode. Default behavior is to remove it on both success
    # and failure so repeated runs don't leak temp dirs under scripts/.
    keep_tmp = os.environ.get("PROTRAIN_BENCHMARK_KEEP_TMP", "") == "1"

    try:
        bs = 2
        seq = 256
        n_iters = 6
        n_warmup = 2

        # Each mode gets its own port to avoid bind collisions across
        # sequential subprocess lifetimes on the same host.
        ports = {
            "single": "29540",
            "ddp": "29541",
            "replicated": "29542",
            "zero3": "29543",
        }

        t0 = time.perf_counter()
        results = {}

        # Single-rank baseline — isolate on CUDA_VISIBLE_DEVICES=1 so it
        # doesn't trip over the multi-rank env. world_size=1 means no
        # process group setup; same as running on a fresh shell.
        print("\n[benchmark] single-rank baseline (GPU 1)...", flush=True)
        stats = _launch_mode(
            mode="single",
            world_size=1,
            cuda_visible="1",
            bs=bs,
            seq=seq,
            n_iters=n_iters,
            n_warmup=n_warmup,
            work_dir=work_dir,
            master_port=ports["single"],
        )
        results["single"] = _summarize("single", stats, n_warmup)

        for mode in ("ddp", "replicated", "zero3"):
            print(f"\n[benchmark] {mode} world_size=4 (GPUs 1,4,5,7)...", flush=True)
            stats = _launch_mode(
                mode=mode,
                world_size=4,
                cuda_visible="1,4,5,7",
                bs=bs,
                seq=seq,
                n_iters=n_iters,
                n_warmup=n_warmup,
                work_dir=work_dir,
                master_port=ports[mode],
            )
            results[mode] = _summarize(mode, stats, n_warmup)

        wall_s = time.perf_counter() - t0

        # Persist JSON (ordered + with wall clock).
        summary_order = ["single", "ddp", "replicated", "zero3"]
        summaries: list[dict] = [results[m] for m in summary_order if m in results]
        payload = {
            "workload": {
                "model": "Llama-3B (fresh-init, LoRA r=8)",
                "bs_per_rank": bs,
                "seq": seq,
                "n_iters": n_iters,
                "n_warmup": n_warmup,
                "dtype": "fp16",
                "gpus": "1,4,5,7 (RTX 3090)",
            },
            "wall_clock_s": wall_s,
            "summaries": summaries,
        }
        out_json = root / "multi_gpu_benchmark_results.json"
        with out_json.open("w") as f:
            json.dump(payload, f, indent=2)

        md = _render_markdown(summaries)
        print("\n" + "=" * 72)
        print("ProTrain multi-GPU benchmark — 4x RTX 3090 (GPUs 1,4,5,7)")
        print("=" * 72)
        print(md)
        print()
        print(f"Wall clock: {wall_s:.1f}s")
        print(f"JSON written to: {out_json}")
        return 0
    finally:
        if keep_tmp:
            print(
                f"[benchmark] PROTRAIN_BENCHMARK_KEEP_TMP=1 — retaining {work_dir}",
                flush=True,
            )
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
