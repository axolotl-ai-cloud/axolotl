"""Benchmark checkpoint-input offload against existing modes.

Example:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python benchmarks/bench_checkpoint_activation_offload.py \
    --model Qwen/Qwen3-8B --seq-lens 4096 8192 16384 --modes gc checkpoint_hs alst trl_all
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import subprocess  # nosec B404
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from trl.models.activation_offloading import OffloadActivations

from axolotl.monkeypatch.activation_offload_checkpoint import (
    patch_hidden_states_offload,
    unpatch_hidden_states_offload,
)
from axolotl.monkeypatch.checkpoint_activation_offload import (
    CheckpointHiddenStatesOffload,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


@dataclass
class BenchResult:
    model: str
    mode: str
    sequence_len: int
    steps: int
    mean_step_ms: float | None
    tokens_per_sec: float | None
    cuda_peak_allocated_mb: float | None
    cuda_peak_reserved_mb: float | None
    cuda_peak_delta_mb: float | None
    cpu_rss_peak_mb: float | None
    cpu_rss_delta_mb: float | None
    checkpoint_saved_tensors_seen: int | None = None
    checkpoint_marked_tensors: int | None = None
    checkpoint_offloaded_tensors: int | None = None
    checkpoint_offloaded_mb: float | None = None
    error: str | None = None


class RssSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)


def _clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _load_model(model_id: str, attn_implementation: str):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )


def _make_input_ids(sequence_len: int, vocab_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234 + sequence_len)
    return torch.randint(
        3,
        vocab_size - 1,
        (1, sequence_len),
        dtype=torch.long,
        generator=generator,
    )


def _configure_mode(model, mode: str):
    ctx = contextlib.nullcontext()
    cleanup = contextlib.ExitStack()
    checkpoint_kwargs = {"use_reentrant": False}

    if mode == "gc":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpoint_kwargs
        )
    elif mode == "trl_gc":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpoint_kwargs
        )
        ctx = OffloadActivations(use_streams=True, min_offload_size=0)
        ctx.update_model_params(model)
    elif mode == "trl_all":
        model.gradient_checkpointing_disable()
        ctx = OffloadActivations(use_streams=True, min_offload_size=0)
        ctx.update_model_params(model)
    elif mode == "checkpoint_hs":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpoint_kwargs
        )
        ctx = CheckpointHiddenStatesOffload(use_streams=True, min_offload_size=0)
    elif mode == "alst":
        patch_hidden_states_offload()
        cleanup.callback(unpatch_hidden_states_offload)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    return ctx, cleanup


def _run_one(
    model_id: str,
    sequence_len: int,
    mode: str,
    steps: int,
    warmup_steps: int,
    attn_implementation: str,
) -> BenchResult:
    _clean_gpu()
    model = None
    input_ids = None
    outputs = None
    loss = None
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    result = BenchResult(
        model=model_id,
        mode=mode,
        sequence_len=sequence_len,
        steps=steps,
        mean_step_ms=None,
        tokens_per_sec=None,
        cuda_peak_allocated_mb=None,
        cuda_peak_reserved_mb=None,
        cuda_peak_delta_mb=None,
        cpu_rss_peak_mb=None,
        cpu_rss_delta_mb=None,
    )

    try:
        config = AutoConfig.from_pretrained(model_id)
        input_ids = _make_input_ids(sequence_len, config.vocab_size).cuda()
        model = _load_model(model_id, attn_implementation).cuda()
        model.train()
        model.config.use_cache = False
        ctx, cleanup = _configure_mode(model, mode)

        _clean_gpu()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_rss = process.memory_info().rss
        step_times = []
        with cleanup, RssSampler() as rss_sampler, ctx:
            for step_idx in range(warmup_steps + steps):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if step_idx == warmup_steps:
                    torch.cuda.reset_peak_memory_stats()
                    baseline_allocated = torch.cuda.memory_allocated()
                start = time.perf_counter()
                outputs = model(input_ids=input_ids, logits_to_keep=1)
                loss = outputs.logits.float().mean()
                loss.backward()
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if step_idx >= warmup_steps:
                    step_times.append(elapsed_ms)

        result.mean_step_ms = sum(step_times) / len(step_times)
        result.tokens_per_sec = (sequence_len * len(step_times)) / (
            sum(step_times) / 1000.0
        )
        peak_allocated = torch.cuda.max_memory_allocated()
        result.cuda_peak_allocated_mb = peak_allocated / 1e6
        result.cuda_peak_reserved_mb = torch.cuda.max_memory_reserved() / 1e6
        result.cuda_peak_delta_mb = (peak_allocated - baseline_allocated) / 1e6
        result.cpu_rss_peak_mb = rss_sampler.peak_rss / 1e6
        result.cpu_rss_delta_mb = (rss_sampler.peak_rss - baseline_rss) / 1e6
        if isinstance(ctx, CheckpointHiddenStatesOffload):
            result.checkpoint_saved_tensors_seen = ctx.stats.saved_tensors_seen
            result.checkpoint_marked_tensors = ctx.stats.marked_tensors
            result.checkpoint_offloaded_tensors = ctx.stats.offloaded_tensors
            result.checkpoint_offloaded_mb = ctx.stats.offloaded_bytes / 1e6
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        result.error = "CUDA OOM"
    finally:
        del model, input_ids, outputs, loss
        _clean_gpu()

    return result


def _print_result(result: BenchResult):
    payload = asdict(result)
    print(json.dumps(payload, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=_positive_int,
        default=[4096, 8192, 16384],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["gc", "trl_gc", "trl_all", "checkpoint_hs", "alst"],
        default=["gc", "checkpoint_hs", "alst", "trl_all"],
    )
    parser.add_argument("--steps", type=_positive_int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--output-jsonl")
    parser.add_argument("--isolate-runs", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    if args.isolate_runs and os.environ.get("AXOLOTL_BENCH_CHILD") != "1":
        for sequence_len in args.seq_lens:
            for mode in args.modes:
                cmd = [
                    sys.executable,
                    __file__,
                    "--model",
                    args.model,
                    "--seq-lens",
                    str(sequence_len),
                    "--modes",
                    mode,
                    "--steps",
                    str(args.steps),
                    "--warmup-steps",
                    str(args.warmup_steps),
                    "--attn-implementation",
                    args.attn_implementation,
                ]
                if args.output_jsonl:
                    cmd.extend(["--output-jsonl", args.output_jsonl])
                env = os.environ.copy()
                env["AXOLOTL_BENCH_CHILD"] = "1"
                subprocess.run(cmd, check=True, env=env)  # nosec B603
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for sequence_len in args.seq_lens:
        for mode in args.modes:
            result = _run_one(
                args.model,
                sequence_len,
                mode,
                args.steps,
                args.warmup_steps,
                args.attn_implementation,
            )
            _print_result(result)
            if output_path:
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
