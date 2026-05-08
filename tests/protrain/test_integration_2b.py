"""Fast surrogate of the 7B headline integration test on a 2B-class Llama.

Same end-to-end pipeline, same tolerance assertions, same OFFLOAD-engaging
configuration shape — sized small enough to land in 3-5 minutes on a single
RTX 3090 / 3090 Ti so the calibration loop stays interactive. The 7B
headline (``test_protrain_7b_end_to_end``) takes 25-30 minutes and remains
the canonical accuracy target; this 2B smoke runs the same structural paths
(persistent + buffer-pool layout, OFFLOAD bumps, phase-2 measurement,
cfg-delta floor) and reproduces the same peak under-/over-prediction
patterns at a tenth of the wall-clock cost.

Sizing: hidden=2048, layers=12, heads=16, kv_heads=16, intermediate=5632,
vocab=32000 — ~1.4B params (roughly Llama-3 1B-class). ``capacity_bytes =
4 GiB`` is tight enough that the searcher MUST place some chunks on
OFFLOAD (model state alone is ~3 GB, leaving very little for activations
and buffer-pool slots), exercising the same code paths the 7B test stresses
at 20 GiB capacity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from axolotl.integrations.protrain.chunk import ChunkManager


def _mark(stage: str) -> None:
    """Emit a progress marker that survives pytest output buffering."""
    import sys

    line = f"[protrain-2b] {stage}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    sys.stderr.write(line)
    sys.stderr.flush()


@pytest.mark.slow
@pytest.mark.gpu
def test_protrain_2b_lora_smoke() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    _mark("starting — importing Llama config")
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    # ---- Fresh-init Llama-2B-class architecture (no weight download) ----
    # Roughly the smallest size that still has multiple transformer blocks
    # with realistic chunk-layout ratios; LoRA on q/k/v/o_proj mirrors the
    # 7B test's deployment shape.
    cfg = LlamaConfig(
        hidden_size=2048,
        num_hidden_layers=22,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=5632,
        vocab_size=32000,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        torch_dtype="float16",
        use_cache=False,
    )

    _mark("constructing fresh-init Llama-2B on CPU")
    model = LlamaForCausalLM(cfg).half().to("cuda")
    _mark(f"base model on GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    _mark("applying LoRA adapters (r=8 on q/k/v/o_proj)")
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _mark(
        f"LoRA applied: trainable={trainable / 1e6:.2f}M total={total / 1e9:.2f}B "
        f"gpu_alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    # Small synthetic batch — same shape as the 7B test for parity.
    bs, seq = 1, 256
    input_ids = torch.randint(
        0, cfg.vocab_size, (bs, seq), device="cuda", dtype=torch.long
    )
    labels = input_ids.clone()
    batch = {"input_ids": input_ids, "labels": labels}

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    hw = HardwareProfile(
        gpu_sku=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_count=1,
        pcie_h2d_bps=13e9,
        pcie_d2h_bps=13e9,
        has_nvlink=False,
    )
    # Tight 4 GiB capacity. Llama-2B fp16 model state ≈ 4.6 GB resident
    # (22 layers × ~210 MB/layer); with capacity=4 GiB the searcher
    # MUST place chunks on OFFLOAD — exercising the same code paths
    # the 7B test exercises at 20 GiB capacity (~14 GB model state,
    # ~6 GB headroom).
    capacity_bytes = 4 * (1 << 30)
    _mark("entering protrain_model_wrapper (profiler + layout + search)")
    wrapped = protrain_model_wrapper(
        model,
        model_config=cfg,
        hardware_profile=hw,
        batch_size=bs,
        seq_len=seq,
        capacity_bytes=capacity_bytes,
    )
    _mark(
        f"wrapper done: cfg={wrapped.search_result.cfg} "
        f"peak_pred={wrapped.search_result.predicted_peak_bytes / 1e9:.2f} GB "
        f"iter_pred={wrapped.search_result.predicted_iter_s:.3f} s "
        f"gpu_alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    # Same calibration-premise gate as the 7B test: when CPU-Adam is
    # unavailable, the runtime calibration target is undefined.
    measured_hw = getattr(wrapped, "_hardware_profile", None)
    if measured_hw is not None and measured_hw.cpu_adam_bytes_per_sec <= 0.0:
        pytest.skip(
            "calibration premise unmet: DeepSpeedCPUAdam unavailable on "
            "this rig (cpu_adam_bytes_per_sec=0)."
        )

    optim = protrain_optimizer_wrapper(wrapped, lr=1e-4)
    _mark(f"optimizer built; gpu_alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Track losses for descent assertion — surrogate for "training is real".
    losses: list[float] = []

    N_ITERS = 4
    iter_s: list[float] = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    _mark(f"about to run {N_ITERS} training iterations (fwd+bwd+step)")
    for i in range(N_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            out = wrapped.module(**batch)
        except Exception as e:  # noqa: BLE001 - diagnostic passthrough
            _mark(f"iter {i} forward FAILED: {type(e).__name__}: {e!s:.400}")
            raise
        loss_val = float(out.loss)
        losses.append(loss_val)
        _mark(
            f"iter {i} forward done: loss={loss_val:.4f} "
            f"gpu_alloc={torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )
        loss = out.loss
        try:
            loss.backward()
        except Exception as e:  # noqa: BLE001 - diagnostic passthrough
            _mark(f"iter {i} backward FAILED: {type(e).__name__}: {e!s:.400}")
            raise
        optim.step()
        optim.zero_grad()
        end.record()
        torch.cuda.synchronize()
        iter_s.append(start.elapsed_time(end) / 1000.0)
        _mark(f"iter {i} done: {iter_s[-1]:.3f} s")

    actual_peak = torch.cuda.max_memory_allocated()
    import math
    import statistics

    steady = iter_s[2:]
    actual_iter_s = statistics.median(steady) if steady else iter_s[-1]
    iter_s_all = iter_s

    predicted_peak = wrapped.search_result.predicted_peak_bytes
    predicted_iter_s = wrapped.search_result.predicted_iter_s

    print(
        "\nProTrain 2B integration:\n"
        f"  predicted peak: {predicted_peak / 1e9:.2f} GB  "
        f"actual: {actual_peak / 1e9:.2f} GB\n"
        f"  predicted iter: {predicted_iter_s:.2f} s    "
        f"actual (median iters 2-3): {actual_iter_s:.3f} s\n"
        f"  all iter times (s): {[round(t, 3) for t in iter_s_all]}\n"
        f"  losses: {[round(loss_v, 4) for loss_v in losses]}\n"
        f"  chosen config: {wrapped.search_result.cfg}\n"
        f"  S_chunk={cast('ChunkManager', wrapped.chunk_manager).layout.S_chunk} "
        f"N_chunk={cast('ChunkManager', wrapped.chunk_manager).layout.N_chunk}"
    )

    peak_err = abs(predicted_peak - actual_peak) / max(1, actual_peak)
    runtime_err = abs(predicted_iter_s - actual_iter_s) / max(1e-9, actual_iter_s)

    # Loss sanity — no NaN/Inf and at least one descent across the full run.
    for loss_v in losses:
        assert math.isfinite(loss_v), f"loss not finite: {losses}"
    assert losses[-1] < losses[0], (
        f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )

    # OOM-safety invariant: actual peak within capacity budget.
    assert actual_peak < capacity_bytes, (
        f"actual peak {actual_peak / 1e9:.2f} GB exceeded capacity "
        f"{capacity_bytes / 1e9:.2f} GiB"
    )
    # Same under/over-predict tolerances as the 7B headline.
    assert predicted_peak >= actual_peak * 0.95, (
        f"peak UNDER-predict: predicted {predicted_peak / 1e9:.2f} GB < "
        f"actual {actual_peak / 1e9:.2f} GB"
    )
    assert peak_err < 0.10, f"peak prediction off by {peak_err * 100:.1f}%"
    assert runtime_err < 0.10, (
        f"runtime prediction off by {runtime_err * 100:.1f}% — iter_s_all={iter_s_all}"
    )
