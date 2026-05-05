"""Smoke tests for the paper-Figure-1 ergonomic helper :func:`auto_wrap`.

The helper composes:
1. live ``torch.cuda`` queries -> :class:`HardwareProfile`
2. :func:`protrain_model_wrapper` with ``auto_mode=True``

so direct API users (notebooks, scripts, integrations outside Axolotl)
can drop in ProTrain with one call. The plugin path (Axolotl YAML
users) remains unchanged.

These tests do NOT run a training step — confirming the wrapper
constructs end-to-end is enough to lock down the helper's contract;
heavier-weight forward+backward+step coverage lives in
:mod:`tests/protrain/test_api.py`.
"""

from __future__ import annotations

import pytest


def _tiny_gpt2(device):
    """Return a TINY GPT-2 LM head model already on ``device``.

    Mirrors the helper in :mod:`test_api.py` — the smallest HF arch the
    profiler's default batch factory (causal-LM, ``input_ids`` keyword)
    drives end-to-end without registering a custom factory.
    """
    pytest.importorskip("transformers")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=128,
        n_positions=128,
    )
    return GPT2LMHeadModel(cfg).to(device)


@pytest.mark.gpu
def test_auto_wrap_constructs_wrapped_model() -> None:
    """``auto_wrap`` returns a ready-to-use :class:`WrappedModel`."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain import auto_wrap
    from axolotl.integrations.protrain.types import WrappedModel

    device = torch.device("cuda")
    model = _tiny_gpt2(device)

    wrapped = auto_wrap(model, batch_size=2, seq_len=8)

    assert isinstance(wrapped, WrappedModel)
    assert wrapped.module is model
    assert wrapped.search_result is not None
    assert wrapped.chunk_manager is not None
    assert wrapped.scheduler is not None
    assert len(wrapped._hook_handles) > 0


@pytest.mark.gpu
def test_auto_wrap_hardware_profile_matches_device() -> None:
    """The synthesised :class:`HardwareProfile` reflects the live device."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain.api.hardware import build_hardware_profile

    hw = build_hardware_profile()

    assert hw.gpu_memory_bytes > 0
    assert hw.gpu_sku == torch.cuda.get_device_name(torch.cuda.current_device())
    assert hw.gpu_count >= 1
    # Direct API path defaults to non-sharded mode regardless of how
    # many devices are visible — ZeRO-3 is opt-in via the lower-level
    # wrapper.
    assert hw.zero3_shard is False


def test_auto_wrap_raises_when_cuda_unavailable(monkeypatch) -> None:
    """``auto_wrap`` rejects CPU-only environments with a clear error."""
    pytest.importorskip("torch")
    import torch

    from axolotl.integrations.protrain import auto_wrap

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="CUDA"):
        auto_wrap(model, batch_size=1, seq_len=4)


@pytest.mark.gpu
def test_auto_wrap_rejects_cpu_resident_model() -> None:
    """Surface the GPU-placement contract before the profiler trips on it."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain import auto_wrap

    model = torch.nn.Linear(4, 4)  # NOT moved to GPU
    with pytest.raises(RuntimeError, match="GPU"):
        auto_wrap(model, batch_size=1, seq_len=4)
