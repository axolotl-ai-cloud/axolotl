"""Tests for the ProTrain M4b public API wrappers (api/).

These tests exercise the full composition pipeline: profiler (cached)
-> layout -> searcher -> chunk manager -> scheduler -> wrapped model.
They do NOT run a training iteration on a real model — the M4b agent's
integration test lives under ``tests/protrain/integration/`` once the
7B smoke test lands.
"""

from __future__ import annotations

import importlib.util

import pytest

# ---------------------------------------------------------------------------
# Serialization guard: the searcher is written by a parallel agent. If it
# hasn't landed at test time, skip the smoke tests instead of failing.
# Production code imports ``search`` at module load so this only affects
# local test runs — the production import is unconditional.
# ---------------------------------------------------------------------------
_SEARCH_AVAILABLE = (
    importlib.util.find_spec("axolotl.integrations.protrain.search") is not None
)

_SEARCH_SKIP_REASON = (
    "blocked on M4a search landing "
    "(axolotl.integrations.protrain.search not importable)"
)


def _hw_profile_3090():
    """Return a HardwareProfile describing an RTX 3090."""
    from axolotl.integrations.protrain.types import HardwareProfile

    return HardwareProfile(
        gpu_sku="NVIDIA GeForce RTX 3090",
        gpu_memory_bytes=24 * (1 << 30),  # 24 GiB
        gpu_count=1,
        pcie_h2d_bps=16.0 * (1 << 30),  # PCIe 4.0 x16 nominal
        pcie_d2h_bps=16.0 * (1 << 30),
        has_nvlink=False,
    )


def _tiny_gpt2(device):
    """Return a TINY GPT-2 LM head model already on ``device``."""
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


# ---------------------------------------------------------------------------
# Wrapper smoke test — composes the full pipeline without running training.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _SEARCH_AVAILABLE, reason=_SEARCH_SKIP_REASON)
def test_protrain_wrapper_smoke(gpu_device):  # noqa: ARG001 — fixture activates CUDA masking
    """``protrain_model_wrapper`` composes profiler+search+runtime end-to-end."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.api import protrain_model_wrapper
    from axolotl.integrations.protrain.types import WrappedModel

    device = torch.device("cuda")
    model = _tiny_gpt2(device)
    hw = _hw_profile_3090()

    wrapped = protrain_model_wrapper(
        model,
        model_config=None,
        hardware_profile=hw,
        batch_size=2,
        seq_len=128,
        capacity_bytes=1 << 30,
    )

    assert isinstance(wrapped, WrappedModel)
    assert wrapped.module is model
    assert wrapped.chunk_manager is not None
    assert wrapped.scheduler is not None
    assert wrapped.search_result is not None
    assert len(wrapped._hook_handles) > 0


# ---------------------------------------------------------------------------
# Optimizer smoke test — verify forward+backward+step actually mutates params.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _SEARCH_AVAILABLE, reason=_SEARCH_SKIP_REASON)
def test_protrain_optimizer_zero_grad_and_step_shapes(gpu_device):  # noqa: ARG001
    """A single fwd+bwd+step cycle updates at least one parameter."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )

    device = torch.device("cuda")
    model = _tiny_gpt2(device)
    hw = _hw_profile_3090()

    wrapped = protrain_model_wrapper(
        model,
        model_config=None,
        hardware_profile=hw,
        batch_size=2,
        seq_len=128,
        capacity_bytes=1 << 30,
    )

    optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)

    # Snapshot a parameter pre-step for the "parameters change" assertion.
    (name, param) = next(iter(model.named_parameters()))
    before = param.detach().clone()

    # Build a trivial batch and run fwd + bwd.
    input_ids = torch.randint(0, 128, (2, 128), device=device, dtype=torch.long)
    labels = input_ids.clone()
    optim.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    optim.step()

    after = param.detach()
    changed = not torch.allclose(before, after)
    assert changed, (
        f"parameter {name!r} unchanged after optim.step() — "
        "update path did not reach it"
    )


# ---------------------------------------------------------------------------
# Capacity-too-small — searcher must raise RuntimeError.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _SEARCH_AVAILABLE, reason=_SEARCH_SKIP_REASON)
def test_protrain_wrapper_raises_if_capacity_too_small():
    """An absurdly small ``capacity_bytes`` forces the searcher to raise."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.api import protrain_model_wrapper

    device = torch.device("cuda")
    model = _tiny_gpt2(device)
    hw = _hw_profile_3090()

    with pytest.raises(RuntimeError):
        protrain_model_wrapper(
            model,
            model_config=None,
            hardware_profile=hw,
            batch_size=2,
            seq_len=128,
            capacity_bytes=1 << 10,
        )
