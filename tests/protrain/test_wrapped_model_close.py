"""Regression tests for the canonical :meth:`WrappedModel.close` cascade.

The close cascade releases every wrapper-owned resource so a torn-down
runtime can be safely replaced (auto-mode rebuild, test-fixture reuse,
end-of-epoch cleanup) without leaking pinned host pools, CPU-Adam
worker threads, or grad hooks. These tests pin the contract.
"""

from __future__ import annotations

import pytest


def _hw_profile_3090():
    from axolotl.integrations.protrain.types import HardwareProfile

    return HardwareProfile(
        gpu_sku="NVIDIA GeForce RTX 3090",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=16.0 * (1 << 30),
        pcie_d2h_bps=16.0 * (1 << 30),
        has_nvlink=False,
    )


def _tiny_gpt2(device):
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


def _wrap_tiny(device, *, capacity_bytes=1 << 30):
    """Build a fresh tiny GPT-2 + ``WrappedModel`` on ``device``."""
    from axolotl.integrations.protrain.api import protrain_model_wrapper

    model = _tiny_gpt2(device)
    hw = _hw_profile_3090()
    wrapped = protrain_model_wrapper(
        model,
        model_config=None,
        hardware_profile=hw,
        batch_size=2,
        seq_len=128,
        capacity_bytes=capacity_bytes,
    )
    return model, wrapped


def _has_protrain_grad_hooks(model) -> bool:
    """True iff any param still carries a ChunkManager-installed grad hook."""
    for param in model.parameters():
        # ``register_post_accumulate_grad_hook`` populates
        # ``_post_accumulate_grad_hooks`` — a ``RemovableHandle``-keyed
        # OrderedDict on each Parameter. ``handle.remove()`` empties it,
        # so a non-empty dict signals a missed remove.
        hooks = getattr(param, "_post_accumulate_grad_hooks", None)
        if hooks:
            return True
    return False


@pytest.mark.gpu
def test_wrapped_model_close_releases_pinned_host_pool() -> None:
    """``close()`` drops the pinned-host param/grad pools."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    device = torch.device("cuda")
    _, wrapped = _wrap_tiny(device)

    chunk_manager = wrapped.chunk_manager

    # Snapshot the pool refs before close so we can assert post-close
    # they are released (close() nulls the manager attrs, so we capture
    # them up front).
    cpu_param_pool = getattr(chunk_manager, "_cpu_param_pool", None)
    cpu_grad_pool = getattr(chunk_manager, "_cpu_grad_pool", None)
    buffer_pool = getattr(chunk_manager, "buffer_pool", None)

    wrapped.close()

    # Manager attributes nulled.
    assert wrapped.chunk_manager is None
    assert wrapped.scheduler is None

    # If the layout had non-persistent chunks the pinned-host param pool
    # was populated — assert it's been freed (``_closed`` is the public
    # post-condition signal on :class:`PinnedHostMemory`). The
    # all-persistent path leaves the pool ``None`` to begin with, which
    # is also a valid post-condition.
    if cpu_param_pool is not None:
        assert cpu_param_pool._closed is True, (
            "cpu_param_pool not closed after WrappedModel.close()"
        )
    if cpu_grad_pool is not None:
        assert cpu_grad_pool._closed is True, (
            "cpu_grad_pool not closed after WrappedModel.close()"
        )

    # ChunkManager's own buffer-pool attribute should be None and the
    # captured pool should be marked closed.
    assert chunk_manager.buffer_pool is None
    if buffer_pool is not None:
        assert buffer_pool._closed is True, (
            "buffer_pool not closed after WrappedModel.close()"
        )


@pytest.mark.gpu
def test_close_is_idempotent() -> None:
    """Calling ``close()`` twice does not raise."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    device = torch.device("cuda")
    _, wrapped = _wrap_tiny(device)

    wrapped.close()
    # Second call must be a no-op — no exceptions, fields stay None.
    wrapped.close()
    assert wrapped.chunk_manager is None
    assert wrapped.scheduler is None
    assert wrapped._closed is True


@pytest.mark.gpu
def test_grad_hooks_removed_after_close() -> None:
    """Model parameters carry no protrain grad hooks after close."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    device = torch.device("cuda")
    model, wrapped = _wrap_tiny(device)

    # Sanity: at least one hook is installed pre-close (``materialize_offload``
    # registers per-param post-accumulate hooks for non-persistent chunks).
    # When the layout ends up all-persistent there are no such hooks; in
    # that case the pre-condition is vacuously satisfied and the
    # post-condition still holds.
    pre = _has_protrain_grad_hooks(model)

    wrapped.close()
    assert not _has_protrain_grad_hooks(model), (
        "post-accumulate grad hooks remain on parameters after close()"
    )
    # The wrapper's own hook handle list is cleared.
    assert wrapped._hook_handles == []
    # If hooks were installed pre-close, the post-close clearing is the
    # load-bearing assertion above. Either way, the wrapper's _closed
    # flag is set.
    assert wrapped._closed is True
    # Surface the pre-close state for debugging (no-op in passing runs).
    del pre


@pytest.mark.gpu
def test_re_wrap_after_close_works() -> None:
    """A fresh wrap on a fresh model after close runs one fwd/bwd cycle."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )

    device = torch.device("cuda")

    # First wrap — drive a forward + backward to exercise the full
    # runtime, then tear down.
    model_a = _tiny_gpt2(device)
    hw = _hw_profile_3090()
    wrapped_a = protrain_model_wrapper(
        model_a,
        model_config=None,
        hardware_profile=hw,
        batch_size=2,
        seq_len=128,
        capacity_bytes=1 << 30,
    )
    optim_a = protrain_optimizer_wrapper(wrapped_a, lr=1e-3)
    input_ids = torch.randint(0, 128, (2, 128), device=device, dtype=torch.long)
    optim_a.zero_grad()
    out = model_a(input_ids=input_ids, labels=input_ids.clone())
    out.loss.backward()
    optim_a.step()

    # Tear down. ``optim_a`` references the now-closed adapters; drop it
    # before close so its __del__ (which calls cpu_optim.shutdown) can't
    # race the cascade.
    del optim_a
    wrapped_a.close()
    del wrapped_a, model_a
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Second wrap on a fresh model — should construct cleanly and drive
    # a fwd/bwd without surfacing leaked state from the first wrap.
    model_b = _tiny_gpt2(device)
    wrapped_b = protrain_model_wrapper(
        model_b,
        model_config=None,
        hardware_profile=hw,
        batch_size=2,
        seq_len=128,
        capacity_bytes=1 << 30,
    )
    try:
        optim_b = protrain_optimizer_wrapper(wrapped_b, lr=1e-3)
        optim_b.zero_grad()
        out_b = model_b(input_ids=input_ids, labels=input_ids.clone())
        out_b.loss.backward()
        optim_b.step()
        del optim_b
    finally:
        wrapped_b.close()
