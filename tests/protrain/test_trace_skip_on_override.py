"""Tests for the trace-pass override-skip gate (Phase 2 M5 stretch goal).

When the user supplies all four explicit-override knobs
(``protrain_n_persist_override`` / ``n_buffer_override`` /
``n_swap_override`` / ``n_checkpoint_override``), the searcher AND the
cost model are bypassed downstream by the ``all_overrides_set`` branch
in :func:`protrain_model_wrapper`. The trace pass itself becomes wasted
work, and on big-model offload configurations (e.g. 30B + 4-bit, 8B +
4-bit at seq=2048 offload) the un-offloaded trace OOMs the device
*before* chunk offload can engage. The model_wrapper short-circuits the
trace pass on this exact path; these tests pin that behaviour.

Two tests:

1. ``test_synth_trace_from_overrides_shape`` — pure unit-level: build
   the synthetic trace and assert the field shapes that downstream
   consumers depend on. CPU-only, no monkey-patching.
2. ``test_run_trace_skipped_on_override_full_path`` — end-to-end on a
   tiny GPT-2 with all four overrides set; monkey-patches ``run_trace``
   so any invocation raises immediately. Asserts the wrapper runs to
   completion. The companion ``test_run_trace_invoked_without_override``
   uses the same setup with overrides cleared and verifies ``run_trace``
   IS called.
"""

from __future__ import annotations

import importlib.util

import pytest

_SEARCH_AVAILABLE = (
    importlib.util.find_spec("axolotl.integrations.protrain.search") is not None
)
_SEARCH_SKIP_REASON = (
    "blocked on M4a search landing "
    "(axolotl.integrations.protrain.search not importable)"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hw_profile_3090():
    """Return a HardwareProfile describing an RTX 3090."""
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
    """Return a TINY GPT-2 LM head model already on ``device``.

    Matches the shape used in ``test_api.py`` so the layout discovery
    path here is identical to the existing wrapper smoke tests. 4
    layers so we have room for distinct n_swap / n_checkpoint values.
    """
    pytest.importorskip("transformers")
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(
        n_layer=4,
        n_head=2,
        n_embd=64,
        vocab_size=128,
        n_positions=128,
    )
    return GPT2LMHeadModel(cfg).to(device)


# ---------------------------------------------------------------------------
# Test 1 — pure unit: synth_trace_from_overrides field shapes
# ---------------------------------------------------------------------------


def test_synth_trace_from_overrides_shape() -> None:
    """The synthetic ``ProfilerTrace`` has the field shape downstream needs.

    CPU-only test: skips the PCIe measurement, asserts that op_order
    is empty, activation_sizes is keyed per discovered block, and
    model_state_bytes is a real (non-zero) measurement of the model's
    param + grad + optim footprint.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from axolotl.integrations.protrain.profiler.trace import (
        synth_trace_from_overrides,
    )
    from axolotl.integrations.protrain.types import ProfilerTrace

    model = _tiny_gpt2(torch.device("cpu"))
    trace = synth_trace_from_overrides(
        model,
        batch_size=2,
        seq_len=64,
        device="cpu",
        world_size=1,
        measure_pcie_bps=False,  # CPU-only test path
    )

    assert isinstance(trace, ProfilerTrace)

    # Op-order is empty — _param_exec_order falls back to named_parameters
    # declaration order, which is correct for uniform transformer stacks.
    assert trace.op_order == ()
    assert trace.intra_op_delta == {}
    assert trace.inter_op_delta == {}
    assert trace.op_latencies == {}
    assert trace.nccl_gather_s == {}
    assert trace.nccl_reduce_s == {}

    # GPT-2 with n_layer=4 should produce 4 entries in activation_sizes.
    # The discovery path may also pick up nested sub-blocks; we just
    # require >= 1 (the bounds check at model_wrapper.py:2096 needs
    # n_block >= 1) and that every value is a positive int.
    assert len(trace.activation_sizes) >= 1
    for bid, size in trace.activation_sizes.items():
        assert isinstance(size, int) and size > 0, (
            f"activation_sizes[{bid}] = {size}; expected positive int"
        )

    # model_state_bytes is a real measurement: GPT-2 with n_layer=4
    # n_embd=64 vocab=128 has roughly 80k params, so ~80k * 16 B (default
    # param+grad+optim per fp16+adam) ≈ 1.3 MB. Bounds-check liberally:
    assert trace.model_state_bytes > 0
    assert trace.model_state_bytes < 100 * (1 << 20)  # < 100 MB sanity

    # PCIe defaults when measure_pcie_bps=False: 13 GB/s Gen3 prior.
    assert trace.pcie_h2d_bps == pytest.approx(13e9)
    assert trace.pcie_d2h_bps == pytest.approx(13e9)

    # Cache key fields populated.
    assert trace.bs == 2
    assert trace.seq == 64
    assert trace.world == 1
    assert isinstance(trace.arch_hash, str) and len(trace.arch_hash) == 64

    # Phase-2 / chunked-runtime fields default to "no measurement"
    # sentinels so the cost model collapses to its v8-or-earlier path.
    assert trace.cpu_adam_bytes_per_sec == 0.0
    assert trace.gpu_adam_bytes_per_sec == 0.0
    assert trace.steady_bwd_chunked_wall_s == 0.0


# ---------------------------------------------------------------------------
# Test 2 — end-to-end: run_trace is NOT called when all four overrides set
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _SEARCH_AVAILABLE, reason=_SEARCH_SKIP_REASON)
def test_run_trace_skipped_on_override_full_path(
    gpu_device, monkeypatch, tmp_path
) -> None:  # noqa: ARG001 — gpu_device fixture activates CUDA masking
    """``run_trace`` MUST NOT be called when all four overrides are set.

    Monkey-patches ``run_trace`` to raise immediately if invoked. The
    wrapper must complete by going through the synthetic-trace path.
    Uses a fresh ``cache_dir=tmp_path`` to guarantee a cache miss (so
    we exercise the override-skip branch rather than the cache-hit
    branch which would also avoid the trace pass).
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.api import (
        model_wrapper as model_wrapper_mod,
        protrain_model_wrapper,
    )
    from axolotl.integrations.protrain.types import WrappedModel

    def _exploding_run_trace(*args, **kwargs):  # noqa: ARG001
        raise AssertionError(
            "run_trace was called on the override-skip path; this is the bug "
            "the trace-pass override-skip gate is supposed to prevent."
        )

    monkeypatch.setattr(model_wrapper_mod, "run_trace", _exploding_run_trace)

    device = torch.device("cuda")
    model = _tiny_gpt2(device)
    hw = _hw_profile_3090()

    # Pick valid override values: persist all chunks, no offload — the
    # SearchResult synthesizer in model_wrapper.py:2140 enforces
    # ``n_swap + n_checkpoint <= N_block`` and ``min_n_buffer_for``
    # invariants. We use the safe "all-persistent" pattern that
    # matches the test_swap.py override pattern.
    #
    # R3-#8: compute N_chunk and N_block dynamically rather than
    # hard-coding ``n_chunk_estimate=1``. If the layout builder /
    # block-discovery heuristics shift (e.g. S_chunk default
    # changes, or block discovery starts pulling in embed/norm as
    # blocks), a hard-coded ``1`` would fail ``min_n_buffer_for``'s
    # validation before we even reach the trace-skip gate the test
    # is supposed to validate — turning this into a flaky
    # non-target failure. The dynamic values mirror what the
    # production wrapper itself computes one layer up.
    from axolotl.integrations.protrain.block.layout_rules import (
        discover_blocks,
        flatten_block_trees,
    )
    from axolotl.integrations.protrain.chunk.layout import build_layout

    discovered = discover_blocks(model)
    flat_blocks = flatten_block_trees(discovered)
    n_block_estimate = len(flat_blocks)
    # Build a layout exactly the way ``protrain_model_wrapper`` does
    # (same S_chunk pick + same block_spans derivation) so the
    # ``n_persist_override == N_chunk`` invariant we want to assert
    # downstream actually holds. ``cfg.num_hidden_layers=4`` produces
    # block_spans for layers 0..3 + embeddings — but the chunk
    # builder operates over named_parameters().
    block_spans: dict = {}
    for name, param in model.named_parameters():
        # Find which block (if any) this param belongs to via the
        # discovered block list.
        for block_idx, block_module in enumerate(flat_blocks):
            if any(p is param for p in block_module.parameters()):
                from axolotl.integrations.protrain.types import (
                    BlockId,
                    ParamId,
                )

                block_spans.setdefault(BlockId(block_idx), []).append(ParamId(name))
                break
    from typing import cast as _cast

    from axolotl.integrations.protrain.types import ParamId as _ParamId

    exec_order = [_cast(_ParamId, n) for n, _ in model.named_parameters()]
    # 4 MiB S_chunk matches the wrapper's default for tiny models;
    # the exact value isn't load-bearing as long as the same value is
    # used inside ``protrain_model_wrapper`` (which it will be, since
    # the override path also takes the wrapper's default S_chunk).
    layout = build_layout(model, exec_order, 4 << 20, block_spans)
    n_chunk_estimate = layout.N_chunk

    wrapped = protrain_model_wrapper(
        model,
        model_config=None,
        hardware_profile=hw,
        batch_size=2,
        seq_len=64,
        capacity_bytes=1 << 30,
        cache_dir=str(tmp_path),  # force cache miss
        n_persist_override=n_chunk_estimate,
        n_buffer_override=0,
        n_swap_override=0,
        n_checkpoint_override=n_block_estimate,
        n_offload_override=0,
        auto_mode=False,
    )

    assert isinstance(wrapped, WrappedModel)
    # The override path's SearchResult round-trips into the wrapper.
    assert wrapped.search_result is not None
    assert wrapped.search_result.cfg.n_swap == 0
    # n_checkpoint is bounded by N_block which is what activation_sizes
    # maps; the synthetic trace populates one entry per discovered
    # block. The wrapper accepted the override so the bounds check
    # passed — sanity check that we land at n_block from the synth.
    assert wrapped.search_result.cfg.n_checkpoint <= n_block_estimate

    # Tear down to release CUDA state for the next test.
    wrapped.close()


@pytest.mark.gpu
@pytest.mark.skipif(not _SEARCH_AVAILABLE, reason=_SEARCH_SKIP_REASON)
def test_run_trace_invoked_without_override(gpu_device, monkeypatch, tmp_path) -> None:  # noqa: ARG001 — gpu_device fixture activates CUDA masking
    """The control: same setup WITHOUT overrides ⇒ ``run_trace`` IS called.

    Wraps ``run_trace`` with a counter so we can assert it ran exactly
    once. Otherwise the override-skip test above could pass trivially
    if the wrapper somehow stopped calling ``run_trace`` on every path.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.api import (
        model_wrapper as model_wrapper_mod,
        protrain_model_wrapper,
    )
    from axolotl.integrations.protrain.types import WrappedModel

    call_count = {"n": 0}
    real_run_trace = model_wrapper_mod.run_trace

    def _counting_run_trace(*args, **kwargs):
        call_count["n"] += 1
        return real_run_trace(*args, **kwargs)

    monkeypatch.setattr(model_wrapper_mod, "run_trace", _counting_run_trace)

    device = torch.device("cuda")
    model = _tiny_gpt2(device)
    hw = _hw_profile_3090()

    wrapped = protrain_model_wrapper(
        model,
        model_config=None,
        hardware_profile=hw,
        batch_size=2,
        seq_len=64,
        capacity_bytes=1 << 30,
        cache_dir=str(tmp_path),  # force cache miss
        # No overrides → searcher path → run_trace must fire.
        auto_mode=False,
    )

    assert isinstance(wrapped, WrappedModel)
    assert call_count["n"] == 1, (
        f"run_trace was called {call_count['n']} times; expected exactly 1 "
        "on the searcher path with a fresh cache_dir"
    )

    wrapped.close()


# ---------------------------------------------------------------------------
# Test 3 — partial overrides do NOT skip the trace pass
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not _SEARCH_AVAILABLE, reason=_SEARCH_SKIP_REASON)
def test_partial_overrides_do_not_skip_trace(gpu_device, monkeypatch, tmp_path) -> None:  # noqa: ARG001 — gpu_device fixture activates CUDA masking
    """A SUBSET of overrides (e.g. only n_persist) must NOT trigger the skip.

    The override-skip gate requires ALL FOUR knobs; partial specifications
    are documented to be ignored on the searcher path. We pin that here:
    setting only ``n_persist_override`` should still invoke ``run_trace``
    (and the searcher), matching the documented contract on the pydantic
    field at ``args.py``.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.api import (
        model_wrapper as model_wrapper_mod,
        protrain_model_wrapper,
    )
    from axolotl.integrations.protrain.types import WrappedModel

    call_count = {"n": 0}
    real_run_trace = model_wrapper_mod.run_trace

    def _counting_run_trace(*args, **kwargs):
        call_count["n"] += 1
        return real_run_trace(*args, **kwargs)

    monkeypatch.setattr(model_wrapper_mod, "run_trace", _counting_run_trace)

    device = torch.device("cuda")
    model = _tiny_gpt2(device)
    hw = _hw_profile_3090()

    wrapped = protrain_model_wrapper(
        model,
        model_config=None,
        hardware_profile=hw,
        batch_size=2,
        seq_len=64,
        capacity_bytes=1 << 30,
        cache_dir=str(tmp_path),
        n_persist_override=1,  # only ONE override set
        # The other three knobs are None ⇒ partial override ⇒ NO skip.
        auto_mode=False,
    )

    assert isinstance(wrapped, WrappedModel)
    assert call_count["n"] == 1, (
        f"run_trace was called {call_count['n']} times; expected exactly 1 "
        "with partial overrides (only n_persist set)"
    )

    wrapped.close()
