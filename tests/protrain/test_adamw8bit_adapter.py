"""Unit tests for ``GpuAdamW8bitAdapter`` construction, state round-trip, and the wrapper dispatch path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest

from axolotl.integrations.protrain.chunk.optim import (
    GpuAdamW8bitAdapter,
    GpuFusedAdamAdapter,
)

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")


pytestmark = pytest.mark.gpu


def _gpu_device() -> "torch.device":
    """Pick a CUDA device that respects ``CUDA_VISIBLE_DEVICES`` and skip cleanly when CUDA is absent."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; test_adamw8bit_adapter requires GPU.")
    return torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Adapter unit tests
# ---------------------------------------------------------------------------


def test_adapter_state_shapes_after_step() -> None:
    """After one step, per-param state must carry the bnb 8-bit moments."""
    bnb = pytest.importorskip("bitsandbytes")
    device = _gpu_device()
    # min_8bit_size defaults to 4096 — we need enough elements per param
    # for bnb to actually 8-bit-quantize the state (smaller params fall
    # back to fp32 state internally and ``state1.dtype`` would be float).
    p = torch.nn.Parameter(torch.randn(128, 128, dtype=torch.float32, device=device))
    adapter = GpuAdamW8bitAdapter(
        params=[p],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    p.grad = torch.randn_like(p)
    adapter.step()

    state = adapter.underlying.state[p]
    assert state["state1"].dtype == torch.uint8
    assert state["state2"].dtype == torch.uint8
    assert state["state1"].shape == p.shape
    assert state["state2"].shape == p.shape
    # Codebooks (256-entry quantization maps) and absmax block scales.
    assert state["qmap1"].shape == (256,)
    assert state["qmap2"].shape == (256,)
    assert state["absmax1"].numel() > 0
    assert state["absmax2"].numel() > 0
    # ``bnb`` is imported by the adapter; keep the reference alive for
    # the assertions to be non-trivial under some lazy-import paths.
    assert bnb is not None


def test_state_dict_round_trip_preserves_8bit_state() -> None:
    """state_dict -> new adapter -> load_state_dict preserves uint8 moments."""
    pytest.importorskip("bitsandbytes")
    device = _gpu_device()
    torch.manual_seed(123)
    p1 = torch.nn.Parameter(torch.randn(256, 256, dtype=torch.float32, device=device))
    adapter1 = GpuAdamW8bitAdapter(params=[p1], lr=1e-3)
    p1.grad = torch.randn_like(p1)
    adapter1.step()

    state1_before = adapter1.underlying.state[p1]["state1"].clone()
    state2_before = adapter1.underlying.state[p1]["state2"].clone()
    qmap1_before = adapter1.underlying.state[p1]["qmap1"].clone()
    absmax1_before = adapter1.underlying.state[p1]["absmax1"].clone()
    sd = adapter1.state_dict()

    # Fresh adapter, identical params, load the saved state.
    p2 = torch.nn.Parameter(p1.detach().clone())
    adapter2 = GpuAdamW8bitAdapter(params=[p2], lr=1e-3)
    adapter2.load_state_dict(sd)

    state1_after = adapter2.underlying.state[p2]["state1"]
    state2_after = adapter2.underlying.state[p2]["state2"]
    qmap1_after = adapter2.underlying.state[p2]["qmap1"]
    absmax1_after = adapter2.underlying.state[p2]["absmax1"]
    assert torch.equal(state1_before, state1_after)
    assert torch.equal(state2_before, state2_after)
    assert torch.equal(qmap1_before, qmap1_after)
    assert torch.equal(absmax1_before, absmax1_after)


def test_cpu_param_raises_clear_error() -> None:
    """Constructing the adapter with CPU params must surface the bail condition."""
    pytest.importorskip("bitsandbytes")
    p = torch.nn.Parameter(torch.randn(128, 128, dtype=torch.float32, device="cpu"))
    with pytest.raises(RuntimeError) as exc_info:
        GpuAdamW8bitAdapter(params=[p], lr=1e-3)
    msg = str(exc_info.value)
    assert "CUDA" in msg
    assert "non-persistent" in msg
    assert "M2.5" in msg or "CpuFusedAdamAdapter" in msg


def test_empty_param_set_is_no_op() -> None:
    """Mode-C with no persistent chunks: empty adapter must short-circuit cleanly."""
    pytest.importorskip("bitsandbytes")
    adapter = GpuAdamW8bitAdapter(params=[], lr=1e-3)
    # No underlying optimizer.
    assert adapter.underlying is None
    # step / zero_grad are silent no-ops; state_dict returns the
    # canonical empty shape.
    adapter.step()
    adapter.zero_grad()
    sd = adapter.state_dict()
    assert sd == {"state": {}, "param_groups": []}
    # load_state_dict accepts the matching empty shell silently.
    adapter.load_state_dict({"state": {}, "param_groups": []})
    # ...but rejects a non-empty payload (Mode-A/Mode-C config mismatch).
    with pytest.raises(ValueError):
        adapter.load_state_dict({"state": {0: {"step": 1}}, "param_groups": []})


def test_paged_variant_constructs_paged_class() -> None:
    """``paged=True`` must instantiate ``bnb.optim.PagedAdamW8bit``."""
    bnb = pytest.importorskip("bitsandbytes")
    device = _gpu_device()
    p = torch.nn.Parameter(torch.randn(128, 128, dtype=torch.float32, device=device))
    adapter = GpuAdamW8bitAdapter(params=[p], lr=1e-3, paged=True)
    assert isinstance(adapter.underlying, bnb.optim.PagedAdamW8bit)


def test_step_actually_updates_params() -> None:
    """One step should mutate ``param.data`` (sanity-check that the kernel ran)."""
    pytest.importorskip("bitsandbytes")
    device = _gpu_device()
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(128, 128, dtype=torch.float32, device=device))
    p_before = p.detach().clone()
    adapter = GpuAdamW8bitAdapter(params=[p], lr=1e-2)
    p.grad = torch.ones_like(p)
    adapter.step()
    # AdamW with positive grads + positive LR moves params toward zero on the
    # first step; the deltas are non-zero everywhere.
    assert not torch.equal(p.detach(), p_before)


# ---------------------------------------------------------------------------
# Dispatch test — protrain_optimizer_wrapper routing
# ---------------------------------------------------------------------------


class _FakeChunkLayout:
    """Minimal stand-in for ``ChunkLayout`` exposing only the ``chunks`` field the wrapper iterates."""

    def __init__(self, chunks: list[list[int]]) -> None:
        self.chunks = chunks


class _FakeChunkManager:
    """Minimal stand-in for ``ChunkManager`` for the dispatch test."""

    def __init__(
        self,
        params_by_id: dict[int, torch.nn.Parameter],
        persistent_ids: set[int],
        chunks: list[list[int]],
    ) -> None:
        self.layout = _FakeChunkLayout(chunks)
        self._params_by_id = params_by_id
        self._persistent_ids = persistent_ids
        self._non_persistent_ids = {
            cid for cid, _ in enumerate(chunks) if cid not in persistent_ids
        }
        self._chunk_shards: dict[int, Any] = {}
        self._cpu_slots: dict[int, list[Any]] = {}
        # cpu_optim / gpu_optim are written by the wrapper at the end.
        self.cpu_optim = None
        self.gpu_optim = None
        self.zero3_shard = False


def _build_dispatch_fixture(
    n_persistent_params: int = 1,
    n_cpu_params: int = 0,
) -> tuple[Any, list[torch.nn.Parameter]]:
    """Build a tiny WrappedModel + persistent-only chunk layout on CUDA."""
    device = _gpu_device()
    persistent = [
        torch.nn.Parameter(torch.randn(128, 128, dtype=torch.float32, device=device))
        for _ in range(n_persistent_params)
    ]
    cpu_params = [
        torch.nn.Parameter(torch.randn(128, 128, dtype=torch.float32, device="cpu"))
        for _ in range(n_cpu_params)
    ]
    all_params = persistent + cpu_params
    params_by_id = {i: p for i, p in enumerate(all_params)}
    chunks = [[i] for i in range(len(all_params))]
    persistent_ids = set(range(n_persistent_params))

    cm = _FakeChunkManager(
        params_by_id=params_by_id,
        persistent_ids=persistent_ids,
        chunks=chunks,
    )
    # ``module`` is consulted by ``_collect_no_decay_param_ids``; an empty
    # nn.Module has no params, so the no-decay set is empty (acceptable
    # for this dispatch test).
    module = torch.nn.Module()
    wrapped = SimpleNamespace(
        module=module,
        chunk_manager=cm,
    )
    return wrapped, persistent


@pytest.mark.parametrize(
    "optim_name",
    ["adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit"],
)
def test_dispatch_routes_8bit_names_to_bnb_adapter(optim_name: str) -> None:
    """All three Axolotl/HF 8-bit names route persistent set through the bnb adapter."""
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("deepspeed")  # CpuFusedAdam path import — ok if missing? skip
    from axolotl.integrations.protrain.api.optim_wrapper import (
        protrain_optimizer_wrapper,
    )

    wrapped, _persistent = _build_dispatch_fixture(
        n_persistent_params=1,
        n_cpu_params=0,
    )
    optim = protrain_optimizer_wrapper(
        wrapped,
        lr=1e-3,
        optimizer_name=optim_name,
    )
    # Inner adapter must be the 8-bit variant.
    assert isinstance(optim._gpu_optim, GpuAdamW8bitAdapter)
    if optim_name == "paged_adamw_8bit":
        assert optim._gpu_optim.paged is True
    else:
        assert optim._gpu_optim.paged is False
    # No CPU chunks in this fixture, so cpu_optim is None.
    assert optim._cpu_optim is None


def test_dispatch_default_optimizer_uses_fused_adam() -> None:
    """``optimizer_name=None`` (and unrelated names) keeps the GpuFusedAdamAdapter path."""
    pytest.importorskip("bitsandbytes")
    from axolotl.integrations.protrain.api.optim_wrapper import (
        protrain_optimizer_wrapper,
    )

    wrapped, _persistent = _build_dispatch_fixture(
        n_persistent_params=1,
        n_cpu_params=0,
    )
    # Default / non-8bit name: persistent set must use the legacy path.
    optim = protrain_optimizer_wrapper(
        wrapped,
        lr=1e-3,
        optimizer_name="adamw_torch",
    )
    assert isinstance(optim._gpu_optim, GpuFusedAdamAdapter)


def test_dispatch_warns_when_8bit_requested_with_cpu_chunks() -> None:
    """Bail-condition warning fires when 8-bit + non-persistent chunks coexist."""
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("deepspeed")
    from axolotl.integrations.protrain.api.optim_wrapper import (
        protrain_optimizer_wrapper,
    )

    wrapped, _persistent = _build_dispatch_fixture(
        n_persistent_params=1,
        n_cpu_params=1,
    )
    # CpuFusedAdamAdapter requires DeepSpeed's compiled CPU Adam kernel —
    # under DS_SKIP_CUDA_CHECK this is JIT-built on demand. Stub it so
    # this test does not depend on the local DS build state.
    captured_warnings: list[str] = []

    def _capture_warning(msg, *args, **kwargs):
        # ``LOG.warning`` from the wrapper uses %-style formatting.
        try:
            captured_warnings.append(msg % args if args else msg)
        except (TypeError, ValueError):
            captured_warnings.append(str(msg))

    with mock.patch(
        "axolotl.integrations.protrain.chunk.optim.CpuFusedAdamAdapter",
        autospec=True,
    ) as fake_cpu_cls:
        fake_cpu_cls.return_value = mock.MagicMock(_optims={})
        with mock.patch(
            "axolotl.integrations.protrain.api.optim_wrapper.CpuFusedAdamAdapter",
            fake_cpu_cls,
        ):
            with mock.patch(
                "axolotl.integrations.protrain.api.optim_wrapper.LOG.warning",
                side_effect=_capture_warning,
            ):
                _optim = protrain_optimizer_wrapper(
                    wrapped,
                    lr=1e-3,
                    optimizer_name="adamw_8bit",
                )
    # The bail-condition warning must surface.
    assert any(
        "8-bit Adam kernels are CUDA-only" in msg for msg in captured_warnings
    ), captured_warnings


# End-to-end smoke: full ProTrain pipeline with adamw_8bit on tiny GPT-2.


def _tiny_gpt2(device):
    """Smallest HF causal-LM the profiler's batch factory drives end-to-end."""
    pytest.importorskip("transformers")
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


@pytest.mark.slow
def test_end_to_end_5_steps_descending_loss() -> None:
    """5 forward+backward+step iterations on tiny GPT-2 with adamw_8bit yield descending loss."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("bitsandbytes")

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from axolotl.integrations.protrain import auto_wrap
    from axolotl.integrations.protrain.api.optim_wrapper import (
        protrain_optimizer_wrapper,
    )

    device = torch.device("cuda")
    model = _tiny_gpt2(device)

    wrapped = auto_wrap(model, batch_size=2, seq_len=8)
    try:
        optim = protrain_optimizer_wrapper(
            wrapped,
            lr=1e-2,  # high enough to see loss move in 5 steps
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            optimizer_name="adamw_8bit",
        )
        # Persistent set on tiny model routes to the 8-bit adapter; no CPU chunks in Mode A.
        assert isinstance(optim._gpu_optim, GpuAdamW8bitAdapter), (
            f"expected GpuAdamW8bitAdapter, got {type(optim._gpu_optim).__name__}"
        )

        # Overfit a single fixed batch so per-iter noise cannot mask the descent.
        torch.manual_seed(42)
        fixed_input = torch.randint(0, 128, (2, 8), device=device)
        losses: list[float] = []
        for _ in range(5):
            out = wrapped.module(input_ids=fixed_input, labels=fixed_input)
            loss = out.loss
            losses.append(float(loss.detach()))
            loss.backward()
            optim.step()
            optim.zero_grad()

        assert len(losses) == 5
        assert all(loss > 0 for loss in losses), f"non-positive loss: {losses}"
        assert losses[-1] < losses[0], f"loss did not descend: {losses}"
    finally:
        # Release CUDA/chunk resources so a failure cannot leak into later GPU tests.
        wrapped.close()
