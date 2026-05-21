"""Pin the shape-preserving placeholder invariant: released params keep their logical shape so autograd records the real size."""

from __future__ import annotations

from typing import cast

import pytest

from axolotl.integrations.protrain.types import (
    BlockId,
    ParamId,
)


def _tiny_model(hidden: int = 64, n_layers: int = 4):
    """A tiny 4-layer transformer-shaped model so each ``h.{i}`` Linear becomes its own block / chunk."""
    import torch
    from torch import nn

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(hidden, hidden, bias=False)
            self.h = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)]
            )
            self.head = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.embed(x)
            for layer in self.h:
                x = layer(x)
            return self.head(x)

    torch.manual_seed(0)
    return TinyTransformer()


def _build_layout_for(model, S_chunk: int):
    from axolotl.integrations.protrain.chunk.layout import build_layout

    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        if name.startswith("h."):
            idx = int(name.split(".")[1])
            block_spans.setdefault(cast(BlockId, idx), []).append(cast(ParamId, name))

    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    return build_layout(model, exec_order, S_chunk, block_spans)


def _build_chunk_manager(
    model,
    n_persist: int,
    S_chunk: int,
    *,
    shape_preserving_placeholders: bool,
    n_buffer: int | None = None,
):
    """Assemble a :class:`ChunkManager` with the shape-preserving-placeholders flag toggled."""
    import torch

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    layout = _build_layout_for(model, S_chunk)
    if n_buffer is None:
        n_buffer = max(2, min(4, layout.N_chunk - n_persist))
    host = PinnedHostMemory(n_buffer=n_buffer, S_chunk=layout.S_chunk)
    pool = BufferPool(
        n_buffer=n_buffer,
        S_chunk=layout.S_chunk,
        pinned_host=host,
        device=torch.device("cuda"),
    )
    mgr = ChunkManager(
        model=model,
        layout=layout,
        n_persist=n_persist,
        buffer_pool=pool,
        cpu_optim=None,
        gpu_optim=None,
        device=torch.device("cuda"),
        shape_preserving_placeholders=shape_preserving_placeholders,
    )
    return mgr, layout, pool, host


def _teardown_chunk_manager(mgr, host, pool) -> None:
    """Best-effort teardown so an assertion failure cannot leak hooks, pinned-host borrows, or buffer-pool state into later tests."""
    try:
        mgr.uninstall()
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass
    try:
        host.close()
    except Exception:  # noqa: BLE001 — best-effort teardown
        pass
    # ``del pool`` drops the local reference so the GC can release
    # the pool's GPU buffer slots immediately rather than at
    # function-return.
    del pool


@pytest.mark.gpu
def test_release_state_preserves_shape() -> None:
    """With the flag on, every non-persistent param keeps its real shape after ``materialize_offload`` (not ``Size([0])``)."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    # Record the canonical shape of every named param BEFORE
    # materialize_offload — we'll compare against this snapshot below.
    original_shapes: dict[str, torch.Size] = {
        name: p.shape for name, p in model.named_parameters()
    }
    original_dtypes: dict[str, torch.dtype] = {
        name: p.dtype for name, p in model.named_parameters()
    }

    mgr, layout, pool, host = _build_chunk_manager(
        model,
        n_persist=1,
        S_chunk=S_chunk,
        shape_preserving_placeholders=True,
    )
    try:
        mgr.materialize_offload()

        # Every non-persistent chunk's params should retain their original
        # shape — the legacy code would have rebound to torch.Size([0]).
        non_persist = sorted(mgr._non_persistent_ids)
        assert non_persist, "need at least one non-persistent chunk"
        for cid in non_persist:
            for pid in layout.chunks[int(cid)]:
                param = dict(model.named_parameters())[str(pid)]
                expected_shape = original_shapes[str(pid)]
                assert param.shape == expected_shape, (
                    f"shape-preserving release violated: param={pid} "
                    f"expected shape={expected_shape}, got {param.shape}"
                )
                assert param.size() == expected_shape, (
                    f"param.size() drift: param={pid} expected {expected_shape}, "
                    f"got {param.size()}"
                )
                # dim() must reflect the original ndim too (LoRA factors
                # are 2-D; embedding is 2-D; layernorm scales are 1-D — the
                # bug surface includes shape AND dim consistency).
                assert param.dim() == len(expected_shape), (
                    f"param.dim() drift: param={pid} expected {len(expected_shape)}, "
                    f"got {param.dim()}"
                )
                assert param.dtype == original_dtypes[str(pid)], (
                    f"dtype drift: param={pid} expected {original_dtypes[str(pid)]}, "
                    f"got {param.dtype}"
                )
                assert param.device.type == "cuda", (
                    f"released param expected on cuda, got {param.device}"
                )

    finally:
        _teardown_chunk_manager(mgr, host, pool)


@pytest.mark.gpu
def test_release_state_default_off_is_unchanged() -> None:
    """Default ``shape_preserving_placeholders=False`` keeps the legacy ``numel()==0`` placeholder semantics intact."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    mgr, layout, pool, host = _build_chunk_manager(
        model,
        n_persist=1,
        S_chunk=S_chunk,
        shape_preserving_placeholders=False,
    )
    try:
        mgr.materialize_offload()

        # Legacy invariant: every non-persistent chunk's params have a
        # torch.Size([0]) placeholder after release.
        non_persist = sorted(mgr._non_persistent_ids)
        for cid in non_persist:
            for pid in layout.chunks[int(cid)]:
                param = dict(model.named_parameters())[str(pid)]
                assert param.data.numel() == 0, (
                    f"legacy invariant broken: param={pid} expected numel==0, "
                    f"got numel={param.data.numel()} shape={param.shape}"
                )

    finally:
        _teardown_chunk_manager(mgr, host, pool)


@pytest.mark.gpu
def test_gather_offload_round_trip_shape() -> None:
    """After gather→offload, released shape is preserved — confirms ``offload()`` honours the flag, not just ``materialize_offload``."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    original_shapes: dict[str, torch.Size] = {
        name: p.shape for name, p in model.named_parameters()
    }

    mgr, layout, pool, host = _build_chunk_manager(
        model,
        n_persist=1,
        S_chunk=S_chunk,
        shape_preserving_placeholders=True,
    )
    try:
        mgr.materialize_offload()

        non_persist = sorted(mgr._non_persistent_ids)
        assert non_persist, "need at least one non-persistent chunk"
        cid = non_persist[0]

        # gather → params should be at real shape with real storage
        mgr.gather(cid)
        for pid in layout.chunks[int(cid)]:
            param = dict(model.named_parameters())[str(pid)]
            assert param.shape == original_shapes[str(pid)]
            assert param.data.numel() > 0, "gathered param should have real storage"

        # offload → released; under the flag, shape must still match.
        mgr.offload(cid)
        for pid in layout.chunks[int(cid)]:
            param = dict(model.named_parameters())[str(pid)]
            assert param.shape == original_shapes[str(pid)], (
                f"post-offload shape drift on flag=True: param={pid} "
                f"expected {original_shapes[str(pid)]}, got {param.shape}"
            )

    finally:
        _teardown_chunk_manager(mgr, host, pool)


@pytest.mark.gpu
def test_storage_footprint_is_bounded() -> None:
    """The shape-preserving placeholder costs ~zero extra bytes: one 1-element scratch per dtype, shared via expand."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    mgr, layout, pool, host = _build_chunk_manager(
        model,
        n_persist=1,
        S_chunk=S_chunk,
        shape_preserving_placeholders=True,
    )
    try:
        mgr.materialize_offload()

        # Walk the released params; bucket their storage pointers by dtype.
        seen_storage_ptrs: dict[torch.dtype, set[int]] = {}
        for cid in sorted(mgr._non_persistent_ids):
            for pid in layout.chunks[int(cid)]:
                param = dict(model.named_parameters())[str(pid)]
                ptr = param.data.untyped_storage().data_ptr()
                seen_storage_ptrs.setdefault(param.dtype, set()).add(ptr)

        # For each dtype represented in the released set, every param's
        # released-state storage_ptr should equal the per-dtype scratch's
        # storage_ptr.
        for dtype, ptrs in seen_storage_ptrs.items():
            scratch = mgr._shape_scratch_by_dtype.get(dtype)
            assert scratch is not None, (
                f"no scratch cached for dtype={dtype} but released params exist"
            )
            # Scratch is 1 element wide; expand views share that storage.
            assert scratch.numel() == 1, (
                f"scratch for dtype={dtype} should be 1-element, got "
                f"numel={scratch.numel()}"
            )
            scratch_ptr = scratch.untyped_storage().data_ptr()
            assert ptrs == {scratch_ptr}, (
                f"dtype={dtype}: released params should all share scratch's "
                f"storage_ptr={scratch_ptr}, got {ptrs}"
            )

    finally:
        _teardown_chunk_manager(mgr, host, pool)


@pytest.mark.gpu
def test_autograd_shape_capture_on_released_param() -> None:
    """Direct reproducer of the autograd race: a forward over the placeholder must record the real shape, not ``[0]``."""
    pytest.importorskip("torch")
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    # Build a Parameter with a non-trivial 2D shape (mirrors a LoRA
    # factor [out_features, r]).
    real_shape = (256, 16)
    dtype = torch.bfloat16
    param = nn.Parameter(
        torch.empty(0, dtype=dtype, device="cuda")
    )  # initial "released" state

    # ---- Legacy [0] placeholder path: param.size() == [0] ----------
    assert param.shape == torch.Size([0])
    # Calling F.linear in this state fails BEFORE the autograd record
    # can complete — the kernel's shape check trips.
    x = torch.randn(4, real_shape[1], dtype=dtype, device="cuda")
    with pytest.raises(RuntimeError):
        _ = nn.functional.linear(x, param)

    # ---- Shape-preserving placeholder path: param.size() == real_shape ---
    # We construct a manager just to use the helper method
    # ``_shape_preserving_placeholder`` directly; full materialize is
    # not needed for this micro-test.
    hidden = 64
    model = _tiny_model(hidden=hidden, n_layers=2).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096
    mgr, _layout, pool, host = _build_chunk_manager(
        model,
        n_persist=1,
        S_chunk=S_chunk,
        shape_preserving_placeholders=True,
    )
    try:
        placeholder = mgr._shape_preserving_placeholder(real_shape, dtype)
        assert placeholder.shape == torch.Size(real_shape)
        assert placeholder.dtype == dtype
        assert placeholder.device.type == "cuda"
        # Storage cost: one element (the scratch).
        assert placeholder.untyped_storage().nbytes() == placeholder.element_size()

        param.data = placeholder
        assert param.shape == torch.Size(real_shape)
        assert param.size() == torch.Size(real_shape)
        assert param.dim() == 2

        # Forward must run while the placeholder is still bound so autograd records its shape (not the real-data rebind).
        x = torch.randn(
            4, real_shape[1], dtype=dtype, device="cuda", requires_grad=True
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y_placeholder = nn.functional.linear(x, param)
        # The matmul-output shape must reflect the placeholder's reported
        # weight shape; if the placeholder shrank back to ``[0]`` the
        # output would be ``(batch, 0)`` and the shape assertion below
        # would catch it BEFORE backward fires.
        assert y_placeholder.shape == torch.Size([4, real_shape[0]]), (
            f"forward through placeholder produced wrong-shape output: "
            f"expected (4, {real_shape[0]}), got {tuple(y_placeholder.shape)} — "
            f"placeholder.size() likely regressed."
        )

        # Rebind to real storage before backward; a placeholder-shape regression would surface as a ToCopyBackward0 error.
        real_data = torch.randn(*real_shape, dtype=dtype, device="cuda")
        param.data = real_data

        loss = y_placeholder.sum()
        loss.backward()
        assert param.grad is not None
        assert param.grad.shape == torch.Size(real_shape), (
            f"autograd recorded the WRONG shape: expected {real_shape}, "
            f"got {tuple(param.grad.shape)} — the shape-preserving "
            f"placeholder invariant has regressed."
        )

        # Also exercise the post-gather steady-state forward+backward
        # path so a regression that only fires on the placeholder side
        # is distinguishable from one that fires on the real-data side.
        param.grad = None
        x_real = torch.randn(
            4, real_shape[1], dtype=dtype, device="cuda", requires_grad=True
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y_real = nn.functional.linear(x_real, param)
        y_real.sum().backward()
        assert param.grad is not None
        assert param.grad.shape == torch.Size(real_shape)

    finally:
        _teardown_chunk_manager(mgr, host, pool)


@pytest.mark.gpu
def test_release_state_placeholder_is_write_unsafe() -> None:
    """The expand placeholder is NOT write-safe: any in-place write trips PyTorch's shared-storage hazard (DDP broadcast root cause)."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    model = _tiny_model(hidden=hidden, n_layers=2).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096
    mgr, _layout, pool, host = _build_chunk_manager(
        model,
        n_persist=1,
        S_chunk=S_chunk,
        shape_preserving_placeholders=True,
    )
    try:
        placeholder = mgr._shape_preserving_placeholder(
            torch.Size([hidden, hidden]), torch.float32
        )
        # Shape preserved by the placeholder.
        assert placeholder.shape == torch.Size([hidden, hidden])
        # Storage points at the per-dtype scratch (1 element).
        assert placeholder.untyped_storage().nbytes() == placeholder.element_size()

        # In-place write fails with the shared-storage hazard. Any of
        # ``copy_``, ``add_``, ``zero_``, ``mul_`` triggers it.
        real_payload = torch.zeros(hidden, hidden, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="more than one element"):
            placeholder.copy_(real_payload)

    finally:
        _teardown_chunk_manager(mgr, host, pool)


@pytest.mark.gpu
def test_chunk_managed_param_names_excludes_persistent() -> None:
    """``chunk_managed_param_names()`` returns exactly the non-persistent param names that DDP must skip on broadcast."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    mgr, layout, pool, host = _build_chunk_manager(
        model,
        n_persist=1,
        S_chunk=S_chunk,
        shape_preserving_placeholders=True,
    )
    try:
        mgr.materialize_offload()

        ignored = mgr.chunk_managed_param_names()

        # Build the expected set: every param in a non-persistent chunk.
        expected: set[str] = set()
        for cid in mgr._non_persistent_ids:
            for pid in layout.chunks[int(cid)]:
                expected.add(str(pid))
        assert ignored == expected, (
            f"chunk_managed_param_names mismatch: "
            f"expected={sorted(expected)} got={sorted(ignored)}"
        )

        # Persistent chunk params are explicitly NOT in the set.
        persistent_names: set[str] = set()
        for cid in mgr._persistent_ids:
            for pid in layout.chunks[int(cid)]:
                persistent_names.add(str(pid))
        assert ignored.isdisjoint(persistent_names), (
            f"persistent params leaked into ignore set: "
            f"intersection={ignored & persistent_names}"
        )

        # Sanity: every returned name resolves through named_parameters().
        by_name = dict(model.named_parameters())
        for name in ignored:
            assert name in by_name, f"unknown param name in ignore set: {name}"

    finally:
        _teardown_chunk_manager(mgr, host, pool)


@pytest.mark.gpu
def test_release_state_is_write_safe_through_gather_round_trip() -> None:
    """Gather must rebind ``param.data`` to fresh storage before any write so the write-unsafe placeholder is never written to."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    mgr, layout, pool, host = _build_chunk_manager(
        model,
        n_persist=1,
        S_chunk=S_chunk,
        shape_preserving_placeholders=True,
    )
    try:
        mgr.materialize_offload()

        non_persist = sorted(mgr._non_persistent_ids)
        assert non_persist, "need at least one non-persistent chunk"
        cid = non_persist[0]

        # Pre-gather: param.data IS the expand placeholder (write-unsafe).
        target_pid = str(layout.chunks[int(cid)][0])
        target_param = dict(model.named_parameters())[target_pid]
        pre_gather_storage_ptr = target_param.data.untyped_storage().data_ptr()

        # gather → param.data must rebind to a fresh typed view of the pool
        # buffer before any write reaches the placeholder.
        mgr.gather(cid)
        target_param = dict(model.named_parameters())[target_pid]
        post_gather_storage_ptr = target_param.data.untyped_storage().data_ptr()
        assert post_gather_storage_ptr != pre_gather_storage_ptr, (
            "gather did not rebind param.data — still pointing at the "
            "expand placeholder; in-place write would trip the hazard"
        )

        # Confirm the gathered param IS write-safe: an in-place fill must
        # succeed (proving the rebind landed on real storage).
        target_param.data.fill_(0.5)
        assert torch.allclose(
            target_param.data,
            torch.full_like(target_param.data, 0.5),
        ), "in-place fill on gathered param did not take effect"

        # Round-trip: offload returns to placeholder; another gather must
        # again rebind to fresh storage. This pins the cycle.
        mgr.offload(cid)
        target_param = dict(model.named_parameters())[target_pid]
        placeholder_storage_ptr = target_param.data.untyped_storage().data_ptr()
        # Re-gather and confirm the rebind happens before any write.
        mgr.gather(cid)
        target_param = dict(model.named_parameters())[target_pid]
        re_gather_storage_ptr = target_param.data.untyped_storage().data_ptr()
        assert re_gather_storage_ptr != placeholder_storage_ptr, (
            "re-gather did not rebind param.data after offload returned "
            "it to the expand placeholder"
        )

    finally:
        _teardown_chunk_manager(mgr, host, pool)
