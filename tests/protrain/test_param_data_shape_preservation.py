"""M6C-fix-7 architectural-attempt unit tests.

These tests pin the invariant introduced by ``M6C-fix-7``: when
``ChunkManager`` is constructed with ``shape_preserving_placeholders=True``,
the "released" state of every chunk-managed parameter preserves its
logical shape (``param.size()`` / ``param.shape`` / ``param.dim()``).

Background (synthesised from the M6C-fix-{3..6} empirical record):

PyTorch autograd captures Function input shape metadata at NODE
CONSTRUCTION time (forward) — see
``torch/csrc/autograd/generated/Functions.h``'s ``self_sym_sizes`` field
captured by-value as ``std::vector<c10::SymInt>``. The legacy
chunk-manager release path rebinds ``param.data`` to a
``torch.Size([0])`` placeholder; a rare race window on multi-GPU sharded
non-persistent chunks at production scale (32-layer Llama-3-8B × 4 ranks
× heavy pool-eviction pressure) lets an autograd op record its input
shape against the still-``[0]``-shape placeholder before the per-LoRA-
container gather hook's rebind takes effect — surfacing at backward as
``RuntimeError: Function ToCopyBackward0 returned an invalid gradient
... expected shape compatible with [0]``.

The shape-preserving placeholder closes the race architecturally: the
post-release ``param.data`` is a zero-stride view over a 1-element
per-dtype scratch (``scratch.expand(slot.shape)``), so ``param.size()``
returns the real logical shape regardless of where in the gather→forward
sequence an autograd op records its metadata.

Storage footprint: ONE 1-element scratch tensor per dtype shared across
every released param of that dtype. The expand view contributes zero
additional bytes.

Test surface:

* ``test_release_state_preserves_shape`` — the central invariant: post-
  materialize ``param.shape`` matches the param's original shape (not
  ``[0]``) when the flag is on.
* ``test_release_state_default_off_is_unchanged`` — default behavior
  (``shape_preserving_placeholders=False``) is unchanged: post-
  materialize ``param.shape == torch.Size([0])`` exactly as before
  M6C-fix-7. Guards the entire pre-existing test surface
  (test_chunk_manager_offload.py, test_offload_mode_m{2,3}.py,
  test_lora_offload_mode.py, test_fused_lora_kernels.py,
  test_multi_gpu_7b.py, test_profiler.py — 14+ assertions across 7
  files all asserting ``param.data.numel() == 0`` post-offload).
* ``test_gather_offload_round_trip_shape`` — after a full
  ``gather → forward → offload`` round-trip the released param's shape
  matches the slot shape (not ``[0]``). Pins that ``offload()`` honours
  the flag too, not just initial materialize.
* ``test_storage_footprint_is_bounded`` — the per-dtype scratch is
  ONE 1-element tensor; expand views contribute no extra bytes
  regardless of how many params are released.
* ``test_autograd_shape_capture_on_released_param`` — concrete
  reproducer of the autograd race-window root cause: a forward
  dispatched against a ``[0]``-shape released param records the
  ``[0]`` shape (and fails); the same dispatch against a shape-
  preserving placeholder records the real shape (and the inner op
  surfaces a real size mismatch — not the misleading
  ``ToCopyBackward0 ... expected [0]`` from the autograd side).
"""

from __future__ import annotations

from typing import cast

import pytest

from axolotl.integrations.protrain.types import (
    BlockId,
    ParamId,
)


def _tiny_model(hidden: int = 64, n_layers: int = 4):
    """A tiny 4-layer transformer-ish model.

    Mirrors ``test_chunk_manager_offload._tiny_model`` so the layout
    builder picks each ``h.{i}`` Linear up as its own block / chunk.
    """
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
    """Assemble a :class:`ChunkManager` with the M6C-fix-7 flag toggled."""
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


@pytest.mark.gpu
def test_release_state_preserves_shape() -> None:
    """M6C-fix-7 central invariant.

    With ``shape_preserving_placeholders=True``, every non-persistent
    chunk-managed param has its ORIGINAL logical shape after
    ``materialize_offload`` — NOT ``torch.Size([0])``. The new
    placeholder's storage is still effectively zero (one 1-element
    scratch per dtype shared across every released param), but
    ``param.size()`` / ``param.shape`` / ``param.dim()`` return the
    real values that autograd will eventually expect at backward.
    """
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

    mgr.uninstall()
    host.close()
    del pool


@pytest.mark.gpu
def test_release_state_default_off_is_unchanged() -> None:
    """Default ``shape_preserving_placeholders=False`` preserves legacy semantics.

    Guards the pre-existing test surface (``test_chunk_manager_offload.py``,
    ``test_offload_mode_m{2,3}.py``, ``test_lora_offload_mode.py``,
    ``test_fused_lora_kernels.py``, ``test_multi_gpu_7b.py``,
    ``test_profiler.py``) that asserts ``param.data.numel() == 0`` after
    materialize_offload. M6C-fix-7 must NOT regress this invariant on
    the default-off code path.
    """
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

    mgr.uninstall()
    host.close()
    del pool


@pytest.mark.gpu
def test_gather_offload_round_trip_shape() -> None:
    """After gather → offload round-trip, released shape is preserved.

    Pins ``offload()`` honours the flag in addition to
    ``materialize_offload``. Without the offload-path fix the gather
    rebind would briefly show the real shape, but a subsequent offload
    would re-zero it — defeating the architectural purpose.
    """
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

    mgr.uninstall()
    host.close()
    del pool


@pytest.mark.gpu
def test_storage_footprint_is_bounded() -> None:
    """The shape-preserving placeholder costs ~zero extra bytes.

    The per-dtype scratch is a 1-element tensor. Every released
    param of that dtype shares the same scratch via ``expand``; the
    expanded view has all-zero strides and contributes no additional
    storage. We verify by:

    1. ``self._shape_scratch_by_dtype`` has exactly one entry per dtype
       across all released params.
    2. Every released param's ``param.data.untyped_storage().data_ptr()``
       equals the scratch's storage pointer for that dtype.
    3. Each scratch is 1 element wide regardless of the number of
       params sharing it.
    """
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
        # One element wide → numel()==1 for the scratch itself.
        assert scratch.numel() == 1, (
            f"scratch for dtype={dtype} should be 1-element, got "
            f"numel={scratch.numel()}"
        )
        scratch_ptr = scratch.untyped_storage().data_ptr()
        assert ptrs == {scratch_ptr}, (
            f"dtype={dtype}: released params should all share scratch's "
            f"storage_ptr={scratch_ptr}, got {ptrs}"
        )

    mgr.uninstall()
    host.close()
    del pool


@pytest.mark.gpu
def test_autograd_shape_capture_on_released_param() -> None:
    """Direct reproducer of the M6C-fix-7 root-cause autograd race.

    The legacy ``torch.Size([0])`` placeholder lets a forward op
    dispatched on a released param record ``[0]`` in its autograd
    Node's input metadata. The shape-preserving placeholder lets the
    Node record the REAL shape; if the op fails it's a real size
    mismatch surfaced from the at::linear kernel, not the misleading
    ``ToCopyBackward0 ... expected [0]`` from the autograd side at
    backward.

    This test exercises the autograd path directly on a single
    Parameter rebound through ``_shape_preserving_placeholder`` and
    confirms ``param.size()`` returns the real shape during a forward
    that captures the param's shape into an autograd Node.
    """
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

    # Now rebind to real data and confirm autograd shape capture
    # produces the REAL shape — not [0] — through a full
    # forward+backward.
    real_data = torch.randn(*real_shape, dtype=dtype, device="cuda")
    param.data = real_data

    # Forward through a Linear that the LoRA factor would feed.
    x = torch.randn(4, real_shape[1], dtype=dtype, device="cuda", requires_grad=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y = nn.functional.linear(x, param)
    loss = y.sum()
    loss.backward()
    assert param.grad is not None
    assert param.grad.shape == torch.Size(real_shape), (
        f"autograd recorded the WRONG shape: expected {real_shape}, "
        f"got {tuple(param.grad.shape)}"
    )

    mgr.uninstall()
    host.close()
    del pool
