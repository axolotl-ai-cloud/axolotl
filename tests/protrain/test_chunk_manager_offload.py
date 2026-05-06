"""Tests for the M4.5 chunk-manager offload primitives.

Covers :meth:`ChunkManager.materialize_offload` and the per-param
post-accumulate-grad hooks — the two runtime gaps closed in M4.5. Every
test here runs on GPU (``@pytest.mark.gpu``); there's no meaningful CPU
equivalent because the offload semantics are defined in terms of
``torch.cuda.memory_allocated`` dropping.
"""

from __future__ import annotations

from typing import cast

import pytest

from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    ParamId,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model(hidden: int = 64, n_layers: int = 4):
    """A tiny 4-layer "transformer-ish" model.

    Each layer is one Linear — enough to give the layout builder N_block=4
    and 4 separable param groups. We use nn.ModuleList so the block
    discovery logic in layout.py picks it up as the transformer stack.
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
    """Build a ChunkLayout where each ``h.{i}`` linear is its own chunk."""
    from axolotl.integrations.protrain.chunk.layout import build_layout

    # Block spans: each h.i is a block. embed and head are unaffiliated.
    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        if name.startswith("h."):
            idx = int(name.split(".")[1])
            block_spans.setdefault(cast(BlockId, idx), []).append(cast(ParamId, name))

    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    return build_layout(model, exec_order, S_chunk, block_spans)


def _build_chunk_manager(
    model, n_persist: int, S_chunk: int, n_buffer: int | None = None
):
    """Assemble a :class:`ChunkManager` from scratch for offload tests."""
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
    )
    return mgr, layout, pool, host


# ---------------------------------------------------------------------------
# Test 1: materialize_offload releases GPU memory
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_materialize_offload_frees_gpu_memory() -> None:
    """Non-persistent chunks' param bytes should leave the GPU after offload."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    # Tiny 4-layer model, one chunk per layer when S_chunk is sized so
    # each layer exactly fills a chunk. hidden=64, fp32 -> 64*64*4 = 16 KB
    # per layer. Set S_chunk at 32 KB so each block lands in its own chunk.
    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")

    # Per-layer weight bytes: 64 * 64 * 4 = 16 KB. Pick S_chunk above that
    # per-param size, but below two-params-worth so each block gets its
    # own chunk.
    per_layer_bytes = hidden * hidden * 4
    S_chunk = per_layer_bytes + 4096  # 16 KB + 4 KB headroom

    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)
    # Expect N_chunk >= n_layers + 1 (+1 for embed / head grouping).
    n_non_persist = layout.N_chunk - 1
    assert n_non_persist >= 2, (
        f"test setup: expected >=2 non-persistent chunks, got {n_non_persist} "
        f"(N_chunk={layout.N_chunk})"
    )

    # Record baseline GPU memory before offload.
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    freed = mgr.materialize_offload()

    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()

    # Expect at least (n_non_persist) * per_layer_bytes to be freed —
    # the non-persistent chunks' params are now on pinned CPU memory.
    # We tolerate some slack because embed / head may land in the
    # persistent chunk and not count toward the saved bytes.
    expected_min_freed = (n_non_persist - 1) * per_layer_bytes
    delta = before - after
    assert delta >= expected_min_freed, (
        f"expected >= {expected_min_freed} bytes freed, got {delta} "
        f"(before={before}, after={after}, reported_freed={freed})"
    )
    assert freed >= expected_min_freed, (
        f"materialize_offload reported freed={freed}, expected >= {expected_min_freed}"
    )

    # Cleanup.
    mgr.uninstall()
    host.close()
    # Silence unused-var warnings — pool is referenced by mgr.
    del pool


# ---------------------------------------------------------------------------
# Test 2: gather / offload rebinds param.data correctly
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gather_rebinds_param_data() -> None:
    """After gather() the param.data is a non-empty GPU view; offload() empties it."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)
    mgr.materialize_offload()

    # Pick any non-persistent chunk id and confirm its params are empty.
    non_persist = sorted(mgr._non_persistent_ids)
    assert non_persist, "need at least one non-persistent chunk for this test"
    cid = non_persist[0]
    param_ids = layout.chunks[int(cid)]

    # Before gather: every non-persistent param has an empty .data tensor.
    for pid in param_ids:
        param = dict(model.named_parameters())[str(pid)]
        assert param.data.numel() == 0, (
            f"param {pid} not offloaded: .data.numel()={param.data.numel()}"
        )

    # Gather and check the params are now GPU-resident with the right shape.
    mgr.gather(cid)
    for pid in param_ids:
        param = dict(model.named_parameters())[str(pid)]
        assert param.data.numel() > 0, (
            f"param {pid} still empty after gather: {param.data.shape}"
        )
        assert param.data.device.type == "cuda", (
            f"param {pid} not on cuda after gather: {param.data.device}"
        )
        # Shape must match the original.
        assert tuple(param.data.shape) == (hidden, hidden), (
            f"param {pid} has wrong shape after gather: {param.data.shape}"
        )

    # Offload again — params should return to the empty placeholder.
    mgr.offload(cid)
    for pid in param_ids:
        param = dict(model.named_parameters())[str(pid)]
        assert param.data.numel() == 0, (
            f"param {pid} not emptied after offload: .data.numel()={param.data.numel()}"
        )

    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 2b: materialize_offload under mixed-dtype chunks (BUG 2 regression)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_materialize_offload_mixed_dtype() -> None:
    """Chunks holding a mix of fp16 + fp32 params must not hit ``view`` alignment.

    Before the fix (BUG 2), a chunk containing fp16 Linear weights
    followed by fp32 LayerNorm scales tripped
    ``RuntimeError: offset is not aligned``: the per-param byte offset
    landed on an odd multiple of 2 after the first fp16 param, and
    ``byte_view.view(torch.float32)`` rejected the unaligned view.

    The fix pads each slot's starting offset up to a multiple of the
    param's ``element_size``. This test builds a mixed-dtype module,
    forces everything into a single non-persistent chunk, and verifies
    materialize + gather both succeed and that ``param.data.dtype`` is
    preserved across the round trip.
    """
    pytest.importorskip("torch")
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    class MixedDtype(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # fp16 Linear + fp32 LayerNorm — the exact pattern Llama
            # emits inside each transformer block when attention
            # weights are fp16 but RMSNorm scales stay fp32. Put them
            # inside a ModuleList so layout.build_layout picks them up
            # as a single "block".
            attn = nn.Linear(32, 32, bias=False).half()
            # An fp32 tensor deliberately ordered AFTER the fp16 one
            # so the running byte offset lands at an odd 2-byte
            # boundary (32*32*2=2048 bytes — actually aligned, but
            # add an odd number of fp16 bytes to force misalignment).
            extra_fp16 = nn.Linear(1, 32, bias=False).half()  # 64 bytes, /=2
            norm = nn.LayerNorm(32).float()  # fp32 weight+bias
            layer = nn.Module()
            layer.attn = attn  # type: ignore[attr-defined]
            layer.extra = extra_fp16  # type: ignore[attr-defined]
            layer.norm = norm  # type: ignore[attr-defined]

            def fwd(x: torch.Tensor) -> torch.Tensor:
                y = layer.attn(x.half())
                y = layer.norm(y.float())
                return y

            layer.forward = fwd  # type: ignore[assignment]
            self.h = nn.ModuleList([layer])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.h[0](x)

    torch.manual_seed(0)
    model = MixedDtype().to("cuda")

    # Large enough S_chunk so the whole ModuleList lands in one chunk.
    S_chunk = 1 << 16  # 64 KB — fits everything
    mgr, layout, pool, host = _build_chunk_manager(
        model, n_persist=0, S_chunk=S_chunk, n_buffer=2
    )

    # Sanity: before the fix, this raised RuntimeError inside
    # ``byte_view.view(torch.float32)``.
    freed = mgr.materialize_offload()
    assert freed > 0, "expected some bytes freed from mixed-dtype chunk"

    # After offload, each param.data should be the empty GPU placeholder
    # with the ORIGINAL dtype preserved.
    expected_dtypes = {
        "h.0.attn.weight": torch.float16,
        "h.0.extra.weight": torch.float16,
        "h.0.norm.weight": torch.float32,
        "h.0.norm.bias": torch.float32,
    }
    for name, param in model.named_parameters():
        assert param.data.dtype == expected_dtypes[name], (
            f"{name} dtype {param.data.dtype} != expected "
            f"{expected_dtypes[name]} after offload"
        )
        assert param.data.numel() == 0, (
            f"{name} still has non-empty .data after offload: {param.data.shape}"
        )

    # Gather every non-persistent chunk and verify dtype+shape survive
    # the round trip without alignment errors.
    for cid_int in sorted(mgr._non_persistent_ids):
        cid = cast(ChunkId, cid_int)
        mgr.gather(cid)

    for name, param in model.named_parameters():
        assert param.data.dtype == expected_dtypes[name], (
            f"{name} dtype changed after gather: {param.data.dtype}"
        )
        assert param.data.device.type == "cuda", (
            f"{name} landed on {param.data.device} after gather"
        )
        assert param.data.numel() > 0, f"{name} still empty after gather"

    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 2d: materialize_offload uses PinnedHostMemory with precise sizing (App B.2)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_materialize_offload_uses_precise_pinned_pool() -> None:
    """The unified pinned pools' total bytes equal the SUM of per-chunk
    aligned bytes — no power-of-2 round-up.

    Paper App B.2 calls out that PyTorch's default ``CUDAHostAllocator``
    rounds pinned allocations up to the next power of two. ProTrain's
    custom :class:`PinnedHostMemory` allocates the exact byte count via
    ``cudaHostAlloc``. ``materialize_offload`` uses ONE
    ``PinnedHostMemory`` for the param shadow region and ONE for the
    grad shadow region, sized to the precise sum of per-chunk aligned
    bytes (params: ``chunk_bytes`` per chunk; grads: per-trainable-param
    or per-region shard bytes).

    This test:

    1. Constructs a mixed-dtype chunk (fp16 + fp32) so the BUG 2
       intra-chunk alignment fix is exercised under the new layout.
    2. Calls ``materialize_offload``.
    3. Asserts ``self._cpu_param_pool`` and ``self._cpu_grad_pool`` are
       :class:`PinnedHostMemory` instances.
    4. Independently recomputes the expected total aligned bytes for
       params and grads, and asserts each pool's
       ``total_bytes`` equals the sum of per-chunk aligned bytes plus
       the inter-chunk 16-byte alignment padding (and crucially is NOT
       a power of two for non-trivial sizes).
    5. Asserts ``is_precise_size`` is True so we know the ctypes
       ``cudaHostAlloc`` path engaged (and not the
       ``torch.empty(pin_memory=True)`` fallback that would re-introduce
       the round-up the test is meant to detect).
    """
    pytest.importorskip("torch")
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    torch.cuda.empty_cache()

    # Mixed-dtype model: fp16 attention weight + fp32 norm scales,
    # repeated across multiple "blocks" so several non-persistent
    # chunks are populated. Hand-picked sizes so the per-chunk total
    # is NOT a power of two — the assertion below would be vacuous if
    # the input bytes happened to be a perfect power of two.
    class MixedDtype(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: list[nn.Module] = []
            for _ in range(3):
                attn = nn.Linear(48, 48, bias=False).half()  # 48*48*2 = 4608 B
                # Trailing fp32 norm forces the BUG 2 alignment pad
                # into the pool's per-chunk byte plan.
                norm = nn.LayerNorm(48).float()  # weight + bias = 48*4*2 = 384 B
                layer = nn.Module()
                layer.attn = attn  # type: ignore[attr-defined]
                layer.norm = norm  # type: ignore[attr-defined]

                def fwd(x: torch.Tensor, *, _layer=layer) -> torch.Tensor:
                    y = _layer.attn(x.half())
                    y = _layer.norm(y.float())
                    return y

                layer.forward = fwd  # type: ignore[assignment]
                blocks.append(layer)
            self.h = nn.ModuleList(blocks)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.h:
                x = layer(x)
            return x

    torch.manual_seed(0)
    model = MixedDtype().to("cuda")

    # S_chunk just big enough to hold one block (4608 + small fp32 pad
    # + 384 = ~5000 B). 8 KB is comfortably above that.
    S_chunk = 1 << 13
    mgr, layout, pool, host = _build_chunk_manager(
        model, n_persist=0, S_chunk=S_chunk, n_buffer=2
    )

    # Independent ground truth for the per-chunk aligned-byte plan,
    # mirroring materialize_offload's BUG 2 alignment pass.
    def _per_chunk_aligned_bytes(param_ids: list) -> int:
        offset = 0
        for pid in param_ids:
            p = dict(model.named_parameters()).get(str(pid))
            if p is None:
                continue
            esz = int(p.element_size())
            nbytes = int(p.numel()) * esz
            if nbytes == 0:
                continue
            offset = ((offset + esz - 1) // esz) * esz
            offset += nbytes
        return offset

    def _align_up(n: int, a: int) -> int:
        return ((n + a - 1) // a) * a

    _INTER_CHUNK_ALIGN = 16
    expected_param_bytes = 0
    expected_grad_bytes = 0
    for cid_int in sorted(mgr._non_persistent_ids):
        chunk_param_ids = layout.chunks[cid_int]
        if not chunk_param_ids:
            continue
        chunk_bytes = _per_chunk_aligned_bytes(chunk_param_ids)
        if chunk_bytes == 0:
            continue
        expected_param_bytes = (
            _align_up(expected_param_bytes, _INTER_CHUNK_ALIGN) + chunk_bytes
        )
        # Replicated path: every trainable param contributes its full
        # nbytes (no shard split). Sum across the chunk's params.
        chunk_grad_bytes = 0
        for pid in chunk_param_ids:
            p = dict(model.named_parameters()).get(str(pid))
            if p is None or not p.requires_grad:
                continue
            chunk_grad_bytes += int(p.numel()) * int(p.element_size())
        expected_grad_bytes = (
            _align_up(expected_grad_bytes, _INTER_CHUNK_ALIGN) + chunk_grad_bytes
        )

    # Sanity check: the byte total must NOT be a power of two — if it
    # were, the ``not power-of-two`` assertion below would be vacuous.
    assert expected_param_bytes & (expected_param_bytes - 1) != 0, (
        f"expected_param_bytes={expected_param_bytes} happens to be a power "
        "of two — pick a different model shape so the precise-size assertion "
        "is meaningful"
    )

    mgr.materialize_offload()

    # ---- Assertion 1: pools are PinnedHostMemory instances ------------
    assert isinstance(mgr._cpu_param_pool, PinnedHostMemory), (
        f"_cpu_param_pool should be PinnedHostMemory, got "
        f"{type(mgr._cpu_param_pool).__name__}"
    )
    assert isinstance(mgr._cpu_grad_pool, PinnedHostMemory), (
        f"_cpu_grad_pool should be PinnedHostMemory, got "
        f"{type(mgr._cpu_grad_pool).__name__}"
    )

    # ---- Assertion 2: pool total_bytes is the exact sum, no round-up
    assert mgr._cpu_param_pool.total_bytes == expected_param_bytes, (
        f"param pool size {mgr._cpu_param_pool.total_bytes} != expected "
        f"{expected_param_bytes} (sum of per-chunk aligned bytes); "
        "App B.2 round-up regressed?"
    )
    assert mgr._cpu_grad_pool.total_bytes == expected_grad_bytes, (
        f"grad pool size {mgr._cpu_grad_pool.total_bytes} != expected "
        f"{expected_grad_bytes} (sum of per-chunk trainable-param aligned "
        "bytes); App B.2 round-up regressed?"
    )

    # ---- Assertion 3: ctypes cudaHostAlloc path engaged ---------------
    # If this assertion fails, the libcudart fallback to
    # ``torch.empty(pin_memory=True)`` is in effect — that path goes
    # through CUDAHostAllocator, which IS the round-up source the
    # paper rejects. The test would still pass total_bytes (the
    # fallback also uses the requested size as the request), but the
    # paper-fidelity claim is broken.
    assert mgr._cpu_param_pool.is_precise_size, (
        "PinnedHostMemory fell back to torch.empty(pin_memory=True); "
        "ctypes cudaHostAlloc path failed — App B.2 fidelity not honored"
    )
    assert mgr._cpu_grad_pool.is_precise_size, (
        "grad pool fell back to torch.empty(pin_memory=True)"
    )

    # ---- Assertion 4: per-slot views still pass BUG 2 alignment -------
    # Mirrored from test_materialize_offload_mixed_dtype: every slot's
    # cpu_data and cpu_grad must round-trip through the dtype view
    # without raising "offset is not aligned".
    for cid_int in sorted(mgr._non_persistent_ids):
        slots = mgr._cpu_slots.get(ChunkId(cid_int), [])
        for slot in slots:
            if slot.cpu_data is not None:
                assert slot.cpu_data.dtype == slot.dtype
                assert tuple(slot.cpu_data.shape) == tuple(slot.shape)
            if slot.cpu_grad is not None:
                assert slot.cpu_grad.dtype == slot.dtype
                assert tuple(slot.cpu_grad.shape) == tuple(slot.shape)

    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 2c: param.data returns to empty-GPU placeholder between iterations (BUG 4)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_param_data_empty_between_iters() -> None:
    """After CPU Adam step, ``param.data`` must be a zero-element GPU tensor.

    BUG 4: before the fix, ``_ensure_cpu_grads_attached`` repointed
    ``param.data`` at the CPU shard for the CPU Adam step and nothing
    repointed it back. Between end-of-iter and start-of-next-iter,
    ``param.data`` was a CPU tensor — any intermediate code reading
    ``.data`` (``clip_grad_norm_``, Trainer metric hooks, checkpoint
    save) saw CPU where GPU was expected.

    The fix registers a ``post_step`` callback on ``step_async`` that
    repoints ``.data`` back to ``_empty_placeholder(dtype)`` after the
    CPU Adam step resolves. This test runs a full fwd+bwd+step cycle
    and asserts post-step that every non-persistent param has
    ``param.data.numel() == 0`` AND ``param.data.device.type == "cuda"``.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")
    # DeepSpeedCPUAdam compiles a CUDA extension lazily — import
    # success doesn't imply it can build. Probe cheaply so the test
    # gracefully skips in envs where nvcc↔torch CUDA versions
    # disagree (the runtime path handles the missing adapter; this
    # test just isolates BUG 4's repointing semantics).
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        _probe = DeepSpeedCPUAdam([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
        del _probe
    except Exception:  # noqa: BLE001
        pytest.skip("DeepSpeedCPUAdam unavailable — BUG 4 path requires CPU optim")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    S_chunk = hidden * hidden * 4 + 4096

    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    layout_probe = _build_layout_for(model, S_chunk)
    n_non_persist = layout_probe.N_chunk - 1
    mgr, layout, pool, host = _build_chunk_manager(
        model, n_persist=1, S_chunk=S_chunk, n_buffer=n_non_persist
    )
    mgr.materialize_offload()

    # Build a CPU Adam adapter so the BUG 4 repoint callback fires.
    from axolotl.integrations.protrain.chunk.optim import CpuFusedAdamAdapter

    cpu_params_per_chunk: dict = {}
    for cid_int in sorted(mgr._non_persistent_ids):
        params = [
            dict(model.named_parameters())[str(pid)]
            for pid in layout.chunks[int(cid_int)]
            if str(pid) in dict(model.named_parameters())
        ]
        if params:
            cpu_params_per_chunk[cid_int] = params

    cpu_optim = CpuFusedAdamAdapter(params_per_chunk=cpu_params_per_chunk, lr=1e-4)
    mgr.cpu_optim = cpu_optim

    # Drive one fwd+bwd+step cycle. Gather everything manually (no
    # scheduler in this bare test).
    for cid_int in range(layout.N_chunk):
        mgr.gather(cast(ChunkId, cid_int))

    x = torch.randn(2, hidden, device="cuda")
    y = model(x)
    loss = y.sum()
    loss.backward()

    # The per-param hooks fired step_async on the CPU optim. Block
    # until every future has resolved — the post_step callback runs
    # inside that wait, so after this line param.data MUST be the
    # empty GPU placeholder.
    mgr.wait_cpu_optim_all()

    for cid_int in sorted(mgr._non_persistent_ids):
        cid = cast(ChunkId, cid_int)
        slots = mgr._cpu_slots.get(cid, [])
        for slot in slots:
            param = dict(model.named_parameters())[str(slot.param_id)]
            if not param.requires_grad:
                continue
            assert param.data.numel() == 0, (
                f"non-persistent param {slot.param_id}.data non-empty "
                f"between iters: shape={param.data.shape} "
                f"device={param.data.device}"
            )
            assert param.data.device.type == "cuda", (
                f"non-persistent param {slot.param_id}.data on "
                f"{param.data.device} between iters (BUG 4 regression)"
            )

    cpu_optim.shutdown()
    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 3: per-param grad hooks fire and drain to CPU shards
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_grad_offload_hook_fires() -> None:
    """After backward, the CPU grad shards hold the correct grad values.

    We compare against a reference run of the same model WITHOUT ProTrain
    wrapping — both runs should produce identical grads on identical
    inputs, with the ProTrain run's grads landing on the CPU shards
    instead of ``param.grad``.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    S_chunk = hidden * hidden * 4 + 4096

    # ---- Reference run: plain PyTorch -----------------------------------
    torch.manual_seed(7)
    ref_model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    x = torch.randn(2, hidden, device="cuda")
    y_ref = ref_model(x)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    ref_grads = {
        name: p.grad.detach().clone().cpu() for name, p in ref_model.named_parameters()
    }

    # ---- ProTrain-wrapped run ------------------------------------------
    torch.manual_seed(7)  # same init → same params
    model = _tiny_model(hidden=hidden, n_layers=n_layers).to("cuda")
    # n_buffer large enough to gather every non-persistent chunk at once —
    # the scheduler normally rotates through a smaller pool, but this
    # test runs without the scheduler and needs every param resident
    # simultaneously for the forward pass to succeed.
    layout_probe = _build_layout_for(model, S_chunk)
    n_non_persist = layout_probe.N_chunk - 1
    mgr, layout, pool, host = _build_chunk_manager(
        model, n_persist=1, S_chunk=S_chunk, n_buffer=n_non_persist
    )
    # The grad-offload hook routes to ``cm.cpu_optim.step_async`` once a
    # chunk's last param drains; ChunkManager raises RuntimeError when
    # ``cpu_optim is None`` on that path (CodeRabbit R2-05 — silent skip
    # would mask stale offloaded weights). This test only validates the
    # grad-offload portion of the hook, not the optimizer step, so a
    # no-op stub satisfies the contract without depending on
    # DeepSpeedCPUAdam being available on the rig.

    class _NoOpCpuOptim:
        """Minimal CpuFusedAdamAdapter surface used by the chunk-step path."""

        def step_async(self, chunk_id, *, d2h_event=None, post_step=None):  # noqa: ARG002
            return None

        def wait_all(self) -> None:
            return None

    mgr.cpu_optim = _NoOpCpuOptim()  # type: ignore[assignment]
    mgr.materialize_offload()

    # Gather all non-persistent chunks so the forward has GPU-resident
    # params. Without the scheduler pumping this (it's not installed in
    # this bare-metal test), we drive it manually.
    for cid_int in range(layout.N_chunk):
        mgr.gather(cast(ChunkId, cid_int))

    # Forward / backward with the SAME input as the reference.
    y = model(x)
    loss = y.sum()
    loss.backward()

    # The per-param hook should have offloaded every non-persistent
    # param's .grad to the pinned-CPU shard. After the last param in a
    # chunk fires its hook, :meth:`_ensure_cpu_grads_attached` repoints
    # ``param.grad`` at the CPU shard so the optimizer adapter can consume
    # it — so ``param.grad`` is either None (draining in progress) or a
    # CPU tensor (fully drained), but NEVER a GPU tensor.
    for cid_int in sorted(mgr._non_persistent_ids):
        cid = cast(ChunkId, cid_int)
        slots = mgr._cpu_slots.get(cid, [])
        for slot in slots:
            param = dict(model.named_parameters())[str(slot.param_id)]
            if not param.requires_grad:
                continue
            # Hook should have drained the GPU grad. ``param.grad`` is
            # either None or a CPU tensor; it must NOT be a GPU tensor.
            if param.grad is not None:
                assert param.grad.device.type == "cpu", (
                    f"non-persistent param {slot.param_id} still has a GPU "
                    f".grad of shape {param.grad.shape}; hook did not "
                    "drain to CPU"
                )
            # The CPU grad shard must match the reference grad.
            ref = ref_grads[str(slot.param_id)]
            got = slot.cpu_grad
            assert got is not None, (
                f"slot {slot.param_id}: cpu_grad shard was not allocated"
            )
            assert torch.allclose(ref, got.cpu().float(), atol=1e-4, rtol=1e-4), (
                f"CPU grad for {slot.param_id} diverged from reference: "
                f"max abs diff = {(ref - got.cpu().float()).abs().max().item()}"
            )

    # Persistent-chunk params keep their GPU grads (not hook-drained).
    for cid_int in sorted(mgr._persistent_ids):
        cid = cast(ChunkId, cid_int)
        for pid in layout.chunks[int(cid)]:
            param = dict(model.named_parameters())[str(pid)]
            if not param.requires_grad:
                continue
            assert param.grad is not None, (
                f"persistent param {pid} unexpectedly had grad drained"
            )
            ref = ref_grads[str(pid)]
            assert torch.allclose(
                ref, param.grad.cpu().float(), atol=1e-4, rtol=1e-4
            ), f"persistent-chunk grad for {pid} diverged from reference"

    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# restore_to_gpu — inverse of materialize_offload (phase-2 profiler bootstrap)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_restore_to_gpu_round_trip_preserves_param_values() -> None:
    """materialize_offload → restore_to_gpu must leave every param byte-identical.

    The phase-2 profiler builds a bootstrap chunk-manager, runs a
    chunked fwd+bwd+step measurement loop, then needs to tear down and
    rebuild under a (potentially different) post-research config. The
    teardown lives in :meth:`ChunkManager.restore_to_gpu`. Round-trip
    correctness is the hard correctness invariant — without it the
    rebuilt manager would see corrupted weights.
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

    # Snapshot every parameter's value BEFORE we touch the manager. The
    # round-trip must reproduce these byte-for-byte.
    reference: dict[str, torch.Tensor] = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }

    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)

    freed = mgr.materialize_offload()
    assert freed > 0, "test setup: expected non-persistent bytes to be freed"

    any_empty = any(p.data.numel() == 0 for name, p in model.named_parameters())
    assert any_empty, (
        "test setup invariant: at least one param should be offloaded to "
        "an empty placeholder before restore"
    )

    # Gather persistent chunks so their pool-buffer view becomes the
    # source-of-truth bytes that restore_to_gpu must extract.
    for cid_int in sorted(mgr._persistent_ids):
        mgr.gather(cast(ChunkId, cid_int))

    moved = mgr.restore_to_gpu()
    assert moved > 0, "restore_to_gpu reported 0 bytes moved — should be > 0"

    for name, p in model.named_parameters():
        assert p.data.numel() == reference[name].numel(), (
            f"param {name}: numel changed across restore "
            f"({reference[name].numel()} -> {p.data.numel()})"
        )
        assert p.data.device.type == "cuda", (
            f"param {name} not on cuda after restore: {p.data.device}"
        )
        assert torch.equal(p.data, reference[name]), (
            f"param {name} bytes diverged across "
            "materialize_offload -> restore_to_gpu round-trip"
        )

    # Internal state cleared so a new manager can rebuild from scratch.
    assert not mgr._cpu_slots, "restore_to_gpu must clear _cpu_slots"
    assert not mgr._persistent_buffers, "restore_to_gpu must clear _persistent_buffers"
    assert not mgr._grad_hook_handles, (
        "restore_to_gpu must remove all grad hook handles"
    )

    host.close()
    del pool


@pytest.mark.gpu
def test_restore_to_gpu_idempotent_on_unmaterialized_manager() -> None:
    """A manager that never offloaded is a no-op restore — no exception, returns 0."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    model = _tiny_model(hidden=hidden, n_layers=4).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    mgr, _layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)

    assert mgr.restore_to_gpu() == 0
    assert mgr.restore_to_gpu() == 0  # twice in a row

    host.close()
    del pool


@pytest.mark.gpu
def test_restore_to_gpu_enables_clean_rebuild_under_new_config() -> None:
    """Restore lets a fresh ChunkManager be built on the same model with a new n_persist.

    This is the actual phase-2 use case: bootstrap manager -> measure ->
    restore -> build a second manager with a different config. The
    second materialize_offload must run successfully (i.e. not see the
    first manager's leftover state on the model parameters).
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

    reference: dict[str, torch.Tensor] = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }

    # Bootstrap: n_persist=1.
    mgr1, _layout1, pool1, host1 = _build_chunk_manager(
        model, n_persist=1, S_chunk=S_chunk
    )
    mgr1.materialize_offload()
    for cid_int in sorted(mgr1._persistent_ids):
        mgr1.gather(cast(ChunkId, cid_int))
    mgr1.restore_to_gpu()
    host1.close()
    del mgr1, pool1

    # Post-research: a different n_persist on the same model.
    mgr2, _layout2, pool2, host2 = _build_chunk_manager(
        model, n_persist=2, S_chunk=S_chunk
    )
    freed2 = mgr2.materialize_offload()
    assert freed2 > 0, (
        "second materialize_offload reported 0 freed — restore left "
        "stale state on the model that prevented re-offload"
    )

    # Gather everything so we can compare against the reference.
    for cid_int in sorted(mgr2._persistent_ids):
        mgr2.gather(cast(ChunkId, cid_int))
    for cid_int in sorted(mgr2._non_persistent_ids):
        mgr2.gather(cast(ChunkId, cid_int))
    for name, p in model.named_parameters():
        assert torch.equal(p.data, reference[name]), (
            f"param {name} corrupted across two materialize/restore cycles"
        )

    mgr2.uninstall()
    host2.close()
    del pool2


# ---------------------------------------------------------------------------
# protrain_optimizer_wrapper partitioning — regression for non-contiguous
# _persistent_ids (the non-block-chunk pin produces e.g. {0..n-1, last} on
# Llama with an untied lm_head; a prefix ``cid < n_persist`` test would
# misroute that high-cid persistent chunk to the CPU adam path).
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_optimizer_partition_uses_persistent_id_set_not_prefix() -> None:
    """When _persistent_ids is non-contiguous, partitioning must follow the SET."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    from axolotl.integrations.protrain.api.optim_wrapper import (
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import WrappedModel

    torch.cuda.empty_cache()
    hidden = 64
    model = _tiny_model(hidden=hidden, n_layers=4).to("cuda")
    S_chunk = hidden * hidden * 4 + 4096

    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)
    # Force a non-contiguous persistent set: {0, last}. This is the
    # shape the wrapper's non-block-chunk pin produces when an untied
    # lm_head sits at the tail of N_chunk. The fix must route chunk
    # ``last`` into the GPU optimizer's param list (its params are
    # GPU-resident, never offloaded), and chunks 1..last-1 into the
    # CPU FusedAdam path (their params will be offloaded by
    # materialize_offload).
    last = layout.N_chunk - 1
    assert last >= 2, "test setup needs N_chunk >= 3 for a useful gap"
    mgr._persistent_ids = {cast(ChunkId, 0), cast(ChunkId, last)}
    mgr._non_persistent_ids = {
        cast(ChunkId, c) for c in range(layout.N_chunk) if c not in mgr._persistent_ids
    }

    # materialize_offload to set up the CPU shards for non-persistent
    # chunks — protrain_optimizer_wrapper consults
    # chunk_manager._chunk_shards / cpu_slots to derive the CPU adam
    # adapter's per-chunk param lists.
    mgr.materialize_offload()

    # Build a placeholder WrappedModel (only the fields the optim
    # wrapper reads matter).
    wrapped = WrappedModel(
        module=model,
        search_result=None,  # type: ignore[arg-type]
        chunk_manager=mgr,
        scheduler=None,
        _hook_handles=[],
    )

    # Patch CpuFusedAdamAdapter at the optim_wrapper module's lookup
    # site to capture the partitioning without requiring DeepSpeed's
    # CPU-Adam C++ extension (this rig may not have it compiled — see
    # the CUDA-version mismatch warning the wrapper emits). The
    # capture lets us inspect the EXACT keys the partition produced.
    from unittest.mock import patch

    captured_keys: dict = {}

    class _StubCpuAdam:
        def __init__(self, params_per_chunk, **_kwargs):
            captured_keys["keys"] = set(int(k) for k in params_per_chunk.keys())
            captured_keys["params_per_chunk"] = params_per_chunk

        def zero_grad(self, set_to_none: bool = True):
            pass

    with patch(
        "axolotl.integrations.protrain.api.optim_wrapper.CpuFusedAdamAdapter",
        _StubCpuAdam,
    ):
        _ = protrain_optimizer_wrapper(wrapped, lr=1e-3)

    assert "keys" in captured_keys, (
        "CpuFusedAdamAdapter constructor was never invoked — "
        "partitioning must have routed every chunk to the GPU optim "
        "(unexpected for a {0, last} persistent set)"
    )
    cpu_keys = captured_keys["keys"]
    expected_cpu_keys = set(int(c) for c in mgr._non_persistent_ids)
    assert cpu_keys == expected_cpu_keys, (
        f"CPU adam partitioning misroutes chunks: got cpu_keys="
        f"{sorted(cpu_keys)}, expected exactly the non-persistent set "
        f"{sorted(expected_cpu_keys)}. Persistent chunks at high cid "
        "(non-block-pinned tail like an untied lm_head) leak into the "
        "CPU adam partition under a prefix ``cid < n_persist`` test."
    )

    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Sharded restore_to_gpu (zero3_shard=True) — gloo 2-rank round-trip
# ---------------------------------------------------------------------------
#
# The sharded teardown path was added so the phase-2 profiler can rebuild
# the chunk-manager under a new config in a distributed run. Round-trip
# correctness here means: after materialize_offload partitions every
# chunk's bytes across ranks, restore_to_gpu reassembles them via
# per-region all_gather and rebinds param.data so every rank's model
# matches the pre-offload weights bit-for-bit. Mirrors the existing
# ``test_zero3_sharded_roundtrip_2rank`` pattern in
# ``test_chunk_manager_distributed.py`` (gloo + ``mp.spawn`` + CPU device
# pool — the byte-level operations are identical to the CUDA path).


def _worker_sharded_restore_round_trip(rank: int, world_size: int, tmpdir: str) -> None:
    """Child process body: sharded materialize_offload -> restore_to_gpu.

    Builds a small mixed-dtype model (fp16 Linear + fp32 LayerNorm) so
    the test exercises the multi-region branch of the sharded restore —
    a homogeneous-dtype chunk would only issue ONE all_gather and miss
    the per-region loop. After restore every param's bytes must equal
    the pre-offload snapshot.
    """
    import os as _os

    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import (
        PinnedHostMemory,
    )

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29551")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-restore",
        rank=rank,
        world_size=world_size,
    )

    try:
        # Same seed across ranks => identical fresh-init weights.
        torch.manual_seed(0)
        from torch import nn

        class _MixedLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(16, 16, bias=True).to(torch.float16)
                self.norm = nn.LayerNorm(16).to(torch.float32)

        layer = _MixedLayer()
        model = nn.Module()
        model.h = nn.ModuleList([layer])  # type: ignore[attr-defined]

        block_spans: dict = {}
        for name, _p in model.named_parameters():
            block_spans.setdefault(BlockId(0), []).append(ParamId(name))  # type: ignore[index]
        exec_order = [ParamId(n) for n, _ in model.named_parameters()]
        S_chunk = 1 << 14
        layout = build_layout(model, exec_order, S_chunk, block_spans)

        host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
        pool = BufferPool(
            n_buffer=1,
            S_chunk=layout.S_chunk,
            pinned_host=host,
            device=torch.device("cpu"),
        )

        # Snapshot every param BEFORE materialize_offload — restore must
        # reproduce these bytes exactly.
        pre_data = {
            str(name): p.detach().clone() for name, p in model.named_parameters()
        }

        mgr = ChunkManager(
            model=model,
            layout=layout,
            n_persist=0,
            buffer_pool=pool,
            cpu_optim=None,
            gpu_optim=None,
            device=torch.device("cpu"),
            world_size=world_size,
            rank=rank,
            zero3_shard=True,
        )

        try:
            mgr.materialize_offload()
        except RuntimeError as exc:
            if "gloo" in str(exc).lower():
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-unsupported: {exc}\n")
                return
            raise

        # Sharding must have actually engaged for the test to be
        # meaningful — a silent fall-back to replicated would route
        # restore through the non-sharded branch and leave the new
        # all_gather code uncovered.
        assert mgr.sharded_chunk_ids() == [ChunkId(0)], (
            f"rank {rank}: expected chunk 0 sharded, got {mgr.sharded_chunk_ids()}"
        )
        # Multi-region invariant: mixed-dtype chunk produces 2 regions.
        shard_state = mgr._chunk_shards[ChunkId(0)]
        assert len(shard_state.regions) == 2, (
            f"rank {rank}: expected 2 dtype regions (fp16 + fp32), "
            f"got {len(shard_state.regions)}"
        )

        # Every param's data should be an empty placeholder after
        # materialize_offload — confirms the test exercises the path
        # where restore_to_gpu has real work to do.
        any_empty = any(p.data.numel() == 0 for _n, p in model.named_parameters())
        assert any_empty, f"rank {rank}: post-offload param data should be empty"

        # The actual round-trip: sharded restore must reassemble every
        # chunk via all_gather and rebind param.data on every rank.
        try:
            moved = mgr.restore_to_gpu()
        except RuntimeError as exc:
            if "not implemented" in str(exc).lower() or "gloo" in str(exc).lower():
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-collective-unsupported: {exc}\n")
                return
            raise

        assert moved > 0, (
            f"rank {rank}: restore_to_gpu reported 0 bytes moved — "
            "should be > 0 with sharded chunks present"
        )

        # Bit-exact match against the pre-offload snapshot. fp16/fp32
        # tensors are checked with torch.equal because no arithmetic
        # ran between materialize and restore — only memcpy through
        # all_gather. Any mismatch indicates the byte layout flipped
        # somewhere in the per-region reassembly.
        for name, p in model.named_parameters():
            snap = pre_data[str(name)]
            assert p.data.shape == snap.shape, (
                f"rank {rank}: shape changed for {name}: {p.data.shape} vs {snap.shape}"
            )
            assert p.data.dtype == snap.dtype, (
                f"rank {rank}: dtype changed for {name}: {p.data.dtype} vs {snap.dtype}"
            )
            assert torch.equal(p.data, snap), (
                f"rank {rank}: param {name} bytes diverged across "
                "sharded materialize_offload -> restore_to_gpu round-trip"
            )

        # Internal-state cleanup is the same contract as the
        # non-sharded restore: every per-chunk dict must be empty
        # after teardown so a fresh manager can be built on the same
        # model.
        assert not mgr._cpu_slots, f"rank {rank}: restore_to_gpu must clear _cpu_slots"
        assert not mgr._chunk_shards, (
            f"rank {rank}: restore_to_gpu must clear _chunk_shards"
        )
        assert not mgr._grad_hook_handles, (
            f"rank {rank}: restore_to_gpu must remove grad hook handles"
        )

        host.close()
        del pool

    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.gpu  # paired with the rest of the distributed lane
def test_sharded_restore_to_gpu_round_trip_2rank(tmp_path) -> None:
    """2-rank gloo: sharded materialize_offload -> restore_to_gpu round-trip.

    Documents the full-distributed paper-fidelity invariant: after a
    sharded ``materialize_offload`` partitions every chunk across ranks
    and a subsequent ``restore_to_gpu`` reassembles them via per-region
    ``all_gather_into_tensor``, every param on every rank must hold the
    exact same bytes as before the round-trip. This is what the phase-2
    profiler needs to bootstrap-then-rebuild under a new config in a
    distributed run.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_sharded_restore_round_trip,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    # Downgrade to a skip if any rank hit an unsupported gloo collective
    # (older torch builds may not expose all_gather_into_tensor on CPU).
    skip_files = list(tmp_path.glob("rank*.skip"))
    if skip_files:
        reasons = [f.read_text().strip() for f in skip_files]
        pytest.skip(f"gloo does not support required collective(s): {reasons}")


def test_sharded_restore_to_gpu_requires_initialized_distributed() -> None:
    """Pre-flight: sharded restore must raise a clean error sans dist init.

    The sharded path issues ``all_gather_into_tensor`` per region —
    that requires a live process group. Calling restore on a sharded
    manager AFTER ``destroy_process_group`` (or before init) is a
    programmer error; the manager raises ``RuntimeError`` with a clear
    message instead of letting torch.distributed surface an opaque
    "default process group not initialized" later in the call stack.

    Exercised single-process by manually planting a ``_chunk_shards``
    entry on a manager that was constructed with
    ``zero3_shard=False`` then forced into the sharded branch — same
    code path the round-trip test takes through legitimate
    ``materialize_offload`` but without needing a live gloo cluster.
    """
    pytest.importorskip("torch")
    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        pytest.skip(
            "torch.distributed already initialized — cannot exercise "
            "the uninitialized-dist guard"
        )

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.manager import (
        ChunkManager,
        _ChunkShardState,
    )
    from axolotl.integrations.protrain.chunk.pinned_alloc import (
        PinnedHostMemory,
    )

    # Build a tiny single-chunk manager on CPU; we do NOT init dist.
    # Manager constructor forces ``zero3_shard=False`` when world_size
    # is 1, so we flip both flags by hand to drive restore_to_gpu
    # into its sharded branch.
    hidden = 8
    model = _tiny_model(hidden=hidden, n_layers=2)
    layout = _build_layout_for(model, S_chunk=hidden * hidden * 4 + 4096)

    host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
    pool = BufferPool(
        n_buffer=1,
        S_chunk=layout.S_chunk,
        pinned_host=host,
        device=torch.device("cpu"),
    )
    mgr = ChunkManager(
        model=model,
        layout=layout,
        n_persist=0,
        buffer_pool=pool,
        cpu_optim=None,
        gpu_optim=None,
        device=torch.device("cpu"),
    )

    # Force the sharded-restore branch by populating both
    # ``zero3_shard`` and ``_chunk_shards`` / ``_cpu_slots`` directly.
    # The chunk shard's regions list can be empty — the guard fires on
    # the dict membership before any per-region work happens.
    mgr.zero3_shard = True
    cid = cast(ChunkId, 0)
    mgr._chunk_shards[cid] = _ChunkShardState(regions=[], chunk_bytes=0, shard_bytes=0)
    # An empty cpu_slots entry keeps the non-sharded copy loop a no-op
    # while still satisfying the "_cpu_slots or _chunk_shards" trigger.
    mgr._cpu_slots[cid] = []

    with pytest.raises(RuntimeError, match="torch.distributed is not initialized"):
        mgr.restore_to_gpu()

    # Cleanup — restore_to_gpu raised so its own clear() never ran.
    mgr._chunk_shards.clear()
    mgr._cpu_slots.clear()
    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# snapshot_cpu_state / restore_cpu_state — phase-2 measurement rollback path
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_snapshot_cpu_state_restores_mutated_pinned_bytes() -> None:
    """``snapshot_cpu_state`` must capture, and ``restore_cpu_state`` must reinstate,
    the pinned-host bytes that back every non-persistent param.

    Phase-2 correctness regression: ``measure_chunked_steady`` snapshots
    model state via ``model.state_dict()`` before its timed loop and
    rolls it back via ``model.load_state_dict()`` afterward. But
    ``materialize_offload`` rebinds every non-persistent param's
    ``param.data`` to an empty placeholder — the real weights live in
    :attr:`ChunkManager._cpu_slots` (replicated path) — so
    ``state_dict()`` only sees empty tensors for those params and the
    rollback is a no-op. The timed loop's ``optimizer.step()`` then
    leaves the CPU shadows permanently mutated past the caller's
    pre-measurement state.

    This test reproduces the bug end-to-end: materialize a manager,
    snapshot via the new helper, mutate one CPU slot's bytes by +1.0
    (simulating an in-place Adam update), restore via the new helper,
    then call ``restore_to_gpu`` and verify every param's GPU bytes
    match the pre-snapshot reference. Without the snapshot/restore
    pair this assertion fails by exactly the +1.0 perturbation.
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

    # Reference: snapshot every parameter's original bytes BEFORE any
    # offload so we can compare against the post-restore state.
    reference: dict[str, torch.Tensor] = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }

    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)
    freed = mgr.materialize_offload()
    assert freed > 0, "test setup: expected non-persistent bytes to be freed"

    # Pick a non-persistent chunk to mutate. The chunk's pinned CPU
    # shadow lives in ``_cpu_slots[cid][i].cpu_data``.
    non_persist = sorted(mgr._non_persistent_ids)
    assert non_persist, "test setup: need at least one non-persistent chunk"
    target_cid = non_persist[0]
    slots = mgr._cpu_slots[target_cid]
    assert slots, f"test setup: chunk {target_cid} has no slots"
    mutated_slot = next((s for s in slots if s.cpu_data is not None), None)
    assert mutated_slot is not None, (
        "test setup: replicated chunk should have at least one slot with cpu_data"
    )

    # Snapshot via the new helper. This must clone the bytes
    # (independent storage) — otherwise the in-place mutation below
    # would silently advance the snapshot too and the restore would be
    # a no-op for the wrong reason.
    snap = mgr.snapshot_cpu_state()
    assert target_cid in snap, "snapshot must include every non-persistent chunk"

    # Mutate the live pinned tensor IN PLACE — exactly what
    # ``optimizer.step()`` does in the offload path
    # (``_ensure_cpu_grads_attached`` repoints param.data at the CPU
    # shadow and Adam writes through it).
    pre_mutation = mutated_slot.cpu_data.detach().clone()
    mutated_slot.cpu_data.add_(1.0)
    assert not torch.equal(mutated_slot.cpu_data, pre_mutation), (
        "test setup: in-place add_ should have mutated the pinned tensor"
    )

    # Restore via the new helper. The pinned tensor's storage is reused
    # — ``copy_`` writes back into the same buffer the snapshot sliced
    # out of — so any optimizer / grad bookkeeping aliased to it stays
    # consistent.
    mgr.restore_cpu_state(snap)
    assert torch.equal(mutated_slot.cpu_data, pre_mutation), (
        "restore_cpu_state failed to roll back the in-place mutation"
    )

    # End-to-end check: gather persistent chunks, restore_to_gpu, then
    # compare every param against the pre-offload reference. Without
    # snapshot/restore_cpu_state this assertion fails by exactly +1.0
    # for the mutated slot's params.
    for cid_int in sorted(mgr._persistent_ids):
        mgr.gather(cast(ChunkId, cid_int))
    moved = mgr.restore_to_gpu()
    assert moved > 0, "restore_to_gpu reported 0 bytes moved — should be > 0"
    for name, p in model.named_parameters():
        assert torch.equal(p.data, reference[name]), (
            f"param {name} bytes diverged across snapshot+mutate+restore round-trip "
            "— restore_cpu_state did not restore the pinned bytes"
        )

    host.close()
    del pool


@pytest.mark.gpu
def test_snapshot_cpu_state_independent_storage() -> None:
    """Snapshot tensors must have storage independent of the live pinned views.

    A naive snapshot that shares storage with the pinned shadows would
    be silently advanced by an ``optimizer.step()`` between snapshot
    and restore — defeating the rollback. Mutate the snapshot AFTER
    the call and confirm the live pinned tensor is unchanged.
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

    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)
    mgr.materialize_offload()

    non_persist = sorted(mgr._non_persistent_ids)
    target_cid = non_persist[0]
    slots = mgr._cpu_slots[target_cid]
    live_slot = next(s for s in slots if s.cpu_data is not None)
    pre_value = live_slot.cpu_data.detach().clone()

    snap = mgr.snapshot_cpu_state()
    snap_tensors = snap[target_cid]["slots"]
    assert snap_tensors is not None, (
        "replicated chunk should populate the 'slots' key in the snapshot"
    )
    snap_tensor = next(t for t in snap_tensors if t is not None)

    # Mutate the snapshot — must not touch the live pinned tensor.
    snap_tensor.add_(7.5)
    assert torch.equal(live_slot.cpu_data, pre_value), (
        "snapshot tensor shares storage with the live pinned shadow — "
        "snapshot must be deep-cloned"
    )

    # Cleanup.
    for cid_int in sorted(mgr._persistent_ids):
        mgr.gather(cast(ChunkId, cid_int))
    mgr.restore_to_gpu()
    host.close()
    del pool


def _worker_sharded_snapshot_restore_round_trip(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """2-rank gloo body: sharded snapshot -> mutate -> restore round-trip.

    Mirrors :func:`_worker_sharded_restore_round_trip`'s setup (mixed-
    dtype model so the sharded chunk produces ``len(regions) == 2``)
    but exercises the phase-2 ROLLBACK path explicitly: the bytes that
    survive the round-trip live in ``region.cpu_shard_bytes``, not in
    ``param.data`` (which is empty on offloaded chunks). Without
    ``snapshot_cpu_state`` covering the sharded branch, a phase-2
    measurement that updates the pinned shards would not roll back on
    Mode-C and the next iter would train from post-step weights.
    """
    import os as _os

    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import (
        PinnedHostMemory,
    )

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29553")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-snapshot-restore",
        rank=rank,
        world_size=world_size,
    )

    try:
        torch.manual_seed(0)
        from torch import nn

        class _MixedLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(16, 16, bias=True).to(torch.float16)
                self.norm = nn.LayerNorm(16).to(torch.float32)

        layer = _MixedLayer()
        model = nn.Module()
        model.h = nn.ModuleList([layer])  # type: ignore[attr-defined]

        block_spans: dict = {}
        for name, _p in model.named_parameters():
            block_spans.setdefault(BlockId(0), []).append(ParamId(name))  # type: ignore[index]
        exec_order = [ParamId(n) for n, _ in model.named_parameters()]
        S_chunk = 1 << 14
        layout = build_layout(model, exec_order, S_chunk, block_spans)

        host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
        pool = BufferPool(
            n_buffer=1,
            S_chunk=layout.S_chunk,
            pinned_host=host,
            device=torch.device("cpu"),
        )

        mgr = ChunkManager(
            model=model,
            layout=layout,
            n_persist=0,
            buffer_pool=pool,
            cpu_optim=None,
            gpu_optim=None,
            device=torch.device("cpu"),
            world_size=world_size,
            rank=rank,
            zero3_shard=True,
        )

        try:
            mgr.materialize_offload()
        except RuntimeError as exc:
            if "gloo" in str(exc).lower():
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-unsupported: {exc}\n")
                return
            raise

        # Sharding must have engaged for the snapshot path under test.
        assert mgr.sharded_chunk_ids() == [ChunkId(0)], (
            f"rank {rank}: expected chunk 0 sharded, got {mgr.sharded_chunk_ids()}"
        )
        shard_state = mgr._chunk_shards[ChunkId(0)]
        assert len(shard_state.regions) == 2, (
            f"rank {rank}: expected 2 dtype regions (fp16 + fp32), "
            f"got {len(shard_state.regions)}"
        )

        # Capture the live region pointers + a pre-snapshot byte image.
        # Storage identity must survive ``restore_cpu_state`` so any
        # optimizer state aliased to these tensors stays valid.
        live_regions = list(shard_state.regions)
        pre_storage_ids = [
            r.cpu_shard_bytes.untyped_storage().data_ptr() for r in live_regions
        ]
        pre_bytes = [r.cpu_shard_bytes.detach().clone() for r in live_regions]

        snap = mgr.snapshot_cpu_state()
        assert ChunkId(0) in snap, (
            f"rank {rank}: snapshot must include the sharded chunk"
        )
        region_snaps = snap[ChunkId(0)]["regions"]
        assert region_snaps is not None, (
            f"rank {rank}: sharded chunk should populate 'regions', got None"
        )
        assert len(region_snaps) == len(live_regions), (
            f"rank {rank}: snapshot region count {len(region_snaps)} != "
            f"live region count {len(live_regions)}"
        )

        # Mutate every live region in place — simulates an optimizer
        # step that wrote new weights into the pinned shadows. Region
        # storage is uint8 (raw chunk bytes), so use an integer delta.
        for region in live_regions:
            region.cpu_shard_bytes.add_(torch.ones_like(region.cpu_shard_bytes))
        for region, pre in zip(live_regions, pre_bytes, strict=True):
            assert not torch.equal(region.cpu_shard_bytes, pre), (
                f"rank {rank}: pre-restore mutation expected to differ "
                "from the snapshot's reference bytes"
            )

        # The actual rollback under test.
        mgr.restore_cpu_state(snap)

        # Bit-exact restoration.
        for region, pre in zip(live_regions, pre_bytes, strict=True):
            assert torch.equal(region.cpu_shard_bytes, pre), (
                f"rank {rank}: restore_cpu_state did not reinstate the "
                "pre-mutation bytes for sharded region — phase-2 rollback "
                "would silently advance Mode-C weights"
            )

        # Storage identity preserved (copy_, not rebind). Optimizer
        # state aliased to these tensors must survive.
        post_storage_ids = [
            r.cpu_shard_bytes.untyped_storage().data_ptr() for r in live_regions
        ]
        assert post_storage_ids == pre_storage_ids, (
            f"rank {rank}: restore_cpu_state must preserve underlying "
            f"storage; pre={pre_storage_ids} post={post_storage_ids}"
        )

        # Mutating the snapshot AFTER restore must not touch the live
        # shadow — defends against a degenerate copy_ that aliases.
        for region_snap in region_snaps:
            region_snap.add_(torch.ones_like(region_snap))
        for region, pre in zip(live_regions, pre_bytes, strict=True):
            assert torch.equal(region.cpu_shard_bytes, pre), (
                f"rank {rank}: snapshot tensor shares storage with the "
                "live pinned shard — must be deep-cloned"
            )

        # Tear down via the canonical sharded restore path so the
        # manager exits in a known state (mirrors the round-trip test).
        try:
            mgr.restore_to_gpu()
        except RuntimeError as exc:
            if "not implemented" in str(exc).lower() or "gloo" in str(exc).lower():
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-collective-unsupported: {exc}\n")
                return
            raise

        host.close()
        del pool

    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.gpu  # paired with the rest of the distributed lane
def test_sharded_snapshot_cpu_state_round_trip_2rank(tmp_path) -> None:
    """2-rank gloo: sharded snapshot_cpu_state -> mutate -> restore_cpu_state.

    Closes the residual coverage gap left by the replicated-only unit
    tests (``test_snapshot_cpu_state_restores_mutated_pinned_bytes``,
    ``test_snapshot_cpu_state_independent_storage``): they exercise
    ``_cpu_slots[].cpu_data`` but never touch
    ``_chunk_shards[].regions[].cpu_shard_bytes`` because sharding
    requires a live process group. Without this test, a regression
    in the sharded branch of ``snapshot_cpu_state`` /
    ``restore_cpu_state`` would survive CI — and silently corrupt
    Mode-C weights across a phase-2 rebuild.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_sharded_snapshot_restore_round_trip,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    skip_files = list(tmp_path.glob("rank*.skip"))
    if skip_files:
        reasons = [f.read_text().strip() for f in skip_files]
        pytest.skip(f"gloo does not support required collective(s): {reasons}")
