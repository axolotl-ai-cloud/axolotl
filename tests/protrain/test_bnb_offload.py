"""bnb 4-bit / 8-bit composition with the ProTrain offload path (M3).

These tests close the M3 audit gap: ``load_in_4bit: true`` (QLoRA) +
ProTrain offload mode (Mode C-style — non-persistent chunks live on
pinned CPU and are gathered on demand). The empirical question the
audit raised was whether the bnb-quantized weight tensors (uint8
storage with a Python ``quant_state`` attribute holding the NF4
absmax / double-quant state) survive ProTrain's chunk gather/offload
round-trip.

The investigation that produced this test file (see
``M3 bnb offload-mode integration agent report``) found that the
existing chunk-manager primitives compose with bnb 4-bit cleanly:

1. ``layout._param_bytes`` uses ``numel * element_size`` against the
   uint8-packed storage → byte counts are correct.
2. ``materialize_offload`` copies the uint8 ``param.data`` to pinned
   CPU and rebinds ``param.data`` to an empty placeholder. The
   ``Params4bit`` instance's ``quant_state`` Python attribute and
   its GPU-resident ``absmax`` tensor survive untouched (they live
   on the Parameter object, not on the storage we replaced).
3. ``gather`` rebinds ``param.data`` to a typed view into the GPU
   pool buffer — the ``quant_state`` attribute is still attached
   to the same ``Params4bit`` instance, so ``bnb.MatMul4Bit.forward``
   reads correct dequant metadata.

These tests assert each of those invariants. The third (``5_steps``
e2e) is gated behind ``@pytest.mark.gpu`` because it walks the
ChunkManager + a real ``bnb.nn.Linear4bit`` forward+backward; it
would silently no-op on a CPU-only host because bnb's MatMul4Bit
kernel is CUDA-only.
"""

from __future__ import annotations

from typing import cast

import pytest

from axolotl.integrations.protrain.types import (
    BlockId,
    ParamId,
)

# ---------------------------------------------------------------------------
# Helpers — mirror the patterns used in test_chunk_manager_offload.py
# ---------------------------------------------------------------------------


def _bnb_or_skip():
    """Import bitsandbytes, skipping the test if the install is missing.

    bnb is an optional dependency of axolotl (and a hard requirement
    of QLoRA), so it is reasonable for a CPU-only CI lane to lack
    the package. The protrain test lane runs on hosts with a CUDA
    runtime AND bnb available.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401

        return bnb
    except ImportError as exc:  # pragma: no cover — env probe
        pytest.skip(f"bitsandbytes unavailable: {exc}")


def _tiny_bnb_model(hidden: int = 64, n_layers: int = 2):
    """A tiny model whose transformer-like blocks use ``bnb.nn.Linear4bit``.

    Mirrors ``_tiny_model`` in ``test_chunk_manager_offload.py`` but
    swaps the per-block ``nn.Linear`` for a ``bnb.nn.Linear4bit`` so the
    offload path exercises real ``Params4bit`` storage. Block layout
    matches Llama (``model.layers.{i}``) so ``discover_blocks`` finds
    the block list via ``_KNOWN_BLOCK_PATHS``; each block exposes a
    ``self_attn`` attribute so the attention-heuristic fallback would
    also catch it.
    """
    bnb = _bnb_or_skip()

    import torch
    from torch import nn

    class TinyBlock(nn.Module):
        """One transformer-shaped block: a Linear4bit acting as ``self_attn``."""

        def __init__(self) -> None:
            super().__init__()
            self.self_attn = bnb.nn.Linear4bit(
                hidden,
                hidden,
                bias=False,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
                quant_storage=torch.uint8,
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.self_attn(x)

    class InnerLlama(nn.Module):
        """Inner ``model.layers`` container; matches the Llama path layout."""

        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Linear(hidden, hidden, bias=False).to(
                dtype=torch.bfloat16
            )
            self.layers = nn.ModuleList([TinyBlock() for _ in range(n_layers)])

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.embed_tokens(x)
            for layer in self.layers:
                x = layer(x)
            return x

    class TinyBnbLlama(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = InnerLlama()
            self.lm_head = nn.Linear(hidden, hidden, bias=False).to(
                dtype=torch.bfloat16
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.lm_head(self.model(x))

    torch.manual_seed(0)
    return TinyBnbLlama()


def _build_layout_for(model, S_chunk: int):
    """Build a ChunkLayout where each ``model.layers.{i}`` block is its own chunk."""
    from axolotl.integrations.protrain.chunk.layout import build_layout

    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        if name.startswith("model.layers."):
            idx = int(name.split(".")[2])
            block_spans.setdefault(cast(BlockId, idx), []).append(cast(ParamId, name))

    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    return build_layout(model, exec_order, S_chunk, block_spans)


def _build_chunk_manager(
    model, n_persist: int, S_chunk: int, n_buffer: int | None = None
):
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
# Test 1: bnb 4-bit module discovery
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_bnb_4bit_module_discovery_in_trace() -> None:
    """``discover_blocks`` finds blocks containing ``bnb.nn.Linear4bit``.

    The trace pass relies on ``layout_rules.discover_blocks`` to find
    transformer-like ``nn.ModuleList`` block roots. Because bnb's
    ``Linear4bit`` is a regular ``nn.Module`` subclass, blocks whose
    children are quantized linears must be discovered identically to
    blocks whose children are ``nn.Linear``. This test guards against
    a future refactor that special-cases standard linears in the
    discovery walk and accidentally drops bnb modules.
    """
    bnb = _bnb_or_skip()

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime (Linear4bit needs cuda)")

    from axolotl.integrations.protrain.block.layout_rules import discover_blocks

    model = _tiny_bnb_model(hidden=64, n_layers=4).to("cuda")

    trees = discover_blocks(model)
    assert trees, "discover_blocks returned no block trees for bnb model"

    # Walk the discovered trees and confirm 4 ``model.layers.*`` blocks
    # were enumerated. ``BlockTree.blocks`` is the authoritative list of
    # block instances (the ``model.layers.{i}`` modules) and
    # ``parent_path`` records where in the dotted tree they live.
    block_count = sum(len(tree.blocks) for tree in trees)
    assert block_count == 4, (
        f"discover_blocks expected 4 bnb blocks, got {block_count} "
        f"({[t.parent_path for t in trees]})"
    )
    parent_paths = {tree.parent_path for tree in trees}
    assert "model.layers" in parent_paths, (
        f"discover_blocks did not anchor to model.layers (got {parent_paths})"
    )

    # Confirm the discovered block instances are the bnb-bearing
    # ``TinyBlock``s (i.e. discovery did not silently swap them out for
    # something else) and their inner ``self_attn`` is a real Linear4bit.
    for tree in trees:
        for block in tree.blocks:
            assert isinstance(block.self_attn, bnb.nn.Linear4bit), (
                f"discovered block.self_attn is not Linear4bit: "
                f"{type(block.self_attn).__name__}"
            )
            assert isinstance(block.self_attn.weight, bnb.nn.Params4bit), (
                f"discovered block weight is not Params4bit: "
                f"{type(block.self_attn.weight).__name__}"
            )


# ---------------------------------------------------------------------------
# Test 2: quant_state survives offload-restore round trip
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_quant_state_survives_offload_round_trip() -> None:
    """A ``Params4bit``'s ``quant_state`` survives a chunk-manager round trip.

    The offload path replaces ``param.data`` with an empty placeholder,
    then ``gather`` rebinds it to a typed view into the GPU pool. The
    ``quant_state`` Python attribute (and its GPU-resident ``absmax``)
    must remain attached to the ``Params4bit`` instance throughout, and
    a forward through ``bnb.nn.Linear4bit`` must still produce sensible
    output afterwards.

    This is the key correctness invariant for QLoRA + ProTrain Mode C.
    """
    # Skip-if-missing probe; we don't need the bnb handle here because
    # the model's bnb modules are accessed via their PyTorch instances.
    _bnb_or_skip()

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    # 4 Linear4bit blocks. With S_chunk sized to fit one block's
    # uint8-packed weight per chunk, ``embed_tokens`` and ``lm_head``
    # (the non-block params) absorb the first/last chunk and get
    # marked ``mandatory_persistent`` by the layout — leaving 2-4
    # block-only chunks free to be non-persistent. n_persist=1
    # therefore reliably yields >= 2 non-persistent chunks for the
    # offload pass.
    hidden = 64
    n_layers = 4
    model = _tiny_bnb_model(hidden=hidden, n_layers=n_layers).to("cuda")

    # Trigger the lazy quantization by running one forward — bnb only
    # populates ``quant_state`` once Params4bit.cuda() OR a Linear4bit
    # forward call has happened. ``.to("cuda")`` above takes care of
    # the move; this forward populates the per-weight state2 etc.
    x0 = torch.randn(2, hidden, dtype=torch.bfloat16, device="cuda")
    y_pre = model(x0).detach().clone()

    # Snapshot every Linear4bit's pre-offload quant_state identity and
    # absmax bytes so we can compare against the post-restore state.
    pre_state = {}
    for i in range(n_layers):
        layer = model.model.layers[i].self_attn
        qs = layer.weight.quant_state
        assert qs is not None, (
            f"model.layers.{i}.self_attn.weight.quant_state is None pre-offload"
        )
        pre_state[i] = {
            "qs_id": id(qs),
            "absmax_bytes": qs.absmax.detach().clone(),
            "absmax_device": qs.absmax.device,
            "shape": qs.shape,
            "quant_type": qs.quant_type,
        }

    # Build the chunk manager. We want each block's Linear4bit weight
    # to land in its own chunk AND we want embed_tokens/lm_head (the
    # non-block params) to land in chunks separate from any block, so
    # the non-block chunks become mandatory_persistent and the
    # block-only chunks can offload. embed_tokens is bf16 64*64 = 8192
    # bytes; a single Linear4bit weight is 64*64/2 = 2048 packed bytes;
    # an S_chunk of 4096 gives embed_tokens its own (oversize) chunk
    # and each block weight its own chunk.
    S_chunk = 4096
    mgr, layout, pool, host = _build_chunk_manager(model, n_persist=1, S_chunk=S_chunk)
    # Sanity: layout produced enough non-persistent chunks to exercise.
    nonp_count = sum(
        1
        for cid in range(layout.N_chunk)
        if cid >= 1 and cid not in layout.mandatory_persistent
    )
    assert nonp_count >= 2, (
        f"test setup wants >= 2 non-persistent chunks, got {nonp_count} "
        f"(N_chunk={layout.N_chunk}, "
        f"mandatory={sorted(layout.mandatory_persistent)})"
    )

    # Offload — non-persistent chunks' param.data goes to pinned CPU.
    freed = mgr.materialize_offload()
    assert freed > 0, "materialize_offload freed 0 bytes (expected > 0)"

    # The Params4bit instance's quant_state must still be attached even
    # though param.data is now an empty placeholder. This is the
    # critical post-offload invariant — without it, a subsequent
    # gather + forward would crash inside bnb.MatMul4Bit because dequant
    # metadata went missing.
    for i in range(n_layers):
        layer = model.model.layers[i].self_attn
        qs = layer.weight.quant_state
        assert qs is not None, (
            f"layers.{i}.self_attn.weight.quant_state vanished after offload"
        )
        assert id(qs) == pre_state[i]["qs_id"], (
            f"layers.{i}.self_attn.weight.quant_state was replaced (id mismatch)"
        )
        # absmax stays on the GPU — it's owned by the QuantState
        # Python object, not the chunk-managed ``data`` storage.
        assert qs.absmax.device == pre_state[i]["absmax_device"], (
            f"layers.{i}.self_attn.weight.quant_state.absmax migrated devices: "
            f"was {pre_state[i]['absmax_device']}, now {qs.absmax.device}"
        )
        assert torch.equal(qs.absmax, pre_state[i]["absmax_bytes"]), (
            f"layers.{i}.self_attn.weight.quant_state.absmax bytes changed"
        )

    # Gather every non-persistent chunk back. Linear4bit forward must
    # then succeed and produce numerically-identical output.
    for cid in sorted(mgr._non_persistent_ids):
        mgr.gather(cid)

    # Confirm post-gather quant_state attribute is still intact and
    # param.data is GPU-resident at the right shape.
    for i in range(n_layers):
        layer = model.model.layers[i].self_attn
        assert layer.weight.data.device.type == "cuda"
        assert layer.weight.data.numel() > 0
        qs = layer.weight.quant_state
        assert id(qs) == pre_state[i]["qs_id"], (
            f"layers.{i}.self_attn quant_state replaced during gather"
        )

    # End-to-end correctness: forward should match pre-offload bit-for-bit
    # because we never modified any weight bytes — only moved them.
    y_post = model(x0)
    assert torch.allclose(y_pre, y_post, rtol=0, atol=0), (
        "Linear4bit forward produced different output after offload-restore "
        "round trip — quant_state metadata is out of sync with stored bytes"
    )

    mgr.uninstall()
    host.close()
    del pool


# ---------------------------------------------------------------------------
# Test 3: 5-step training smoke through ProTrain offload + bnb 4-bit
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_offload_mode_4bit_e2e_5_steps() -> None:
    """Five-step training through Linear4bit + ProTrain offload mode.

    Builds a tiny LoRA-adapted bnb 4-bit model, materializes the
    offload, and runs 5 manual forward + backward + gather/offload
    iterations. Asserts:

    1. All five steps complete without exception (gather + bnb dequant
       + LoRA adapter forward + backward + offload all compose).
    2. The last step's loss is strictly less than the first step's
       — proves real gradients flowed back through the LoRA adapters.

    This is the unit-scale analogue of the 8B + 4-bit Mode C smoke
    that gated the M3 acceptance. Keeping it tiny means the test
    runs in a few seconds in CI rather than minutes.
    """
    # Skip-if-missing probe; the bnb instances live inside the model
    # factory and are accessed via PyTorch's module tree, not directly.
    _bnb_or_skip()

    import torch
    from torch import nn

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    hidden = 64
    n_layers = 4
    model = _tiny_bnb_model(hidden=hidden, n_layers=n_layers).to("cuda")

    # Freeze all base weights inside the block sequence — those are
    # the params that will be chunk-managed and offloaded.
    for layer in model.model.layers:
        for p in layer.parameters():
            p.requires_grad_(False)
    # embed_tokens / lm_head are outside the block sequence and will
    # land in mandatory_persistent chunks; freeze them too so the only
    # trainable params are the LoRA adapters added below — the test
    # is about offload + bnb correctness, not full base-weight training.
    for p in model.model.embed_tokens.parameters():
        p.requires_grad_(False)
    for p in model.lm_head.parameters():
        p.requires_grad_(False)

    # Tiny LoRA adapter set, kept OUTSIDE the chunked block sequence —
    # they live as ``model.lora_adapters.{i}`` so the layout's
    # block_spans (built from ``model.layers.*``) does not claim them.
    # Non-block params land in mandatory_persistent chunks (always
    # GPU-resident, never offloaded), so the trainable LoRA grads do
    # not engage the per-param offload-time grad hook (which would
    # require a CPU optimizer attached to the chunk manager).
    class LoRAAdapter(nn.Module):
        def __init__(self, in_f: int, out_f: int, r: int = 2) -> None:
            super().__init__()
            self.lora_a = nn.Linear(in_f, r, bias=False).to(
                dtype=torch.bfloat16, device="cuda"
            )
            self.lora_b = nn.Linear(r, out_f, bias=False).to(
                dtype=torch.bfloat16, device="cuda"
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.lora_b(self.lora_a(x))

    model.lora_adapters = nn.ModuleList(
        [LoRAAdapter(hidden, hidden) for _ in range(n_layers)]
    )

    # Patch each block's forward to add the corresponding LoRA delta
    # AFTER the base bnb forward — same algebraic shape as a real QLoRA
    # adapter, but with the adapter layer kept outside the block tree.
    for i, block in enumerate(model.model.layers):
        adapter = model.lora_adapters[i]
        base_forward = block.forward

        def _patched(x, _base=base_forward, _adapter=adapter):
            return _base(x) + _adapter(x)

        block.forward = _patched

    # Prime quant_state via one forward.
    x = torch.randn(2, hidden, dtype=torch.bfloat16, device="cuda")
    _ = model(x)

    # Build chunk manager with overrides forcing the offload path:
    # n_persist=1, S_chunk small enough that each block's params land in
    # their own chunk separate from embed_tokens/lm_head (the non-block
    # params, which become mandatory_persistent). n_buffer is sized to
    # the number of non-persistent chunks so a naive "gather all up
    # front" pattern fits — a real run uses a tighter scheduling rhythm
    # but the correctness invariant we're checking (bnb dequant works
    # against the rebound buffer) doesn't depend on the schedule.
    S_chunk = 4096
    mgr, layout, pool, host = _build_chunk_manager(
        model, n_persist=1, S_chunk=S_chunk, n_buffer=n_layers
    )
    freed = mgr.materialize_offload()
    assert freed > 0, (
        f"materialize_offload freed 0 bytes — no non-persistent chunks "
        f"(N_chunk={layout.N_chunk}, "
        f"mandatory={sorted(layout.mandatory_persistent)})"
    )

    # Build a tiny optimizer over the LoRA-adapter params only — we
    # don't need ProTrain's per-chunk optim adapter for this test;
    # the goal is to prove the gather + bnb dequant + adapter
    # backprop + offload sequence works.
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable params — LoRA wrap didn't take"
    optim = torch.optim.AdamW(trainable, lr=1e-3)

    # Helper: gather every non-persistent chunk before forward, offload
    # after the optim step. This mimics the all-resident approximation
    # of what the block scheduler does on a real run; a finer-grained
    # gather/offload schedule isn't needed to validate the bnb
    # composition correctness invariant the M3 audit cares about.
    nonp = sorted(mgr._non_persistent_ids)

    losses: list[float] = []
    target = torch.zeros(2, hidden, dtype=torch.bfloat16, device="cuda")

    for _step in range(5):
        for cid in nonp:
            mgr.gather(cid)
        out = model(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        for cid in nonp:
            mgr.offload(cid)
        losses.append(float(loss.detach()))

    # 5 steps completed; loss should descend monotonically on this
    # trivial regression-to-zero objective. Use a tolerance so the
    # last step is required to be at least 5% lower than the first
    # — far enough below noise that a regression in the gather path
    # (e.g. quant_state desyncs across iterations) would fail it.
    assert len(losses) == 5
    assert losses[-1] < losses[0] * 0.95, (
        f"loss did not descend across 5 steps: {losses}"
    )

    mgr.uninstall()
    host.close()
    del pool
