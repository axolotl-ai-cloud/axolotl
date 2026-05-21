"""bnb 4-bit / 8-bit composition with the ProTrain offload path: gather/offload must not perturb ``quant_state``."""

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
    """Import bitsandbytes or skip — CPU-only CI lanes may lack the optional package."""
    try:
        import bitsandbytes as bnb  # noqa: F401

        return bnb
    except ImportError as exc:  # pragma: no cover — env probe
        pytest.skip(f"bitsandbytes unavailable: {exc}")


def _tiny_bnb_model(hidden: int = 64, n_layers: int = 2):
    """A tiny Llama-shaped model whose blocks use ``bnb.nn.Linear4bit`` so the offload path hits real ``Params4bit`` storage."""
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
    """``discover_blocks`` finds blocks containing ``bnb.nn.Linear4bit`` (no special-casing of standard linears)."""
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
    """A ``Params4bit``'s ``quant_state`` survives a chunk-manager offload/gather round trip (QLoRA + Mode C invariant)."""
    # Skip-if-missing probe; we don't need the bnb handle here because
    # the model's bnb modules are accessed via their PyTorch instances.
    _bnb_or_skip()

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    torch.cuda.empty_cache()

    # n_persist=1 with this S_chunk leaves >= 2 non-persistent block-only chunks to exercise.
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
    try:
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

        # quant_state must still be attached after offload; otherwise gather + forward would crash in bnb.MatMul4Bit.
        for i in range(n_layers):
            layer = model.model.layers[i].self_attn
            qs = layer.weight.quant_state
            assert qs is not None, (
                f"layers.{i}.self_attn.weight.quant_state vanished after offload"
            )
            assert id(qs) == pre_state[i]["qs_id"], (
                f"layers.{i}.self_attn.weight.quant_state was replaced (id mismatch)"
            )
            # absmax is owned by the QuantState object, not the chunk-managed storage.
            assert qs.absmax.device == pre_state[i]["absmax_device"], (
                f"layers.{i}.self_attn.weight.quant_state.absmax migrated devices: "
                f"was {pre_state[i]['absmax_device']}, now {qs.absmax.device}"
            )
            assert torch.equal(qs.absmax, pre_state[i]["absmax_bytes"]), (
                f"layers.{i}.self_attn.weight.quant_state.absmax bytes changed"
            )

        # Gather every non-persistent chunk back; Linear4bit forward must still produce identical output.
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
    finally:
        # Always free pinned-host buffers and chunk-manager state so a failure cannot bleed into later GPU tests.
        mgr.uninstall()
        host.close()
        del pool


# ---------------------------------------------------------------------------
# Test 3: 5-step training smoke through ProTrain offload + bnb 4-bit
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_offload_mode_4bit_e2e_5_steps() -> None:
    """Five-step Linear4bit + ProTrain offload training smoke; loss must descend across the window."""
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

    # n_persist=1, S_chunk sized so each block weight gets its own chunk and embed/lm_head become mandatory_persistent.
    S_chunk = 4096
    mgr, layout, pool, host = _build_chunk_manager(
        model, n_persist=1, S_chunk=S_chunk, n_buffer=n_layers
    )
    try:
        freed = mgr.materialize_offload()
        assert freed > 0, (
            f"materialize_offload freed 0 bytes — no non-persistent chunks "
            f"(N_chunk={layout.N_chunk}, "
            f"mandatory={sorted(layout.mandatory_persistent)})"
        )

        # Optimizer over LoRA-adapter params only; we only need to prove gather + dequant + backprop + offload composes.
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable, "no trainable params — LoRA wrap didn't take"
        optim = torch.optim.AdamW(trainable, lr=1e-3)

        # All-resident approximation: gather every non-persistent chunk before forward, offload after step.
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

        # 5% headroom over noise: a regression in the gather path (e.g. quant_state desync) would fail this.
        assert len(losses) == 5
        assert losses[-1] < losses[0] * 0.95, (
            f"loss did not descend across 5 steps: {losses}"
        )
    finally:
        # Always free pinned-host buffers and chunk-manager state so a failure cannot bleed into later GPU tests.
        mgr.uninstall()
        host.close()
        del pool
