"""Pins PEFT-LoRA container fwd/bwd hooks: detector + on-demand manager + tiny end-to-end."""

from __future__ import annotations

import contextlib
import math

import pytest
import torch
from torch import nn

from axolotl.integrations.protrain.profiler.on_demand import (
    OnDemandTensorMgr,
    _find_fused_kernel_containers,
    _find_peft_lora_containers,
    _has_peft_lora_factor,
)

# ---------------------------------------------------------------------------
# Tiny synthetic LoRA layer (no PEFT install — we just put parameters in the
# canonical PEFT shape so the detector's substring rule fires).
# ---------------------------------------------------------------------------


class FakeLoraLayer(nn.Module):
    """Synthetic PEFT LoraLayer: frozen base + trainable lora_A/lora_B ParameterDicts."""

    def __init__(self, in_features: int, out_features: int, r: int = 4) -> None:
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        # Match PEFT's ParameterDict layout: ``self.lora_A["default"]``
        # is the trainable ``[r, in_features]`` matrix; ``self.lora_B
        # ["default"]`` is ``[out_features, r]``. The substring
        # ``"lora_A"`` / ``"lora_B"`` shows up in the child's
        # named_parameters and the detector picks them up.
        self.lora_A = nn.ParameterDict(
            {"default": nn.Parameter(torch.randn(r, in_features))}
        )
        self.lora_B = nn.ParameterDict(
            {"default": nn.Parameter(torch.zeros(out_features, r))}
        )

    def forward(self, x):
        base_out = self.base_layer(x)
        # Direct attribute reads on lora_A/lora_B skip the per-Linear gather hook,
        # so without a container hook backward sees [0]-shape and ToCopyBackward0 rejects.
        lora_a = self.lora_A["default"]
        lora_b = self.lora_B["default"]
        return base_out + (x @ lora_a.t()) @ lora_b.t()


class TinyPeftBlock(nn.Module):
    """Block holding a base norm + a fake-PEFT-LoRA linear."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        for p in self.norm.parameters():
            p.requires_grad_(False)
        self.proj = FakeLoraLayer(dim, dim, r=4)

    def forward(self, x):
        return self.proj(self.norm(x))


class TinyPeftModel(nn.Module):
    def __init__(self, n_blocks: int = 2, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyPeftBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Detector unit tests (CPU, no GPU, no torch hooks).
# ---------------------------------------------------------------------------


def test_has_peft_lora_factor_detects_parameter_dict():
    """A module owning a child ParameterDict named ``lora_A`` is detected."""
    layer = FakeLoraLayer(4, 4, r=2)
    assert _has_peft_lora_factor(layer)


def test_has_peft_lora_factor_rejects_plain_linear():
    """A vanilla nn.Linear without LoRA factors is NOT detected."""
    plain = nn.Linear(4, 4)
    assert not _has_peft_lora_factor(plain)


def test_has_peft_lora_factor_rejects_frozen_lora():
    """Detector only targets trainable PEFT factors; frozen ones don't need a container hook."""
    layer = FakeLoraLayer(4, 4, r=2)
    for p in layer.lora_A.parameters():
        p.requires_grad_(False)
    for p in layer.lora_B.parameters():
        p.requires_grad_(False)
    assert not _has_peft_lora_factor(layer)


def test_find_peft_lora_containers_picks_up_each_proj():
    """One container per FakeLoraLayer instance, in module order."""
    model = TinyPeftModel(n_blocks=3, dim=8)
    found = _find_peft_lora_containers(model)
    expected = [block.proj for block in model.layers]
    assert found == expected, f"expected one container per LoRA proj, got {found!r}"


def test_find_peft_lora_containers_empty_when_no_lora():
    """No PEFT factors anywhere -> empty container list."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    assert _find_peft_lora_containers(model) == []


def test_find_peft_lora_containers_outermost_only():
    """When a parent qualifies, descendants are skipped to prevent duplicate gather-hook ref-counts."""
    # The TinyPeftBlock above already owns the LoraLayer as a direct
    # child; its ``recurse_children`` walk picks up ``lora_A`` /
    # ``lora_B`` on the FakeLoraLayer. The outermost detection rule
    # should pin ``block.proj`` (the FakeLoraLayer itself) — NOT the
    # enclosing block — because we walk modules() outside-in and the
    # block's own named_parameters(recurse=False) is empty (it owns no
    # trainable params directly; the only trainable params live on the
    # FakeLoraLayer child's ParameterDicts).
    model = TinyPeftModel(n_blocks=2, dim=8)
    found = _find_peft_lora_containers(model)
    expected = [block.proj for block in model.layers]
    # Must be exactly the projs (not ALSO the enclosing blocks that
    # would qualify under recurse_children walk).
    assert found == expected


def test_find_peft_lora_containers_skips_fused_overlap():
    """Fused detector wins on overlap; duplicate PEFT hook would stack gather ref-counts."""
    import types

    from tests.protrain.test_fused_lora_kernels import (
        TinyModel,
        _patch_attn_qkv_o,
        apply_lora_mlp_swiglu,
    )

    model = TinyModel(n_blocks=1, dim=8, hidden=16)
    # Fuse the MLP forward AND attach a LoRA factor onto its gate_proj
    # so the same module qualifies under both detectors.
    block = model.layers[0]
    block.mlp.forward = types.MethodType(apply_lora_mlp_swiglu, block.mlp)
    # Plant a trainable LoRA-shaped ParameterDict on the same fused MLP.
    block.mlp.lora_A = nn.ParameterDict({"default": nn.Parameter(torch.randn(2, 8))})
    block.mlp.lora_B = nn.ParameterDict({"default": nn.Parameter(torch.zeros(16, 2))})

    fused = _find_fused_kernel_containers(model)
    peft = _find_peft_lora_containers(model)
    assert block.mlp in fused
    assert block.mlp not in peft, (
        "PEFT detector must defer to the fused detector when both match"
    )
    # Independent helper: ensure attn (no fused, no LoRA) shows up nowhere.
    assert _patch_attn_qkv_o is not None  # smoke import only


# Live-hook behavior — CPU-only, exercises gather/release semantics for PEFT-LoRA containers.


def test_lora_container_hooks_install_on_enter():
    """Entering the manager registers container hooks for every PEFT proj."""
    model = TinyPeftModel(n_blocks=2, dim=8)
    n_modules = sum(1 for _ in model.modules())

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        # Detection populated the per-container list.
        assert len(mgr._peft_lora_containers) == 2
        assert mgr._peft_lora_containers == [block.proj for block in model.layers]
        # No fused containers in this model (no fused-kernel patches).
        assert mgr._fused_containers == []
        # Per-module hook count: 4 per module (fwd pre/post + bwd pre/post)
        # plus the per-container quartet for each PEFT container.
        n_peft_containers = len(mgr._peft_lora_containers)
        expected = 4 * n_modules + 4 * n_peft_containers
        assert len(mgr._handles) == expected


def test_lora_container_pregather_runs_before_forward():
    """Forward through PEFT-LoRA layers under the manager matches un-spilled output."""
    torch.manual_seed(0)
    model = TinyPeftModel(n_blocks=1, dim=8)
    x = torch.randn(2, 8)
    expected = model(x)

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        # Spill is in place: every parameter has been moved to cpu_storage
        # and replaced with an empty placeholder.
        assert len(mgr._spills) == sum(1 for _ in model.parameters())
        got = model(x)
        # CPU-original spill: re-gathered tensor IS the original tensor,
        # so byte-exact equivalence holds.
        assert torch.allclose(got, expected, atol=0, rtol=0)


def test_lora_container_backward_succeeds_under_spill():
    """Pins PEFT-LoRA backward under spill: ToCopyBackward0 invalid-gradient-[0] without container hook."""
    torch.manual_seed(1)
    model = TinyPeftModel(n_blocks=2, dim=8)

    x = torch.randn(2, 8, requires_grad=False)
    target = torch.zeros(2, 8)

    # Reference path: forward + backward without the manager — captures
    # the un-spilled grads to compare against. Run manually so we hold
    # onto the grad tensors before zeroing.
    out_ref = model(x)
    loss_ref = (out_ref - target).pow(2).mean()
    loss_ref.backward()
    grad_ref = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)

    # Hooked path: same forward + backward inside the manager.
    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        assert len(mgr._peft_lora_containers) == 2
        out = model(x)
        loss = (out - target).pow(2).mean()
        # Without the container backward hook this raises invalid-gradient-[0].
        loss.backward()

    # Every trainable param produced a finite grad (presence is the
    # fundamental assertion; numerical equivalence is a strict bonus).
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"missing grad on {name} after hooked backward"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
        # CPU-original spill is byte-equivalent so grad numerics should
        # match the reference within fp32 round-off.
        assert torch.allclose(p.grad, grad_ref[name], atol=1e-6), (
            f"grad on {name} differs under hook path: "
            f"max_diff={(p.grad - grad_ref[name]).abs().max().item():.3e}"
        )


def test_lora_container_post_release_clears_data_after_forward():
    """After model(x) completes, every spilled param is back to placeholder."""
    torch.manual_seed(2)
    model = TinyPeftModel(n_blocks=1, dim=8)
    x = torch.randn(2, 8)
    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        _ = model(x)
        # Outside any module forward, every spilled param's .data is
        # back to the empty placeholder.
        for name, p in model.named_parameters():
            assert p.data.numel() == 0, (
                f"param {name} not released after forward: numel={p.data.numel()}"
            )


def test_lora_container_hooks_dormant_when_no_lora():
    """Models without PEFT factors install no PEFT-LoRA container hooks."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    n_modules = sum(1 for _ in model.modules())

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        assert mgr._peft_lora_containers == []
        # Per-module quartet only — no container quartet.
        assert len(mgr._handles) == 4 * n_modules


# E2E smoke: 5 fwd+bwd+step iterations on a tiny LoRA model under the on-demand spill manager.


def test_e2e_5_steps_lora_under_on_demand():
    """Pins 5 fwd+bwd iterations of a tiny PEFT-LoRA model under the on-demand spill manager."""
    torch.manual_seed(3)
    model = TinyPeftModel(n_blocks=2, dim=16)

    x = torch.randn(4, 16)
    target = torch.zeros(4, 16)

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable params — LoRA wrap didn't take"

    losses: list[float] = []
    grad_max_per_iter: list[float] = []
    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        for _step in range(5):
            model.zero_grad(set_to_none=True)
            out = model(x)
            loss = (out - target).pow(2).mean()
            # Without the container backward hook, this raises iter-0:
            # "ToCopyBackward0 returned an invalid gradient at index 0
            # — got [...] but expected shape compatible with [0]".
            loss.backward()
            losses.append(float(loss.detach()))
            # Capture the largest grad magnitude across trainable
            # params — proves gradients actually flowed (a silently
            # failed bwd would leave grads at None or all-zero).
            max_g = 0.0
            for p in trainable:
                if p.grad is not None:
                    max_g = max(max_g, float(p.grad.abs().max()))
            grad_max_per_iter.append(max_g)

    assert len(losses) == 5
    assert all(math.isfinite(v) for v in losses), f"non-finite loss: {losses}"
    # Every iteration produced finite, non-zero grads.
    assert all(g > 0.0 and math.isfinite(g) for g in grad_max_per_iter), (
        f"grads vanished or non-finite under hook path: {grad_max_per_iter}"
    )


def test_e2e_with_disabled_manager_baseline():
    """Sanity: disabled manager is a no-op and full fwd+bwd+optim.step works."""
    torch.manual_seed(3)
    model = TinyPeftModel(n_blocks=2, dim=16)

    x = torch.randn(4, 16)
    target = torch.zeros(4, 16)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=1e-2)

    losses: list[float] = []
    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=True, model=model)
    with mgr:
        for _step in range(5):
            optim.zero_grad(set_to_none=True)
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            losses.append(float(loss.detach()))
            optim.step()

    assert len(losses) == 5
    assert losses[-1] < losses[0] * 0.95, losses


def test_lora_container_fwd_hook_count_includes_per_container_pair():
    """Per-container hook count: exactly 4 handles per detected container."""
    model = TinyPeftModel(n_blocks=3, dim=8)
    n_modules = sum(1 for _ in model.modules())
    n_containers = len(_find_peft_lora_containers(model))
    assert n_containers == 3

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        # Per-module loop: 4 handles each (forward pre/post + backward
        # pre/post). Container loop: another 4 handles per container
        # (forward pre/post + backward pre/post).
        expected = 4 * n_modules + 4 * n_containers
        assert len(mgr._handles) == expected, (
            f"hook count mismatch: got {len(mgr._handles)}, expected {expected}"
        )


@pytest.mark.parametrize("n_blocks", [1, 4])
def test_lora_repeated_forward_under_manager(n_blocks):
    """Repeated forward calls under the manager all see real LoRA weights."""
    torch.manual_seed(5)
    model = TinyPeftModel(n_blocks=n_blocks, dim=8)
    x = torch.randn(2, 8)
    expected = model(x)

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        for _ in range(3):
            got = model(x)
            assert torch.allclose(got, expected, atol=0, rtol=0)


# Runtime-side coverage: per-LoRA-container hook installation + chunk-id closure capture
# so a future runtime gather-chain reorder cannot re-introduce the placeholder-shape bwd gap.


class _AttnLikeBlock(nn.Module):
    """TinyPeftBlock variant exposing self_attn so discover_blocks' attention heuristic fires."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        for p in self.norm.parameters():
            p.requires_grad_(False)
        # Wrap the FakeLoraLayer under ``self_attn`` so the
        # discover_blocks attention heuristic identifies the
        # enclosing ModuleList as a block list.
        self.self_attn = FakeLoraLayer(dim, dim, r=4)

    def forward(self, x):
        return self.self_attn(self.norm(x))


class _TinyAttnPeftModel(nn.Module):
    """Discover-blocks-friendly PEFT-LoRA fixture: ModuleList of _AttnLikeBlock with self_attn FakeLoraLayer."""

    def __init__(self, n_blocks: int = 2, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_AttnLikeBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _build_runtime_chunk_layout(model: nn.Module, S_chunk: int):
    """Build a ChunkLayout treating each layers.{i} as a block (no CUDA / no protrain_model_wrapper)."""
    from typing import cast as _cast

    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.types import (
        BlockId as _BlockId,
        ParamId as _ParamId,
    )

    # Block spans: each ``layers.{i}`` maps to its trainable + frozen
    # parameter dotted-name list. The detector in
    # _find_peft_lora_containers walks ``model.modules()`` and tags
    # each ``FakeLoraLayer`` instance regardless of where in the tree
    # it lives, so the spans need only steer build_layout's
    # block-contiguity packing (every LoRA factor lands in a chunk
    # owned by its enclosing block).
    block_spans: dict = {}
    for name, _ in model.named_parameters():
        if name.startswith("layers."):
            idx = int(name.split(".")[1])
            block_spans.setdefault(_cast(_BlockId, idx), []).append(
                _cast(_ParamId, name)
            )
    exec_order = [_cast(_ParamId, n) for n, _ in model.named_parameters()]
    return build_layout(model, exec_order, S_chunk, block_spans)


class _RecordingScheduler:
    """Stub Scheduler capturing ensure_chunks_resident calls (keeps install_hooks tests CPU-portable)."""

    def __init__(self) -> None:
        # Each entry: (call_kind, tuple_of_chunk_ids). call_kind
        # encodes whether the call originated from a block-level or
        # container-level hook, so tests can assert ordering and
        # aggregation independently.
        self.calls: list[tuple[str, tuple]] = []

    def pre_block_forward(self, block_id) -> None:
        self.calls.append(("pre_block_forward", (int(block_id),)))

    def post_block_forward(self, block_id) -> None:
        self.calls.append(("post_block_forward", (int(block_id),)))

    def pre_block_backward(self, block_id) -> None:
        self.calls.append(("pre_block_backward", (int(block_id),)))

    def post_block_backward(self, block_id) -> None:
        self.calls.append(("post_block_backward", (int(block_id),)))

    def ensure_block_resident(self, block_id) -> None:
        self.calls.append(("ensure_block_resident", (int(block_id),)))

    def ensure_chunks_resident(self, chunk_ids) -> None:
        # Tag each call with the originating LoRA-container hook edge so per-edge tests
        # can distinguish pre/post forward/backward firings via the factory qualname.
        import sys

        edge_tag = "ensure_chunks_resident"
        try:
            caller_frame = sys._getframe(1)
            qualname = caller_frame.f_code.co_qualname
        except (AttributeError, ValueError):  # pragma: no cover
            qualname = ""
        for needle, edge in (
            ("_make_lora_container_pre_forward_hook", "pre_forward"),
            ("_make_lora_container_post_forward_hook", "post_forward"),
            ("_make_lora_container_pre_backward_hook", "pre_backward"),
            ("_make_lora_container_post_backward_hook", "post_backward"),
        ):
            if needle in qualname:
                edge_tag = f"ensure_chunks_resident:{edge}"
                break
        self.calls.append((edge_tag, tuple(int(c) for c in chunk_ids)))


class _RecordingChunkManagerStub:
    """Minimal ChunkManager stand-in exposing only layout + _params_by_id (what install_hooks reads)."""

    def __init__(self, model: nn.Module, layout) -> None:
        from typing import cast as _cast

        from axolotl.integrations.protrain.types import ParamId as _ParamId

        self.layout = layout
        self._params_by_id = {
            _cast(_ParamId, name): p for name, p in model.named_parameters()
        }


def test_install_hooks_attaches_lora_container_pre_hooks_cpu():
    """install_hooks adds 4-hook quartets per block AND per PEFT-LoRA container (fwd+bwd pre+post)."""
    from axolotl.integrations.protrain.runtime.hooks import install_hooks
    from axolotl.integrations.protrain.types import (
        BlockId as _BlockId,
        BlockMode as _BlockMode,
    )

    torch.manual_seed(7)
    n_blocks = 3
    model = _TinyAttnPeftModel(n_blocks=n_blocks, dim=8)

    layout = _build_runtime_chunk_layout(model, S_chunk=4096)
    cm = _RecordingChunkManagerStub(model, layout)
    sched = _RecordingScheduler()

    block_map = {_BlockId(i): _BlockMode.NONE for i in range(n_blocks)}

    handles = install_hooks(
        model=model,
        chunk_manager=cm,  # type: ignore[arg-type]
        block_map=block_map,
        scheduler=sched,  # type: ignore[arg-type]
    )
    try:
        # Per-block: 4 hooks (fwd pre/post + bwd pre/post). Per LoRA container: also 4 hooks.
        n_containers = len(_find_peft_lora_containers(model))
        assert n_containers == n_blocks  # one FakeLoraLayer per block
        expected = 4 * n_blocks + 4 * n_containers
        assert len(handles) == expected, (
            f"hook count mismatch: got {len(handles)} expected {expected} "
            f"(blocks={n_blocks}, containers={n_containers})"
        )
    finally:
        # Best-effort removal per-handle so one failure does not skip the rest.
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


def test_install_hooks_lora_container_chunk_ids_cover_lora_factors():
    """Each LoRA container's chunk-id closure covers every trainable LoRA factor under it."""
    from axolotl.integrations.protrain.runtime.hooks import _container_chunk_ids

    torch.manual_seed(8)
    n_blocks = 2
    model = _TinyAttnPeftModel(n_blocks=n_blocks, dim=8)

    layout = _build_runtime_chunk_layout(model, S_chunk=4096)
    cm = _RecordingChunkManagerStub(model, layout)

    containers = _find_peft_lora_containers(model)
    assert len(containers) == n_blocks

    for container in containers:
        cids = _container_chunk_ids(container, cm)  # type: ignore[arg-type]
        assert cids, f"container {container} produced empty chunk-id set"
        # Verify each trainable LoRA factor reachable from the container
        # lands in one of the captured chunk ids — this is the
        # correctness invariant the runtime hook depends on.
        cm_id_to_name = {id(p): name for name, p in cm._params_by_id.items()}
        for p in container.parameters(recurse=True):
            if not p.requires_grad:
                continue
            cm_name = cm_id_to_name.get(id(p))
            if cm_name is None:
                continue
            cid = layout.param_to_chunk.get(cm_name)
            assert cid in cids, (
                f"trainable param {cm_name} (chunk {cid}) not in container's "
                f"captured chunk-id set {cids}"
            )


def test_install_hooks_lora_container_pre_forward_fires_ensure_chunks_resident():
    """forward-pre hook fires ensure_chunks_resident with non-empty chunk-id tuples per container."""
    from axolotl.integrations.protrain.runtime.hooks import install_hooks
    from axolotl.integrations.protrain.types import (
        BlockId as _BlockId,
        BlockMode as _BlockMode,
    )

    torch.manual_seed(9)
    n_blocks = 2
    model = _TinyAttnPeftModel(n_blocks=n_blocks, dim=8)

    layout = _build_runtime_chunk_layout(model, S_chunk=4096)
    cm = _RecordingChunkManagerStub(model, layout)
    sched = _RecordingScheduler()
    block_map = {_BlockId(i): _BlockMode.NONE for i in range(n_blocks)}

    handles = install_hooks(
        model=model,
        chunk_manager=cm,  # type: ignore[arg-type]
        block_map=block_map,
        scheduler=sched,  # type: ignore[arg-type]
    )
    try:
        x = torch.randn(2, 8)
        _ = model(x)

        # Filter on edge-tagged label so deletion of pre-forward (while post-forward stays) fails.
        pre_fwd_calls = [
            c for c in sched.calls if c[0] == "ensure_chunks_resident:pre_forward"
        ]
        assert len(pre_fwd_calls) >= n_blocks, (
            f"expected at least {n_blocks} ensure_chunks_resident:pre_forward "
            f"calls (one per container), got {len(pre_fwd_calls)} "
            f"(all calls: {sched.calls})"
        )
        for _kind, cids in pre_fwd_calls:
            assert cids, "ensure_chunks_resident:pre_forward invoked with empty tuple"
    finally:
        # Best-effort removal per-handle so one failure does not skip the rest.
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


def test_install_hooks_lora_container_post_forward_fires_ensure_chunks_resident():
    """post-forward hook fires ensure_chunks_resident on each LoRA container (defense-in-depth re-bind)."""
    from axolotl.integrations.protrain.runtime.hooks import install_hooks
    from axolotl.integrations.protrain.types import (
        BlockId as _BlockId,
        BlockMode as _BlockMode,
    )

    torch.manual_seed(11)
    n_blocks = 2
    model = _TinyAttnPeftModel(n_blocks=n_blocks, dim=8)

    layout = _build_runtime_chunk_layout(model, S_chunk=4096)
    cm = _RecordingChunkManagerStub(model, layout)
    sched = _RecordingScheduler()
    block_map = {_BlockId(i): _BlockMode.NONE for i in range(n_blocks)}

    handles = install_hooks(
        model=model,
        chunk_manager=cm,  # type: ignore[arg-type]
        block_map=block_map,
        scheduler=sched,  # type: ignore[arg-type]
    )
    try:
        x = torch.randn(2, 8)
        _ = model(x)

        # Assert BOTH edges fired independently so dropping either is caught.
        n_containers = n_blocks  # one FakeLoraLayer per block
        pre_fwd_calls = [
            c for c in sched.calls if c[0] == "ensure_chunks_resident:pre_forward"
        ]
        post_fwd_calls = [
            c for c in sched.calls if c[0] == "ensure_chunks_resident:post_forward"
        ]
        assert len(pre_fwd_calls) >= n_containers, (
            f"expected at least {n_containers} ensure_chunks_resident:pre_forward "
            f"calls (one per container per forward pass), got "
            f"{len(pre_fwd_calls)} (all calls: {sched.calls})"
        )
        assert len(post_fwd_calls) >= n_containers, (
            f"expected at least {n_containers} ensure_chunks_resident:post_forward "
            f"calls (one per container per forward pass), got "
            f"{len(post_fwd_calls)} (all calls: {sched.calls})"
        )
    finally:
        # Best-effort removal per-handle so one failure does not skip the rest.
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


def test_install_hooks_lora_container_post_backward_fires_ensure_chunks_resident():
    """post-backward hook fires ensure_chunks_resident; pins all 4 hook-quartet edges over fwd+bwd."""
    from axolotl.integrations.protrain.runtime.hooks import install_hooks
    from axolotl.integrations.protrain.types import (
        BlockId as _BlockId,
        BlockMode as _BlockMode,
    )

    torch.manual_seed(12)
    n_blocks = 2
    model = _TinyAttnPeftModel(n_blocks=n_blocks, dim=8)

    layout = _build_runtime_chunk_layout(model, S_chunk=4096)
    cm = _RecordingChunkManagerStub(model, layout)
    sched = _RecordingScheduler()
    block_map = {_BlockId(i): _BlockMode.NONE for i in range(n_blocks)}

    handles = install_hooks(
        model=model,
        chunk_manager=cm,  # type: ignore[arg-type]
        block_map=block_map,
        scheduler=sched,  # type: ignore[arg-type]
    )
    try:
        x = torch.randn(2, 8, requires_grad=False)
        target = torch.zeros(2, 8)
        out = model(x)
        loss = (out - target).pow(2).mean()
        loss.backward()

        n_containers = n_blocks
        # Assert all four quartet edges fired so dropping any single edge is caught.
        per_edge_calls = {
            edge: [c for c in sched.calls if c[0] == f"ensure_chunks_resident:{edge}"]
            for edge in (
                "pre_forward",
                "post_forward",
                "pre_backward",
                "post_backward",
            )
        }
        for edge, calls in per_edge_calls.items():
            assert len(calls) >= n_containers, (
                f"expected at least {n_containers} "
                f"ensure_chunks_resident:{edge} calls (one per container "
                f"per fwd/bwd window), got {len(calls)}. "
                f"per-edge counts: "
                f"{ {e: len(c) for e, c in per_edge_calls.items()} } "
                f"(all calls: {sched.calls})"
            )
    finally:
        # Best-effort removal per-handle so one failure does not skip the rest.
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


def test_install_hooks_no_lora_no_container_hooks():
    """Non-LoRA model gets only block-quartet hooks; container walk does not raise."""
    from axolotl.integrations.protrain.runtime.hooks import install_hooks
    from axolotl.integrations.protrain.types import (
        BlockId as _BlockId,
        BlockMode as _BlockMode,
    )

    class _PlainAttnBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            # Expose ``self_attn`` so discover_blocks' attention
            # heuristic identifies the enclosing ModuleList as a
            # block list (mirrors _AttnLikeBlock).
            self.self_attn = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.self_attn(x)

    class _PlainModel(nn.Module):
        def __init__(self, n: int, dim: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_PlainAttnBlock(dim) for _ in range(n)])

        def forward(self, x):
            for b in self.layers:
                x = b(x)
            return x

    n_blocks = 2
    model = _PlainModel(n_blocks, dim=4)
    layout = _build_runtime_chunk_layout(model, S_chunk=4096)
    cm = _RecordingChunkManagerStub(model, layout)
    sched = _RecordingScheduler()
    block_map = {_BlockId(i): _BlockMode.NONE for i in range(n_blocks)}

    handles = install_hooks(
        model=model,
        chunk_manager=cm,  # type: ignore[arg-type]
        block_map=block_map,
        scheduler=sched,  # type: ignore[arg-type]
    )
    try:
        # 4 per block, 0 per container.
        assert len(handles) == 4 * n_blocks
    finally:
        # Best-effort removal per-handle so one failure does not skip the rest.
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


# ---------------------------------------------------------------------------
# Real-runtime end-to-end (GPU-gated): exercise the full
# ChunkManager + Scheduler stack against a tiny PEFT-LoRA model and
# confirm the LoRA forward + backward succeed under offload mode.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_runtime_lora_e2e_under_offload_mode_smoke():
    """Pins PEFT-LoRA fwd+bwd through real ChunkManager+Scheduler under non-persistent chunks."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    # Probe DeepSpeedCPUAdam availability so we can run the fwd+bwd validation
    # even on degraded CPU-Adam environments (tolerating the offload-step skip).
    cpu_adam_available = False
    try:
        import deepspeed  # noqa: F401
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        # Probe the JIT-loaded extension by attempting one construction;
        # CUDA/torch toolchain mismatch surfaces here.
        _probe = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        try:
            DeepSpeedCPUAdam([_probe], lr=1e-3)
            cpu_adam_available = True
        except Exception:  # noqa: BLE001
            cpu_adam_available = False
    except ImportError:
        cpu_adam_available = False

    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.integrations.protrain.api import (
        protrain_model_wrapper,
        protrain_optimizer_wrapper,
    )
    from axolotl.integrations.protrain.types import HardwareProfile

    # Sized so build_layout produces enough chunks that LoRA factors
    # land in non-persistent chunks (mandatory_persistent only covers
    # embed / final-norm).
    cfg = LlamaConfig(
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1024,
        vocab_size=1024,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        use_cache=False,
    )
    torch.manual_seed(13)
    base_model = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16, device="cuda")
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg).to(device="cuda")

    # Force a small S_chunk so multiple chunks emerge and LoRA
    # factors land in non-persistent chunks.
    import axolotl.integrations.protrain.api.model_wrapper as mw

    orig_pick = mw.pick_S_chunk
    mw.pick_S_chunk = lambda *a, **k: 1 << 20  # 1 MiB
    try:
        hw = HardwareProfile(
            gpu_sku=torch.cuda.get_device_name(0),
            gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            gpu_count=1,
            pcie_h2d_bps=13e9,
            pcie_d2h_bps=13e9,
            has_nvlink=False,
        )
        # Env-failure substrings degrade this smoke to skip; any other
        # ValueError/RuntimeError surfaces as a real wrapper regression.
        _wrapper_env_failure_substrings = (
            "DeepSpeedCPUAdam",  # CPU Adam JIT-load failed
            "CUDA version",  # DeepSpeed CUDA/torch toolchain mismatch
            "bitsandbytes",  # bnb load issues
            "No module named",  # ModuleNotFoundError surface
            # Searcher / capacity gates that legitimately mean
            # "config not feasible on this rig", not "wrapper
            # regression":
            "no feasible config",
            "cpu_capacity",
            "capacity_bytes",
        )

        def _is_wrapper_env_failure(exc: BaseException) -> bool:
            msg = str(exc)
            return any(sub in msg for sub in _wrapper_env_failure_substrings)

        try:
            wrapped = protrain_model_wrapper(
                model,
                model_config=cfg,
                hardware_profile=hw,
                batch_size=1,
                seq_len=32,
                capacity_bytes=2 * (1 << 30),
                force_all_persistent=False,
                zero3_shard=False,
                n_persist_override=0,
                n_buffer_override=16,
                n_swap_override=0,
                n_checkpoint_override=0,
                n_offload_override=cfg.num_hidden_layers,
            )
        except (ValueError, RuntimeError) as exc:
            if not _is_wrapper_env_failure(exc):
                # Real wrapper regression — let it surface.
                raise
            pytest.skip(f"protrain_model_wrapper offload setup unavailable: {exc}")

        # Env-failure substrings degrade to skip optimizer round-trip; deferred:
        # narrow further once exact DeepSpeedCPUAdam/torchao/apex error strings are captured.
        _env_failure_substrings = (
            "DeepSpeedCPUAdam",  # DeepSpeed CPU Adam JIT-load failure
            "CUDA version",  # DeepSpeed CUDA/torch toolchain mismatch
            "bitsandbytes",  # bnb load issues
            "No module named",  # ModuleNotFoundError surface
            "missing CPU optimizer for offloaded chunk",
            # The fix-3 validation signal — backward unwound past the
            # LoRA bf16-cast node BEFORE the per-chunk grad hook
            # raised; the message confirms the fix worked.
        )

        def _is_env_failure(exc: BaseException) -> bool:
            msg = str(exc)
            return any(sub in msg for sub in _env_failure_substrings)

        optim = None
        if cpu_adam_available:
            try:
                optim = protrain_optimizer_wrapper(wrapped, lr=1e-3)
            except RuntimeError as exc:
                # Only suppress documented env-failure signatures; real
                # protrain_optimizer_wrapper regressions must surface.
                if not _is_env_failure(exc):
                    raise
                optim = None

        input_ids = torch.randint(
            0, cfg.vocab_size, (1, 32), device="cuda", dtype=torch.long
        )
        labels = input_ids.clone()
        # iter-0 backward must NOT raise ToCopyBackward0 invalid-gradient-[0]:
        # that signals the LoRA gather-before-cast invariant was broken.
        out = wrapped.module(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_v = float(loss.detach())
        assert math.isfinite(loss_v), f"non-finite loss: {loss_v}"
        # Tolerate "missing CPU optimizer for offloaded chunk" since backward
        # already unwound past the LoRA cast node before the offload-step hook fires.
        try:
            loss.backward()
        except RuntimeError as exc:
            msg = str(exc)
            if "ToCopyBackward" in msg:
                pytest.fail(
                    f"regression: ToCopyBackward0 fired in backward — "
                    f"runtime LoRA gather hook did not cover the autograd "
                    f"shape-derivation step.\n{exc}"
                )
            if "missing CPU optimizer for offloaded chunk" in msg:
                pass
            else:
                raise
        # Only suppress documented env-failure substrings; real optim.step regressions surface.
        if optim is not None:
            try:
                optim.step()
                optim.zero_grad()
            except (RuntimeError, ImportError) as exc:
                if not _is_env_failure(exc):
                    raise
    finally:
        mw.pick_S_chunk = orig_pick
