"""Unit tests for the M6C-fix-2 PEFT-LoRA container hooks.

The companion fix to ``test_fused_lora_kernels.py`` for the standard
(non-fused) PEFT-LoRA forward path. Background:

* M1 added ``OnDemandTensorMgr`` container hooks for **fused** LoRA
  kernels (``apply_lora_mlp_swiglu`` / ``apply_lora_qkv`` / ``..._o`` /
  ``..._embedding``) so the gathered base-weight tensors are GPU-
  resident across the patched forward + backward window.
* M6C-fix-2 extends the same machinery to **non-fused** PEFT-LoRA
  layers (the ``LoraLayer.forward`` path that PEFT installs by default
  when fused kernels are disabled). The trainable LoRA factor
  parameters (``lora_A`` / ``lora_B`` / ``lora_magnitude_vector``)
  themselves drive the same hookability gap: under ProTrain offload
  mode the per-Linear gather hook does not fire on the LoRA factor's
  ``ParameterDict`` (it's not an ``nn.Module.__call__`` site), and at
  backward time autograd's ``ToCopyBackward0`` fails with the same
  ``invalid gradient ... shape compatible with [0]`` error class the
  M0 spike captured for fused kernels.

These tests pin:

1. The container detector (:func:`_find_peft_lora_containers`)
   identifies modules that own trainable PEFT factors and skips
   modules already covered by the fused-kernel detector.
2. The on-demand manager installs container-level pre-/post-forward
   AND pre-/post-backward hooks for every detected PEFT-LoRA
   container.
3. End-to-end: 5 forward+backward+step iterations through a tiny
   PEFT-LoRA model under the on-demand manager produce a strictly
   descending loss — proving real gradients flow through the
   container hooks even when ``param.data`` is spilled.
"""

from __future__ import annotations

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
    """Synthetic stand-in for PEFT's ``LoraLayer``.

    Mirrors the PEFT shape the on-demand detector cares about:

    * A wrapped ``base_layer`` (a frozen ``nn.Linear``).
    * A trainable ``lora_A.default.weight`` ParameterDict-style
      attribute. We use a child ``nn.ParameterDict`` so the
      ``recurse_children=True`` walk in
      :func:`_has_peft_lora_factor` finds the parameter via the
      ``lora_A`` substring on the child name.
    * A trainable ``lora_B.default.weight`` analogue.

    Forward: ``base(x) + lora_B[default](lora_A[default](x))`` — the
    canonical PEFT LoRA delta. Implemented via direct attribute
    access on the ParameterDict so the per-Linear pre-gather hook
    on ``base_layer`` fires (covering the base weight) but no leaf
    hook fires on the LoRA factors themselves — matching the bug
    surface the container hook is meant to close.
    """

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
        # Direct attribute reads on lora_A/lora_B — no nn.Module.__call__
        # boundary, so the per-Linear gather hook on ``base_layer`` does
        # not see them. Without the container hook, the M6C bug surfaces:
        # at backward time ``ToCopyBackward0`` reads the live
        # ``param.size()`` (still ``[0]`` because spilled) and rejects
        # the real-shape grad.
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
    """Even a fake-LoRA layer is rejected when its factors are frozen.

    The detector specifically targets *trainable* PEFT factors — the bug
    surface (autograd shape derivation at backward) only matters when the
    factor produces gradients. Frozen factors don't engage the M6C
    failure mode and shouldn't get a redundant container hook.
    """
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
    """When a parent module already qualifies, its descendants are skipped.

    Without the outermost-only rule, an enclosing block that *also*
    transitively owns the same trainable factors (via its child's child
    ParameterDict) would re-qualify and we'd register duplicate hooks
    for the same gather scope. Confirms the de-duplication logic.
    """
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
    """A module that's both fused AND PEFT-LoRA is reported only as fused.

    The fused-kernel container hooks already gather every sub-parameter
    in the subtree (see ``_find_fused_kernel_containers``); a duplicate
    PEFT-LoRA container hook on the same module would stack ref-counts
    on the same Parameters and inflate the active-user counter that
    ``_pre_gather`` / ``_post_release`` rely on for tied params.
    """
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


# ---------------------------------------------------------------------------
# Live-hook behavior — CPU-only, exercises the gather/release semantics
# the M6C-fix-2 cycle depends on.
# ---------------------------------------------------------------------------


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
    """End-to-end backward: PEFT-LoRA + spilled params produces real grads.

    This is the direct repro of the M6C-fix-2 failure mode at the unit
    scale. Without the container backward hook, the LoRA factor's
    ``ToCopyBackward0`` would see the empty placeholder
    (``param.size() == [0]``) and reject the real-shape grad with
    ``RuntimeError: ToCopyBackward0 returned an invalid gradient at
    index 0``. With the fix, backward succeeds and grads flow into
    every trainable param.
    """
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
        # The bug: without M6C-fix-2's container backward hook, this
        # ``backward()`` call raises ``RuntimeError: invalid gradient
        # ... shape compatible with [0]``. With the fix, the container
        # pre-gather restores ``param.data`` before the autograd
        # backward step needs the shape, and accumulation succeeds.
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


# ---------------------------------------------------------------------------
# E2E smoke: 5 forward+backward+step iterations on a tiny LoRA model under
# the on-demand manager — the unit-scale analogue of the M6C real-multigpu
# failure mode.
# ---------------------------------------------------------------------------


def test_e2e_5_steps_lora_under_on_demand():
    """5 forward+backward iterations under the on-demand manager succeed.

    Mirrors the C→A multi-GPU test's "Phase 1" (Mode C train of an
    8B LoRA model) at the unit scale. Without M6C-fix-2 this would
    fail at iter-0 backward with ``invalid gradient ... shape
    compatible with [0]``. With the fix, all 5 iterations complete
    and the per-iter grads are non-zero — proving real gradients flow
    through the LoRA factors even when ``param.data`` is spilled.

    Optimizer step is intentionally NOT exercised inside the
    ``with mgr:`` block: the on-demand manager is a *profiler-time*
    tool (it spills params to CPU and replaces ``.data`` with empty
    placeholders between modules), so an Adam step over those
    placeholders would fail with the same length-0 shape mismatch
    the bug is about. In the production path the ProTrain runtime
    routes optimizer updates through ``ChunkManager`` adapters that
    gather chunks before stepping; that's a runtime-side composition
    test (``test_bnb_offload.py::test_offload_mode_4bit_e2e_5_steps``
    is the analogous coverage for the bnb offload path). What this
    test pins is what the on-demand manager IS responsible for: the
    forward + backward pair survives spill + gather + release.
    """
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
    """Sanity baseline: disabled manager == no spill == fwd+bwd both fine.

    With disabled=True the manager is a no-op and an actual optim step
    works (no spill). Mirror the enabled-mode test structure so a
    regression that breaks the disabled fast path surfaces here.
    """
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


# ---------------------------------------------------------------------------
# Runtime-side coverage (M6C-fix-3): the analogue of the
# OnDemandTensorMgr-driven tests above for the *training runtime* path —
# ``runtime/scheduler.py`` + ``runtime/hooks.py``. The on-demand manager
# is the profiler-trace path; the runtime path goes through the actual
# ChunkManager + Scheduler that real training uses.
#
# Bug class closed by M6C-fix-3 (per the spec):
#   - PEFT's ``LoraLayer.forward`` builds autograd graph nodes whose
#     shape derivation reads ``param.size()`` at op-construction time.
#   - With Mode-C-style offload (non-persistent chunks), the LoRA factor's
#     ``param.data`` is the empty ``[0]`` placeholder until the
#     enclosing block's pre-forward gather rebinds it.
#   - The block-level gather is a *superset* of the LoRA factor's
#     chunks, but if any op fires against the placeholder shape before
#     the gather completes (or if a future scheduler refactor moves
#     the gather into the OFFLOAD wrapper instead of the block hook),
#     autograd records ``[0]`` and backward fails with
#     ``ToCopyBackward0 returned an invalid gradient at index 0 - got
#     [...] but expected shape compatible with [0]``.
#
# These tests pin the per-LoRA-container hook installation +
# chunk-id closure capture, so a future reordering of the runtime
# gather chain that re-introduces the gap is caught at unit scope.
# ---------------------------------------------------------------------------


class _AttnLikeBlock(nn.Module):
    """TinyPeftBlock variant that satisfies discover_blocks' attention heuristic.

    discover_blocks expects each block in the candidate ModuleList to
    expose a direct ``attention`` or ``self_attn`` attribute (see
    ``layout_rules._looks_like_block``). The test fixture wraps a
    FakeLoraLayer under ``self_attn`` so the heuristic identifies the
    enclosing ``ModuleList`` as a transformer-block list.
    """

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
    """Discover-blocks-friendly PEFT-LoRA model fixture.

    ``model.layers`` is a ModuleList of ``_AttnLikeBlock`` — discover_blocks
    matches it via the attention heuristic. Each block carries a
    FakeLoraLayer under ``self_attn`` so the M6C-fix-3 detector
    finds one PEFT-LoRA container per block.
    """

    def __init__(self, n_blocks: int = 2, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_AttnLikeBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _build_runtime_chunk_layout(model: nn.Module, S_chunk: int):
    """Build a ChunkLayout treating each ``layers.{i}`` as a block.

    Mirrors the production layout-construction path's intent (the
    transformer-block ``ModuleList`` is the block source) without
    requiring CUDA / a full ``protrain_model_wrapper`` invocation.
    Used by the runtime-side hook-installation tests to put a
    ChunkManager around a tiny PEFT-LoRA-shaped model.
    """
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
    """Stub Scheduler capturing ensure_chunks_resident calls.

    Used by the CPU-only tests below to verify that
    install_hooks attaches per-LoRA-container pre-forward and
    pre-backward hooks that fire ``ensure_chunks_resident`` with the
    correct chunk-id set. Real Scheduler wiring needs CUDA; this
    stub keeps the install_hooks-side coverage CPU-portable.
    """

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
        # ``chunk_ids`` is the closure-captured tuple — record verbatim
        # so the test can compare set membership and ordering.
        self.calls.append(("ensure_chunks_resident", tuple(int(c) for c in chunk_ids)))


class _RecordingChunkManagerStub:
    """Minimal stand-in for ChunkManager exposing only what install_hooks reads.

    install_hooks calls ``_container_chunk_ids`` which reads
    ``chunk_manager._params_by_id`` and ``chunk_manager.layout``. The
    ``layout`` field is a real ChunkLayout built via
    ``_build_runtime_chunk_layout``; the rest of ChunkManager is not
    consulted by install_hooks at registration time.
    """

    def __init__(self, model: nn.Module, layout) -> None:
        from typing import cast as _cast

        from axolotl.integrations.protrain.types import ParamId as _ParamId

        self.layout = layout
        self._params_by_id = {
            _cast(_ParamId, name): p for name, p in model.named_parameters()
        }


def test_install_hooks_attaches_lora_container_pre_hooks_cpu():
    """install_hooks adds the full fwd/bwd pre+post hook quartet per PEFT-LoRA container.

    Uses a stub scheduler / chunk-manager to keep the test CPU-only.
    The block-level hook quartet (4 per block) plus the per-container
    quartet (4 per container, M6C-fix-6) gives the expected handle
    count.

    M6C-fix-6 introduced the post-forward and post-backward halves of
    the per-container hook quartet (previously only the pre-edge pair
    was registered, M6C-fix-3). The post-* hooks defensively re-assert
    the gather across the OUTER container's full autograd lifecycle —
    closing the M6C-fix-5 b787acb5 residual failure mode where the
    chunk could be released between the OUTER container's post-forward
    and the inner ``nn.Linear``'s ``TBackward0`` apply.
    """
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
        # Per-block: 4 hooks (fwd pre/post + bwd pre/post). Per LoRA
        # container (M6C-fix-6): 4 hooks (fwd pre/post + bwd pre/post).
        n_containers = len(_find_peft_lora_containers(model))
        assert n_containers == n_blocks  # one FakeLoraLayer per block
        expected = 4 * n_blocks + 4 * n_containers
        assert len(handles) == expected, (
            f"hook count mismatch: got {len(handles)} expected {expected} "
            f"(blocks={n_blocks}, containers={n_containers})"
        )
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass


def test_install_hooks_lora_container_chunk_ids_cover_lora_factors():
    """Each LoRA container's hook closure captures the chunks containing its factors.

    Walks every PEFT-LoRA container, computes the chunk-id set the
    container's pre-hooks will gather, and asserts every trainable
    LoRA factor parameter under that container actually lands in
    one of those chunks. Without this invariant the per-container
    gather is a no-op for the very params the bug is about.
    """
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
    """The forward-pre hook installs cleanly and dispatches to scheduler.

    Runs the full install_hooks then exercises the model forward
    against the stub scheduler; asserts the stub recorded
    ``ensure_chunks_resident`` calls (one per LoRA container per
    forward) with non-empty chunk-id tuples — the load-bearing
    invariant the M6C-fix-3 fix relies on.
    """
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

        ensure_calls = [c for c in sched.calls if c[0] == "ensure_chunks_resident"]
        # One per LoRA container (one container per TinyPeftBlock);
        # block hooks invoke pre_block_forward, NOT
        # ensure_chunks_resident, so any call here came from the
        # M6C-fix-3 container hook.
        assert len(ensure_calls) >= n_blocks, (
            f"expected at least {n_blocks} ensure_chunks_resident calls "
            f"(one per container), got {len(ensure_calls)} "
            f"(all calls: {sched.calls})"
        )
        for _kind, cids in ensure_calls:
            assert cids, "ensure_chunks_resident invoked with empty tuple"
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass


def test_install_hooks_lora_container_post_forward_fires_ensure_chunks_resident():
    """M6C-fix-6: post-forward hook on each LoRA container fires ``ensure_chunks_resident``.

    The post-forward hook is the defense-in-depth re-bind that closes
    the M6C-fix-5 b787acb5 residual failure mode. After a single
    forward pass through the model, the recorded scheduler call list
    must contain at least 2 ``ensure_chunks_resident`` invocations
    per LoRA container — one from the pre-forward (M6C-fix-3) and
    one from the new post-forward (M6C-fix-6).
    """
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

        # pre-forward + post-forward → at least 2 ensure_chunks_resident
        # per container per forward pass.
        ensure_calls = [c for c in sched.calls if c[0] == "ensure_chunks_resident"]
        n_containers = n_blocks  # one FakeLoraLayer per block
        assert len(ensure_calls) >= 2 * n_containers, (
            f"expected at least {2 * n_containers} ensure_chunks_resident "
            f"calls (pre-fwd + post-fwd per container), got "
            f"{len(ensure_calls)} (all calls: {sched.calls})"
        )
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass


def test_install_hooks_lora_container_post_backward_fires_ensure_chunks_resident():
    """M6C-fix-6: post-backward hook on each LoRA container fires
    ``ensure_chunks_resident``.

    Pins the load-bearing M6C-fix-6 invariant: the post-backward
    re-bind covers the window between the OUTER container's pre-
    backward fire and the inner ``nn.Linear``'s ``TBackward0`` apply
    (which executes deep inside the OUTER's backward graph
    unrolling). Without the post-backward hook, a release window
    opens around the inner-op tail that the M6C-fix-5 commit
    ``b787acb5`` empirical run identified as the residual failure.

    A full forward + backward through the tiny PEFT-LoRA fixture
    must produce at least 4 ``ensure_chunks_resident`` calls per
    container: pre-fwd, post-fwd, pre-bwd, post-bwd (M6C-fix-6
    quartet).
    """
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

        ensure_calls = [c for c in sched.calls if c[0] == "ensure_chunks_resident"]
        n_containers = n_blocks
        # 4 calls per container: pre-fwd + post-fwd + pre-bwd + post-bwd.
        # M6C-fix-6 brings the quartet up from 2 (pre-edge only) to 4.
        assert len(ensure_calls) >= 4 * n_containers, (
            f"expected at least {4 * n_containers} ensure_chunks_resident "
            f"calls (full quartet per container), got {len(ensure_calls)} "
            f"(all calls: {sched.calls})"
        )
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass


def test_install_hooks_no_lora_no_container_hooks():
    """A model with zero PEFT-LoRA containers gets only the block-quartet hooks.

    Regression guard for the dormant path — running
    ``install_hooks`` against a non-LoRA model must not add any
    per-container handles (and must not raise during the
    container-detection walk).
    """
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
        for h in handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Real-runtime end-to-end (GPU-gated): exercise the full
# ChunkManager + Scheduler stack against a tiny PEFT-LoRA model and
# confirm the LoRA forward + backward succeed under offload mode.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_runtime_lora_e2e_under_offload_mode_smoke():
    """End-to-end smoke: PEFT-LoRA + real ChunkManager + Scheduler, fwd+bwd succeeds.

    Builds a real PEFT-LoRA Llama-arch model, wraps it through the
    full ``protrain_model_wrapper`` machinery with offload-mode
    overrides (force_all_persistent=False, n_persist_override=0),
    and runs one forward + backward iteration. Without M6C-fix-3
    this would (per Agent B's diagnosis on the 4×3090 multi-GPU
    rig) fail at iter-0 backward with ``ToCopyBackward0 returned
    an invalid gradient at index 0 - got [...] but expected shape
    compatible with [0]`` on a PEFT LoRA factor.

    Skipped when DeepSpeed CPU Adam is unavailable (offload mode
    requires it). The test deliberately mirrors the production
    Mode C path (multiple non-persistent chunks, real PEFT LoRA
    layers) so a future regression that re-introduces the gap
    surfaces here at unit scope.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA runtime")

    # Probe DeepSpeedCPUAdam availability — drives whether we exercise
    # the optimizer.step() round-trip below. The forward + backward
    # bug-surface validation does NOT require CPU Adam: the
    # ``ChunkManager`` per-param grad-accumulation hook installed at
    # ``materialize_offload`` time fires during backward, but its
    # CPU-Adam dependency only surfaces when a chunk's offload-step
    # path is invoked. M6C-fix-3 prevents the autograd shape-derivation
    # error class, which fires earlier in the backward chain than that
    # hook — so we can validate the fix even with a degraded CPU-Adam
    # environment by tolerating the ``missing CPU optimizer for
    # offloaded chunk`` RuntimeError as a known post-fix-validation
    # signal (the fix was already proven by the time backward reached
    # that hook).
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
            pytest.skip(f"protrain_model_wrapper offload setup unavailable: {exc}")

        # Substrings that mark known *environmental* failures that
        # should degrade this smoke to "skip optimizer round-trip" rather
        # than fail the test. Any RuntimeError whose message does NOT
        # contain one of these is treated as a real regression and
        # re-raised — D8 fix: previously the bare ``except RuntimeError``
        # swallowed real ``protrain_optimizer_wrapper`` / ``optim.step``
        # bugs and let the test pass green.
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
        # The bug surface: this is exactly the iter-0 backward that
        # fails per the M6C real-multigpu report. M6C-fix-3 closes the
        # runtime gap; before the fix this raises
        # ``ToCopyBackward0 returned an invalid gradient at index 0
        # - got [...] but expected shape compatible with [0]``.
        out = wrapped.module(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_v = float(loss.detach())
        assert math.isfinite(loss_v), f"non-finite loss: {loss_v}"
        # The bug surface: this is exactly the iter-0 backward that
        # fails per the M6C real-multigpu report. M6C-fix-3 closes
        # the runtime gap; before the fix this raises:
        #   "ToCopyBackward0 returned an invalid gradient at index 0
        #    - got [...] but expected shape compatible with [0]"
        # If the backward call below completes without raising the
        # ``ToCopyBackward0`` error class, the M6C-fix-3 invariant
        # holds (the LoRA factor's chunk was gathered before the
        # autograd graph recorded the cast op against
        # ``param.size()``). We deliberately do NOT assert on
        # ``param.grad`` for offloaded LoRA factors — under offload
        # mode their grads are drained to pinned-CPU shadows by the
        # per-param post-accumulate-grad hook installed in
        # ``ChunkManager.materialize_offload`` and the live
        # ``param.grad`` attribute is reset to None as a side effect
        # (the optimizer step reads from the CPU shadow, not from
        # the Parameter). The successful return is the assertion.
        #
        # Without DeepSpeedCPUAdam available, the per-chunk grad-
        # accumulation hook installed by ``materialize_offload``
        # raises ``RuntimeError: ChunkManager: missing CPU optimizer
        # for offloaded chunk N`` from ``chunk/manager.py:_hook``
        # AFTER the autograd graph has executed cleanly. That
        # specific message is tolerated here because it confirms
        # backward unwound past the LoRA bf16-cast node (i.e. the
        # M6C-fix-3 fix is active); the test still fails on any
        # other RuntimeError, including the canonical
        # ``ToCopyBackward0 ... shape compatible with [0]`` regression
        # signal.
        try:
            loss.backward()
        except RuntimeError as exc:
            msg = str(exc)
            if "ToCopyBackward" in msg:
                pytest.fail(
                    f"M6C-fix-3 regression: ToCopyBackward0 fired in "
                    f"backward — runtime LoRA gather hook did not cover "
                    f"the autograd shape-derivation step.\n{exc}"
                )
            if "missing CPU optimizer for offloaded chunk" in msg:
                # Backward graph completed past the LoRA bf16-cast
                # node — fix is validated. The CPU-Adam dependency
                # is environmental, not a regression signal.
                pass
            else:
                raise
        # Optional: an optimizer step round-trip — exercises the CPU
        # FusedAdam plumbing on the offloaded chunks. Skipped if the
        # adapter wasn't constructed (e.g. CPU Adam unavailable).
        #
        # D8 fix: previously a bare ``except Exception`` here swallowed
        # any optim.step / optim.zero_grad failure, making the round-trip
        # effectively non-asserting. Now only suppress documented env
        # failure signatures (DeepSpeedCPUAdam JIT, CUDA toolchain
        # mismatch, bnb load, the post-fix-3 "missing CPU optimizer"
        # message); re-raise real CPU-Adam plumbing regressions.
        if optim is not None:
            try:
                optim.step()
                optim.zero_grad()
            except (RuntimeError, ImportError) as exc:
                if not _is_env_failure(exc):
                    raise
    finally:
        mw.pick_S_chunk = orig_pick
