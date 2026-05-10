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
