"""Unit tests for ProTrain M1 — fused LoRA kernel integration.

The on-demand profiler installs per-Linear pre-/post-forward hooks that
gather weights from CPU just before each ``nn.Linear.__call__``. Axolotl's
fused LoRA kernels (``apply_lora_mlp_swiglu``, ``apply_lora_qkv``,
``apply_lora_o``, ``apply_lora_embedding``) bypass that path entirely:
they read child Linear weights via direct attribute access from inside a
monkey-patched container forward. The M0 spike captured the resulting
``RuntimeError: size mismatch ... vec (0)`` — the fused matmul saw the
empty post-spill placeholder.

These tests pin the M1 fix: the on-demand manager detects fused-kernel
containers and installs an additional pre-/post-forward hook on each
container that gathers ALL sub-parameters before the patched forward runs
(symmetric release after). Verified at the helper level (no GPU) and at
the live-hook level (no GPU — hook firing alone is observable on CPU).
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from axolotl.integrations.protrain.profiler.on_demand import (
    OnDemandTensorMgr,
    _find_fused_kernel_containers,
    _is_fused_method,
)


# Synthetic stand-ins for axolotl.kernels.lora.apply_lora_* — same names
# so the on-demand manager's name-based detector matches them, but with
# trivial implementations that read child Linear weight refs directly
# (the same access pattern the real fused kernels use).
def apply_lora_mlp_swiglu(self, x):  # noqa: D401 — stand-in
    """Stand-in MLP fused kernel: reads gate/up/down weights directly.

    Mirrors the real kernel's access pattern (direct attribute reads on
    child Linears) so the on-demand manager's per-Linear gather hooks
    are bypassed exactly the same way. Math: ``down(silu(gate(x)) * up(x))``.
    """
    gate_w = self.gate_proj.weight  # [hidden, dim]
    up_w = self.up_proj.weight  # [hidden, dim]
    down_w = self.down_proj.weight  # [dim, hidden]
    # Exercise the failure mode the M0 spike found: the matmul
    # ``x @ gate_w.t()`` blows up with size mismatch when gate_w.data is
    # the empty post-spill placeholder. Under the M1 fix, the container
    # pre-hook gathers gate_w before this matmul runs.
    h = torch.nn.functional.silu(x @ gate_w.t()) * (x @ up_w.t())
    return h @ down_w.t()


def apply_lora_qkv(self, x):  # noqa: D401 — stand-in
    """Stand-in QKV fused kernel: reads q/k/v weights directly."""
    return (
        x @ self.q_proj.weight.t(),
        x @ self.k_proj.weight.t(),
        x @ self.v_proj.weight.t(),
    )


def apply_lora_o(self, x):  # noqa: D401 — stand-in
    """Stand-in O fused kernel: reads o_proj weight directly."""
    return x @ self.o_proj.weight.t()


def apply_lora_embedding(self, x):  # noqa: D401 — stand-in
    """Stand-in embed fused kernel: reads embed weight directly."""
    return self.weight[x]


class TinyMLP(nn.Module):
    def __init__(self, dim: int = 8, hidden: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        # Match the fused stand-in's swiglu math so the equivalence check
        # in ``test_container_pregather_runs_before_fused_forward`` is
        # against an identical computation rather than a structural shim.
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class TinyAttn(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def apply_qkv(self, x):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def apply_o(self, x):
        return self.o_proj(x)

    def forward(self, x):
        q, k, v = self.apply_qkv(x)
        attn = (q @ k.transpose(-1, -2)).softmax(dim=-1) @ v
        return self.apply_o(attn)


class TinyBlock(nn.Module):
    def __init__(self, dim: int = 8, hidden: int = 16):
        super().__init__()
        self.self_attn = TinyAttn(dim)
        self.mlp = TinyMLP(dim, hidden)

    def forward(self, x):
        return self.mlp(x + self.self_attn(x))


class TinyModel(nn.Module):
    def __init__(self, n_blocks: int = 2, dim: int = 8, hidden: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(dim, hidden) for _ in range(n_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _patch_mlp_swiglu(model: TinyModel) -> list[nn.Module]:
    """Install fused MLP kernel on every block's ``mlp`` (mirrors apply_lora_kernel_patches)."""
    patched: list[nn.Module] = []
    for block in model.layers:
        block.mlp.forward = types.MethodType(apply_lora_mlp_swiglu, block.mlp)
        patched.append(block.mlp)
    return patched


def _patch_attn_qkv_o(model: TinyModel) -> list[nn.Module]:
    """Install fused QKV + O kernels on every block's ``self_attn``."""
    patched: list[nn.Module] = []
    for block in model.layers:
        block.self_attn.apply_qkv = types.MethodType(apply_lora_qkv, block.self_attn)
        block.self_attn.apply_o = types.MethodType(apply_lora_o, block.self_attn)
        patched.append(block.self_attn)
    return patched


# ---------------------------------------------------------------------------
# Detector helpers — pure logic, no torch hooks, no GPU.
# ---------------------------------------------------------------------------


def test_is_fused_method_recognises_swiglu():
    """A MethodType bound to apply_lora_mlp_swiglu is detected."""
    mlp = TinyMLP()
    assert not _is_fused_method(mlp.forward)
    mlp.forward = types.MethodType(apply_lora_mlp_swiglu, mlp)
    assert _is_fused_method(mlp.forward)


def test_is_fused_method_recognises_all_fused_names():
    """All apply_lora_* method bindings are detected."""
    fns = [
        apply_lora_mlp_swiglu,
        apply_lora_qkv,
        apply_lora_o,
        apply_lora_embedding,
    ]
    holder = nn.Linear(2, 2)
    for fn in fns:
        bound = types.MethodType(fn, holder)
        assert _is_fused_method(bound), (
            f"Detector missed fused kernel binding for {fn.__name__}"
        )


def test_is_fused_method_rejects_unrelated_method():
    """Unrelated ``MethodType`` bindings (e.g. plain Linear forward) are NOT flagged."""

    def some_other_method(self, x):
        return x

    holder = nn.Linear(2, 2)
    bound = types.MethodType(some_other_method, holder)
    assert not _is_fused_method(bound)


def test_find_containers_empty_when_unpatched():
    """No containers when the model has no fused-kernel monkey-patch."""
    model = TinyModel()
    assert _find_fused_kernel_containers(model) == []


def test_find_containers_picks_up_mlp_only():
    """Container set lists every patched ``mlp`` (one per block)."""
    model = TinyModel(n_blocks=3)
    patched = _patch_mlp_swiglu(model)
    found = _find_fused_kernel_containers(model)
    assert found == patched, (
        f"expected exactly the patched mlps, got {found!r} vs {patched!r}"
    )


def test_find_containers_picks_up_qkv_and_o():
    """``self_attn`` is a single container even when both apply_qkv and apply_o are fused."""
    model = TinyModel(n_blocks=2)
    patched = _patch_attn_qkv_o(model)
    found = _find_fused_kernel_containers(model)
    assert found == patched, (
        f"expected exactly the patched self_attns, got {found!r} vs {patched!r}"
    )


def test_find_containers_picks_up_mixed_set():
    """Mix of mlp + self_attn fused kernels yields all containers in module order."""
    model = TinyModel(n_blocks=2)
    mlps = _patch_mlp_swiglu(model)
    attns = _patch_attn_qkv_o(model)
    found = _find_fused_kernel_containers(model)
    # Containers appear in ``model.modules()`` order. Each block emits
    # self_attn then mlp under TinyBlock's ``__init__`` order.
    expected_ordered = []
    for sa, mp in zip(attns, mlps, strict=True):
        expected_ordered.extend([sa, mp])
    assert found == expected_ordered, (
        f"expected interleaved [attn, mlp] x n_blocks, got {found!r}"
    )


# ---------------------------------------------------------------------------
# Live-hook behavior (CPU-only — gather/release semantics are device-agnostic).
# ---------------------------------------------------------------------------


def test_container_pregather_runs_before_fused_forward():
    """Under the on-demand manager, fused-MLP forward sees gathered weights, not placeholders.

    Direct repro of the M0 failure mode: without the fix, ``apply_lora_mlp_swiglu``
    reads ``gate_proj.weight.data`` which the manager spilled to CPU and
    replaced with a length-0 placeholder. The matmul then raises ``size
    mismatch ... vec (0)``. With the M1 container hook, the pre-gather
    fires before the patched forward and the matmul receives the real
    weight tensor.

    Runs on CPU using a CPU-original spill path — the spill replaces
    ``param.data`` with an empty CPU tensor, the pre-hook restores it,
    and we assert numerical equivalence with the un-spilled forward.
    """
    torch.manual_seed(0)
    model = TinyModel(n_blocks=1, dim=8, hidden=16)
    _patch_mlp_swiglu(model)

    x = torch.randn(2, 8)
    # Reference output: run BEFORE entering the manager so weights are
    # still resident at their original locations.
    expected = model(x)

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        # Sanity: every direct param has been spilled (cpu_storage populated).
        assert len(mgr._spills) == sum(1 for _ in model.parameters())
        # Sanity: the fused container set is non-empty.
        assert len(mgr._fused_containers) == 1
        # The patched forward must succeed and match the un-spilled output.
        # CPU-original path: ``_pre_gather`` re-points ``param.data`` at
        # ``cpu_storage`` (no device move on a CPU model), so numeric
        # equivalence is byte-exact.
        got = model(x)
        assert torch.allclose(got, expected, atol=0, rtol=0)


def test_container_pregather_fires_for_qkv_and_o():
    """Both apply_qkv and apply_o entrypoints see real weights inside the patched attn forward."""
    torch.manual_seed(1)
    model = TinyModel(n_blocks=1, dim=8, hidden=16)
    _patch_attn_qkv_o(model)

    x = torch.randn(2, 8)
    expected = model(x)

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        assert len(mgr._fused_containers) == 1
        got = model(x)
        assert torch.allclose(got, expected, atol=0, rtol=0)


def test_pre_post_hook_count_includes_per_container_pair():
    """Container hooks add exactly one pre + one post handle per fused container."""
    model = TinyModel(n_blocks=2, dim=8, hidden=16)
    _patch_mlp_swiglu(model)
    _patch_attn_qkv_o(model)

    n_modules = sum(1 for _ in model.modules())
    n_containers = len(_find_fused_kernel_containers(model))
    assert n_containers == 4  # 2 self_attn + 2 mlp

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        # Per-module loop registers 4 handles each (forward pre/post +
        # backward pre/post). Container loop adds another 4 handles per
        # container (forward pre/post + backward pre/post — backward is
        # required because the fused autograd Function keeps base-weight
        # refs on ctx outside the saved-tensors spill path).
        expected = 4 * n_modules + 4 * n_containers
        assert len(mgr._handles) == expected, (
            f"hook count mismatch: got {len(mgr._handles)}, expected {expected}"
        )


def test_post_release_clears_data_after_container_forward():
    """After the container forward returns, every gathered sub-param is back to empty placeholder."""
    torch.manual_seed(2)
    model = TinyModel(n_blocks=1, dim=8, hidden=16)
    _patch_mlp_swiglu(model)
    _patch_attn_qkv_o(model)

    x = torch.randn(2, 8)
    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        _ = model(x)
        # Outside any module forward (we're back in the with-block but
        # past the model call), the post-release hooks have all fired
        # and every spilled param's .data is the empty placeholder.
        for name, p in model.named_parameters():
            assert p.data.numel() == 0, (
                f"param {name} not released after forward: numel={p.data.numel()}"
            )


def test_unpatched_model_has_no_container_overhead():
    """When no fused kernels are installed, the container code path is a no-op."""
    model = TinyModel(n_blocks=2)
    n_modules = sum(1 for _ in model.modules())
    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        assert mgr._fused_containers == []
        assert len(mgr._handles) == 4 * n_modules


def test_disabled_manager_skips_container_detection():
    """Disabled fast path is a true no-op even with a fully-patched model."""
    model = TinyModel(n_blocks=1)
    _patch_mlp_swiglu(model)
    mgr = OnDemandTensorMgr(device="cpu", disabled=True, model=model)
    with mgr:
        # Fast path: no spills, no container hooks.
        assert mgr._fused_containers == []
        assert mgr._handles == []


def test_container_backward_under_fake_fused_autograd_function():
    """Backward through a fake fused-autograd-Function sees real weights.

    Models the exact failure mode the integration test surfaced: the
    real ``LoRA_MLP`` keeps the base weight as a plain Python attribute
    on ``ctx`` (``ctx.weights = (gate_weight, ...)``), bypassing
    ``ctx.save_for_backward`` and therefore the saved-tensors pack/unpack
    spill path. Without the M1 backward subtree hook, the forward
    post-release would clear ``param.data`` to a length-0 placeholder
    before bwd runs and the autograd's matmul against ``ctx.weights[i]``
    would raise ``size mismatch ... vec (0)``.

    Asserting the backward succeeds end-to-end and the param grads match
    the un-spilled reference proves the container's
    ``register_full_backward_pre_hook`` re-gather is the right fix.
    """

    class FakeFusedMatmul(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            # Save x via the standard path (covered by pack/unpack); keep
            # weight as a plain Python attribute (the LoRA_MLP pattern).
            ctx.save_for_backward(x)
            ctx.weight = weight  # outside save_for_backward — needs gather
            return x @ weight.t()

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            weight = ctx.weight
            # This matmul is what blows up with vec(0) when weight.data
            # was cleared by the forward post-release. Same shape match
            # as ``LoRA_MLP.backward``'s ``matmul_lora`` step.
            grad_x = grad_output @ weight
            grad_w = grad_output.t() @ x
            return grad_x, grad_w

    class FakeFusedMLP(nn.Module):
        def __init__(self, dim: int = 8):
            super().__init__()
            self.proj = nn.Linear(dim, dim, bias=False)

    def fused_forward(self, x):
        return FakeFusedMatmul.apply(x, self.proj.weight)

    class FakeBlock(nn.Module):
        def __init__(self, dim: int = 8):
            super().__init__()
            self.mlp = FakeFusedMLP(dim)

        def forward(self, x):
            return self.mlp(x)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeBlock(8)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(7)
    model = FakeModel()
    # Patch the fused MLP forward so our detector picks the container up.
    for layer in model.layers:
        layer.mlp.forward = types.MethodType(apply_lora_mlp_swiglu, layer.mlp)
    # Also override with the FakeFusedMatmul wiring so the autograd Function
    # actually runs (overrides the swiglu stand-in for THIS test only).
    for layer in model.layers:
        layer.mlp.forward = types.MethodType(fused_forward, layer.mlp)

    x = torch.randn(2, 8, requires_grad=True)
    # Reference: forward + backward without the manager.
    y_ref = model(x)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_ref = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    x.grad = None

    # Re-detect: replace the fwd binding with the swiglu name (so detector
    # picks up the container) but keep fused_forward as the actual call —
    # detection is name-based, so we need a fused-name MethodType in place.
    # Trick: re-bind the swiglu name to fused_forward via __name__ alias.
    fused_forward.__name__ = "apply_lora_mlp_swiglu"  # match the detector
    for layer in model.layers:
        layer.mlp.forward = types.MethodType(fused_forward, layer.mlp)

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        assert len(mgr._fused_containers) == 1
        y = model(x)
        loss = y.sum()
        # The backward call is what the M1 backward subtree hook fixes.
        # Without it, this raises ``size mismatch ... vec (0)`` from
        # the autograd Function's bwd matmul against the post-release
        # placeholder.
        loss.backward()

    # Param grads must match the un-spilled reference (within fp32 tol).
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing grad on {name}"
        assert torch.allclose(p.grad, grad_ref[name], atol=1e-6), (
            f"grad on {name} differs under M1 hook path: "
            f"max_diff={(p.grad - grad_ref[name]).abs().max().item():.3e}"
        )


@pytest.mark.parametrize("n_blocks", [1, 3])
def test_container_hooks_handle_repeated_forward(n_blocks):
    """Repeated forward calls under the manager all see real weights."""
    torch.manual_seed(3)
    model = TinyModel(n_blocks=n_blocks, dim=8, hidden=16)
    _patch_mlp_swiglu(model)
    _patch_attn_qkv_o(model)

    x = torch.randn(2, 8)
    expected = model(x)

    mgr = OnDemandTensorMgr(device=torch.device("cpu"), disabled=False, model=model)
    with mgr:
        for _ in range(3):
            got = model(x)
            assert torch.allclose(got, expected, atol=0, rtol=0)
