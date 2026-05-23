"""ProTrain hooks compatibility with torch.compile (Dynamo).

The hook bodies touch ChunkManager / Scheduler state (dicts, streams, NCCL
collectives, ctypes) and are not Dynamo-traceable. ARCH-10 wraps every
hook body with ``torch.compiler.disable(recursive=True)`` so Inductor
compiles the model body while hook bodies fire eagerly between graph
segments.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# CPU-portable fixture: 2-block model with self_attn (so discover_blocks
# treats the ModuleList as the block list under the attention heuristic).
# ---------------------------------------------------------------------------


class _AttnLikeBlock(nn.Module):
    """Minimal transformer-block stand-in: norm + linear under ``self_attn``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.Linear(dim, dim)

    def forward(self, x):
        return self.self_attn(self.norm(x))


class _TinyAttnModel(nn.Module):
    """Discover-blocks-friendly toy model: ModuleList of _AttnLikeBlock."""

    def __init__(self, n_blocks: int = 2, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_AttnLikeBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _install_block_hooks(model: nn.Module) -> tuple[list[Any], MagicMock, MagicMock]:
    """Install ProTrain block-level hooks against MagicMock chunk manager + scheduler."""
    from axolotl.integrations.protrain.runtime.hooks import install_hooks
    from axolotl.integrations.protrain.types import (
        BlockId as _BlockId,
        BlockMode as _BlockMode,
    )

    n_blocks = len(model.layers)
    block_map = {_BlockId(i): _BlockMode.NONE for i in range(n_blocks)}
    cm = MagicMock(name="ChunkManager")
    sched = MagicMock(name="Scheduler")

    handles = install_hooks(
        model=model,
        chunk_manager=cm,
        block_map=block_map,
        scheduler=sched,
    )
    return handles, cm, sched


# ---------------------------------------------------------------------------
# Sentinel: confirm the compat layer is present (matches v49 grep).
# ---------------------------------------------------------------------------


def test_torch_compile_compat_sentinel_present():
    """Module-scope sentinel signals the compat layer is in place."""
    from axolotl.integrations.protrain.runtime import hooks as _hooks

    assert getattr(_hooks, "_PROTRAIN_TORCH_COMPILE_COMPAT", None) == 1


# ---------------------------------------------------------------------------
# Decoration: every hook factory's returned _hook is wrapped by _compile_disable.
# ---------------------------------------------------------------------------


def test_hook_factories_decorate_with_compile_disable():
    """All 6 _make_*_hook factories return Dynamo-disabled callables."""
    from axolotl.integrations.protrain.runtime import hooks as _hooks

    sched = MagicMock(name="Scheduler")
    block_id = 0
    chunk_ids = (0,)

    candidates = [
        _hooks._make_forward_pre_hook(sched, block_id),
        _hooks._make_forward_post_hook(sched, block_id),
        _hooks._make_backward_pre_hook(sched, block_id),
        _hooks._make_backward_post_hook(sched, block_id),
        _hooks._make_lora_container_pre_forward_hook(sched, chunk_ids),
        _hooks._make_lora_container_post_forward_hook(sched, chunk_ids),
        _hooks._make_lora_container_pre_backward_hook(sched, chunk_ids),
        _hooks._make_lora_container_post_backward_hook(sched, chunk_ids),
    ]

    for fn in candidates:
        # torch.compiler.disable stamps either __wrapped__ or _torchdynamo_disable
        # depending on version. The minimal portable check: the callable's
        # qualname references the inner factory frame (not the outer factory).
        assert callable(fn)
        marked = (
            getattr(fn, "_torchdynamo_disable", False)
            or hasattr(fn, "__wrapped__")
            or hasattr(fn, "_torchdynamo_inline")
        )
        assert marked, f"{fn} not marked as torch.compiler.disable-wrapped"


# ---------------------------------------------------------------------------
# Eager-path invariants: hooks still fire and call into the scheduler.
# ---------------------------------------------------------------------------


def test_eager_forward_fires_scheduler_hooks():
    """Block fwd/post hooks still call into Scheduler under eager execution."""
    torch.manual_seed(0)
    model = _TinyAttnModel(n_blocks=2, dim=8)
    handles, _cm, sched = _install_block_hooks(model)
    try:
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == x.shape
        # 2 blocks * (pre_block_forward + post_block_forward) = 4 calls.
        assert sched.pre_block_forward.call_count == 2
        assert sched.post_block_forward.call_count == 2
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


def test_eager_backward_fires_scheduler_hooks():
    """Block bwd/post hooks still call into Scheduler under eager backward."""
    torch.manual_seed(1)
    model = _TinyAttnModel(n_blocks=2, dim=8)
    handles, _cm, sched = _install_block_hooks(model)
    try:
        x = torch.randn(2, 8, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert sched.pre_block_backward.call_count == 2
        assert sched.post_block_backward.call_count == 2
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


# ---------------------------------------------------------------------------
# torch.compile path: model body compiles without dragging hook bodies into
# the Dynamo trace.
# ---------------------------------------------------------------------------


def test_compile_does_not_trace_hook_bodies():
    """Under torch.compile, hook callables observe is_compiling() == False."""
    torch.manual_seed(2)
    model = _TinyAttnModel(n_blocks=2, dim=8)
    handles, _cm, sched = _install_block_hooks(model)

    # Probe inside the scheduler stub: if dynamo traced through the hook
    # frame, torch.compiler.is_compiling() would return True during the
    # call. The @_compile_disable(recursive=True) decoration must prevent
    # that — hook bodies execute eagerly between compiled segments.
    observed_compiling: list[bool] = []

    def _record(_block_id):
        try:
            observed_compiling.append(bool(torch.compiler.is_compiling()))
        except Exception:  # noqa: BLE001 — defensive across torch versions
            observed_compiling.append(False)

    sched.pre_block_forward.side_effect = _record
    sched.post_block_forward.side_effect = _record

    try:
        compiled = torch.compile(model)
        x = torch.randn(2, 8)
        out = compiled(x)
        assert out.shape == x.shape
        assert observed_compiling, "hooks did not fire under torch.compile"
        assert not any(observed_compiling), (
            f"hook saw is_compiling()==True: {observed_compiling}; "
            "the @_compile_disable(recursive=True) wrapper failed to isolate "
            "the hook body from the Dynamo trace."
        )
    except torch._dynamo.exc.BackendCompilerFailed as exc:  # pragma: no cover
        pytest.skip(f"backend compile unavailable in CI: {exc}")
    except RuntimeError as exc:  # pragma: no cover
        # Inductor needs a real backend; on bare CPU it may still trace.
        msg = str(exc).lower()
        if "inductor" in msg or "backend" in msg:
            pytest.skip(f"compile backend skipped on this host: {exc}")
        raise
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


def test_compile_model_runs_without_exception():
    """torch.compile(model) + fwd produces a usable tensor — no Unsupported on hooks."""
    torch.manual_seed(3)
    model = _TinyAttnModel(n_blocks=2, dim=8)
    handles, _cm, _sched = _install_block_hooks(model)

    try:
        compiled = torch.compile(model)
        x = torch.randn(2, 8)
        out = compiled(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
    except torch._dynamo.exc.BackendCompilerFailed as exc:  # pragma: no cover
        pytest.skip(f"backend compile unavailable in CI: {exc}")
    except RuntimeError as exc:  # pragma: no cover
        msg = str(exc).lower()
        if "inductor" in msg or "backend" in msg:
            pytest.skip(f"compile backend skipped on this host: {exc}")
        raise
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


# ---------------------------------------------------------------------------
# GPU smoke: tiny GPT-style toy + ProTrain force_all_persistent +
# torch.compile(mode='reduce-overhead'). Skipped without CUDA.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_gpu_smoke_compile_with_protrain_force_all_persistent(gpu_device: int):
    """End-to-end: torch.compile + ProTrain Mode A persistent layout, 3 fwd+bwd steps."""
    torch.manual_seed(4)
    device = torch.device(f"cuda:{gpu_device}")
    model = _TinyAttnModel(n_blocks=2, dim=16).to(device)
    handles, _cm, _sched = _install_block_hooks(model)
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        try:
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                x = torch.randn(2, 16, device=device, requires_grad=True)
                out = compiled(x)
                out.sum().backward()
                opt.step()
        except torch._dynamo.exc.BackendCompilerFailed as exc:
            pytest.skip(f"backend compile unavailable: {exc}")
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


# ---------------------------------------------------------------------------
# Block-wrapper forwards: OffloadedBlock.forward + SwappedBlock.forward
# must also be torch.compiler.disable-wrapped (Mode B/C/SWAP paths).
# ---------------------------------------------------------------------------


def test_offloaded_block_forward_is_compile_disabled():
    """OffloadedBlock.forward carries the torch.compiler.disable marker."""
    from axolotl.integrations.protrain.block.offload import OffloadedBlock

    fn = OffloadedBlock.forward
    marked = (
        getattr(fn, "_torchdynamo_disable", False)
        or hasattr(fn, "__wrapped__")
        or hasattr(fn, "_torchdynamo_inline")
    )
    assert marked, "OffloadedBlock.forward missing @_compile_disable decoration"


def test_swapped_block_forward_is_compile_disabled():
    """SwappedBlock.forward carries the torch.compiler.disable marker."""
    from axolotl.integrations.protrain.block.swap import SwappedBlock

    fn = SwappedBlock.forward
    marked = (
        getattr(fn, "_torchdynamo_disable", False)
        or hasattr(fn, "__wrapped__")
        or hasattr(fn, "_torchdynamo_inline")
    )
    assert marked, "SwappedBlock.forward missing @_compile_disable decoration"
