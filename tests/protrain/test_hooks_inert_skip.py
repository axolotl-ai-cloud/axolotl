"""Predicate + integration tests for the Mode-A all-persistent install_hooks skip."""

from __future__ import annotations

import contextlib

import torch
from torch import nn

from axolotl.integrations.protrain.runtime.hooks import (
    _is_runtime_inert,
    install_hooks,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    ChunkId,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures (CPU-only; no CUDA, no real ChunkManager wiring).
# ---------------------------------------------------------------------------


class _LoraLayer(nn.Module):
    """Minimal PEFT-LoraLayer-shaped module so the container detector fires."""

    def __init__(self, in_features: int, out_features: int, r: int = 4) -> None:
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.ParameterDict(
            {"default": nn.Parameter(torch.randn(r, in_features))}
        )
        self.lora_B = nn.ParameterDict(
            {"default": nn.Parameter(torch.zeros(out_features, r))}
        )

    def forward(self, x):
        a = self.lora_A["default"]
        b = self.lora_B["default"]
        return self.base_layer(x) + (x @ a.t()) @ b.t()


class _AttnLikeBlock(nn.Module):
    """Block exposing self_attn so discover_blocks identifies the ModuleList."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        for p in self.norm.parameters():
            p.requires_grad_(False)
        self.self_attn = _LoraLayer(dim, dim, r=4)

    def forward(self, x):
        return self.self_attn(self.norm(x))


class _TinyModel(nn.Module):
    def __init__(self, n_blocks: int = 2, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_AttnLikeBlock(dim) for _ in range(n_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _SchedulerStub:
    """Minimal Scheduler stand-in accepting attribute writes and recording calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._is_inert: bool = False

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
        self.calls.append(("ensure_chunks_resident", tuple(int(c) for c in chunk_ids)))


class _LayoutStub:
    """Frozen-dataclass-bypassing ChunkLayout stand-in (predicate reads N_chunk + param_to_chunk only)."""

    def __init__(
        self, n_chunk: int, param_to_chunk: dict, block_to_chunks: dict
    ) -> None:
        self.N_chunk = n_chunk
        self.param_to_chunk = param_to_chunk
        self.block_to_chunks = block_to_chunks


class _ChunkManagerStub:
    """Minimal ChunkManager stand-in surfacing layout + _persistent_ids + _params_by_id."""

    def __init__(self, model: nn.Module, n_persist: int, n_chunk: int) -> None:
        from typing import cast as _cast

        from axolotl.integrations.protrain.types import (
            BlockId as _BlockId,
            ParamId as _ParamId,
        )

        # Round-robin params into n_chunk synthetic chunks so the predicate
        # sees a multi-chunk layout but install_hooks' LoRA container lookup
        # can still resolve every parameter to a chunk id.
        param_names = [_cast(_ParamId, name) for name, _ in model.named_parameters()]
        param_to_chunk = {
            name: _cast(ChunkId, i % max(1, n_chunk))
            for i, name in enumerate(param_names)
        }
        block_to_chunks: dict = {}
        for name in param_names:
            if not name.startswith("layers."):
                continue
            bid = _cast(_BlockId, int(name.split(".")[1]))
            block_to_chunks.setdefault(bid, set()).add(param_to_chunk[name])
        block_to_chunks = {b: tuple(sorted(cs)) for b, cs in block_to_chunks.items()}

        self.layout = _LayoutStub(n_chunk, param_to_chunk, block_to_chunks)
        self._params_by_id = {
            _cast(_ParamId, name): p for name, p in model.named_parameters()
        }
        self._persistent_ids = {_cast(ChunkId, i) for i in range(n_persist)}


# ---------------------------------------------------------------------------
# Predicate-only tests (no model, no hooks).
# ---------------------------------------------------------------------------


def test_is_runtime_inert_true_when_all_persistent_none_and_ckpt():
    """Predicate fires when n_persist == N_chunk, no OFFLOAD blocks, modes in {NONE, CKPT}."""
    blocks: list[nn.Module] = [nn.Linear(2, 2), nn.Linear(2, 2)]
    block_map = {BlockId(0): BlockMode.NONE, BlockId(1): BlockMode.CKPT}
    assert _is_runtime_inert(blocks, block_map, n_persist=8, N_chunk=8) is True


def test_is_runtime_inert_false_when_n_persist_lt_N_chunk():
    """One non-persistent chunk means hooks must run."""
    blocks: list[nn.Module] = [nn.Linear(2, 2)]
    block_map = {BlockId(0): BlockMode.NONE}
    assert _is_runtime_inert(blocks, block_map, n_persist=7, N_chunk=8) is False


def test_is_runtime_inert_false_when_offloaded_block_present():
    """OffloadedBlock instance in discovered blocks vetoes the skip."""
    from axolotl.integrations.protrain.block.offload import OffloadedBlock

    inner = nn.Linear(2, 2)
    blocks: list[nn.Module] = [inner, OffloadedBlock(inner)]
    block_map = {BlockId(0): BlockMode.NONE, BlockId(1): BlockMode.OFFLOAD}
    assert _is_runtime_inert(blocks, block_map, n_persist=8, N_chunk=8) is False


def test_is_runtime_inert_false_when_swap_mode_present():
    """SWAP mode means chunks actually move; predicate must veto even with n_persist == N_chunk."""
    blocks: list[nn.Module] = [nn.Linear(2, 2)]
    block_map = {BlockId(0): BlockMode.SWAP}
    assert _is_runtime_inert(blocks, block_map, n_persist=8, N_chunk=8) is False


# ---------------------------------------------------------------------------
# install_hooks + Scheduler integration: inert config skips install entirely.
# ---------------------------------------------------------------------------


def test_install_hooks_returns_empty_when_inert_and_fwd_bwd_succeed():
    """Mode-A all-persistent config: install_hooks returns [] and forward/backward still works."""
    torch.manual_seed(11)
    n_blocks = 2
    model = _TinyModel(n_blocks=n_blocks, dim=8)

    cm = _ChunkManagerStub(model, n_persist=8, n_chunk=8)
    sched = _SchedulerStub()

    block_map = {BlockId(i): BlockMode.NONE for i in range(n_blocks)}

    handles = install_hooks(
        model=model,
        chunk_manager=cm,  # type: ignore[arg-type]
        block_map=block_map,
        scheduler=sched,  # type: ignore[arg-type]
    )
    try:
        assert handles == [], (
            f"inert config must skip hook install; got {len(handles)} handles"
        )
        assert sched._is_inert is True, "scheduler._is_inert flag must be set"

        # Forward + backward must still succeed: hooks are pure overhead
        # in the inert config, removing them cannot break execution.
        x = torch.randn(2, 8, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert not sched.calls, (
            "no per-step scheduler calls should fire when hooks are skipped; "
            f"got {sched.calls}"
        )
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


def test_install_hooks_installs_when_not_inert_regression():
    """Regression guard: n_persist < N_chunk must keep the hook quartet installed."""
    torch.manual_seed(12)
    n_blocks = 2
    model = _TinyModel(n_blocks=n_blocks, dim=8)

    # n_persist (7) strictly less than N_chunk (8) means not inert.
    cm = _ChunkManagerStub(model, n_persist=7, n_chunk=8)
    sched = _SchedulerStub()

    block_map = {BlockId(i): BlockMode.NONE for i in range(n_blocks)}

    handles = install_hooks(
        model=model,
        chunk_manager=cm,  # type: ignore[arg-type]
        block_map=block_map,
        scheduler=sched,  # type: ignore[arg-type]
    )
    try:
        assert handles, "non-inert config must install hooks"
        assert sched._is_inert is False, (
            "scheduler._is_inert must remain False on non-inert install"
        )
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()


def test_scheduler_is_inert_short_circuits_per_step_methods():
    """Setting Scheduler._is_inert=True must make per-step methods no-ops."""
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler
    from axolotl.integrations.protrain.types import ChunkLayout

    layout = ChunkLayout(
        S_chunk=1024,
        N_chunk=1,
        chunks=((),),
        param_to_chunk={},
        block_to_chunks={BlockId(0): (ChunkId(0),)},
    )
    block_map = {BlockId(0): BlockMode.NONE}

    class _CM:
        def __init__(self) -> None:
            self.gather_calls = 0
            self.offload_calls = 0
            self.buffer_pool = None

        def gather(self, cid) -> None:
            self.gather_calls += 1

        def offload(self, cid) -> None:
            self.offload_calls += 1

        def reduce_grads_and_offload(self, cid) -> None:
            pass

    cm = _CM()
    sched = Scheduler(
        chunk_manager=cm,  # type: ignore[arg-type]
        block_map=block_map,
        layout=layout,
        effective_h2d_bps=1.0,
        effective_d2h_bps=1.0,
    )
    sched._is_inert = True

    sched.pre_block_forward(BlockId(0))
    sched.post_block_forward(BlockId(0))
    sched.pre_block_backward(BlockId(0))
    sched.post_block_backward(BlockId(0))
    sched.ensure_block_resident(BlockId(0))
    sched.ensure_chunks_resident((ChunkId(0),))

    assert cm.gather_calls == 0, (
        f"inert scheduler must not call gather; got {cm.gather_calls}"
    )
    assert cm.offload_calls == 0, (
        f"inert scheduler must not call offload; got {cm.offload_calls}"
    )
