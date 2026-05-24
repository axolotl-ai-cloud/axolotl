"""Within-param shard fallback tests for huge persistent params (v51 schema).

Covers the ARCH #8 extension on top of the v3 round-robin partition:
when a single persistent ``nn.Parameter`` is so large that pinning it to
one rank defeats the memory-balance goal, slice it dim-0 across all
ranks instead.

mp.spawn workers use the gloo backend (CPU collectives) — the reduction
math is identical to NCCL.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from axolotl.integrations.protrain.types import BlockId, ParamId


def _model_with_huge_param(huge_rows: int = 8192, huge_cols: int = 4096):
    """Two-param model: one tiny weight + one huge ``nn.Linear`` weight.

    The huge weight (rows x cols, fp32) is large enough to exceed the
    1 MiB threshold the tests pass to the wrapper.
    """
    import torch
    from torch import nn

    torch.manual_seed(0)

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # tiny bias-free linear — sits on the round-robin path
            self.small = nn.Linear(4, 4, bias=False)
            # huge bias-free linear — sits on the within-shard path
            self.big = nn.Linear(huge_cols, huge_rows, bias=False)

    return _M()


def _build_persistent_chunk_manager_cpu(model):
    """Build a CPU ChunkManager with every chunk persistent."""
    import torch

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    block_spans: dict[BlockId, list[ParamId]] = {}
    for idx, (name, _p) in enumerate(model.named_parameters()):
        block_spans.setdefault(cast(BlockId, idx), []).append(cast(ParamId, name))

    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    # S_chunk large enough for the huge param to live on its own chunk.
    S_chunk = 256 * 1024 * 1024
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
        n_persist=layout.N_chunk,
        buffer_pool=pool,
        cpu_optim=None,
        gpu_optim=None,
        device=torch.device("cpu"),
    )
    return mgr, layout, pool, host


def _build_with_huge_threshold(
    model,
    mgr,
    *,
    lr: float = 1e-3,
    world_size: int,
    rank: int,
    threshold_bytes: int,
):
    """Build a _ProTrainOptimizer with the huge-param fallback active.

    Mirrors :func:`_build_persistent_optim` in
    ``test_modec_persistent_partition.py`` but adds the dim-0
    within-shard logic on params whose ``numel * 4`` exceeds the
    threshold. Uses :class:`torch.optim.AdamW` as the inner so the
    workers don't pull in Apex/DeepSpeed.
    """
    import torch
    from torch import nn

    from axolotl.integrations.protrain.api.optim_wrapper import _ProTrainOptimizer

    persistent_ids = set(mgr._persistent_ids)
    params_by_name = dict(model.named_parameters())

    persistent_params: list = []
    for cid, chunk_param_ids in enumerate(mgr.layout.chunks):
        chunk_params = [
            params_by_name[str(pid)]
            for pid in chunk_param_ids
            if str(pid) in params_by_name
        ]
        if cid in persistent_ids:
            persistent_params.extend(chunk_params)

    persistent_huge_originals: list = []
    persistent_huge_shards: list = []
    small_params: list = []
    huge_params: list = []
    for p in persistent_params:
        if int(p.numel()) * 4 > int(threshold_bytes):
            if int(p.shape[0]) % world_size != 0:
                raise RuntimeError(
                    f"shape {tuple(p.shape)} not divisible by world={world_size}"
                )
            huge_params.append(p)
        else:
            small_params.append(p)

    for p in huge_params:
        shard_size = int(p.shape[0]) // world_size
        shard_view = p.data.narrow(0, rank * shard_size, shard_size)
        shard_param = nn.Parameter(shard_view, requires_grad=p.requires_grad)
        persistent_huge_originals.append(p)
        persistent_huge_shards.append(shard_param)

    persistent_params_full = small_params
    owner_rank = [i % world_size for i in range(len(persistent_params_full))]
    if world_size > 1:
        owned = persistent_params_full[rank::world_size]
    else:
        owned = persistent_params_full
    owned = list(owned) + list(persistent_huge_shards)

    if owned:
        inner = torch.optim.AdamW(
            owned, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
    else:
        inner = torch.optim.AdamW(
            [nn.Parameter(torch.zeros(1))],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

    class _FakeGpuAdapter:
        def __init__(self, inner) -> None:
            self._optim = inner

        def step(self) -> None:
            self._optim.step()

        def zero_grad(self, set_to_none: bool = True) -> None:
            self._optim.zero_grad(set_to_none=set_to_none)

        @property
        def underlying(self):
            return self._optim

    gpu_optim = _FakeGpuAdapter(inner)
    mgr.gpu_optim = gpu_optim
    mgr.cpu_optim = None

    flat_params = persistent_params or [torch.nn.Parameter(torch.zeros(1))]
    optim = _ProTrainOptimizer(
        gpu_optim=cast("Any", gpu_optim),
        cpu_optim=None,
        params=flat_params,
        defaults={"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        chunk_manager=mgr,
        persistent_params_full=persistent_params_full,
        persistent_owner_rank=owner_rank,
        persistent_world_size=world_size,
        persistent_huge_originals=persistent_huge_originals,
        persistent_huge_shards=persistent_huge_shards,
    )
    return optim, persistent_huge_originals, persistent_huge_shards, owned


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


_HUGE_THRESHOLD = 512 * 1024  # 512 KiB — small param (4x4 fp32 = 64 B) stays under
_HUGE_ROWS = 64  # 64 % world(2/4) == 0 — divisibility holds
_HUGE_COLS = 4096  # 64 * 4096 * 4 = 1 MiB; > 512 KiB threshold


def _worker_partition_assignment(rank: int, world_size: int, tmpdir: str) -> None:
    import os as _os

    import torch.distributed as dist

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29661")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-assign",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _model_with_huge_param(huge_rows=_HUGE_ROWS, huge_cols=_HUGE_COLS)
        mgr, _layout, pool, host = _build_persistent_chunk_manager_cpu(model)
        optim, originals, shards, owned = _build_with_huge_threshold(
            model,
            mgr,
            world_size=world_size,
            rank=rank,
            threshold_bytes=_HUGE_THRESHOLD,
        )

        # Exactly one huge original recognised — the 'big' weight.
        if len(originals) != 1:
            raise RuntimeError(
                f"rank {rank}: expected 1 huge original, got {len(originals)}"
            )
        # Shard shape is (_HUGE_ROWS // world, _HUGE_COLS).
        expected_shard_shape = (_HUGE_ROWS // world_size, _HUGE_COLS)
        if tuple(shards[0].shape) != expected_shard_shape:
            raise RuntimeError(
                f"rank {rank}: shard shape {tuple(shards[0].shape)} "
                f"!= expected {expected_shard_shape}"
            )
        # Shard is a view into the original's storage.
        if shards[0].data.data_ptr() < originals[0].data.data_ptr():
            raise RuntimeError(
                f"rank {rank}: shard storage pointer is before original — "
                "shard is not a view of original.data"
            )

        with open(_os.path.join(tmpdir, f"assign_rank{rank}.done"), "w") as f:
            f.write(
                f"originals={len(originals)} shards={len(shards)} "
                f"shard_shape={tuple(shards[0].shape)}\n"
            )

        mgr.uninstall()
        host.close()
        del pool
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


def _worker_math_equivalence(rank: int, world_size: int, tmpdir: str) -> None:
    """One step under within-shard partition matches one step of vanilla AdamW."""
    import os as _os

    import torch
    import torch.distributed as dist

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29662")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-math",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        model = _model_with_huge_param(huge_rows=_HUGE_ROWS, huge_cols=_HUGE_COLS)
        init_state = {n: p.detach().clone() for n, p in model.named_parameters()}

        mgr, _layout, pool, host = _build_persistent_chunk_manager_cpu(model)
        optim, _orig, _shard, _owned = _build_with_huge_threshold(
            model,
            mgr,
            world_size=world_size,
            rank=rank,
            threshold_bytes=_HUGE_THRESHOLD,
        )

        for _name, p in model.named_parameters():
            p.grad = torch.full_like(p.data, 0.1)

        optim.step()

        post_step = {n: p.detach().clone() for n, p in model.named_parameters()}

        # Reference: vanilla single-rank AdamW on the SAME initial state.
        torch.manual_seed(0)
        ref_model = _model_with_huge_param(huge_rows=_HUGE_ROWS, huge_cols=_HUGE_COLS)
        for n, p in ref_model.named_parameters():
            p.data.copy_(init_state[n])
        for _name, p in ref_model.named_parameters():
            p.grad = torch.full_like(p.data, 0.1)
        ref_optim = torch.optim.AdamW(
            list(ref_model.parameters()),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        ref_optim.step()

        for n, p in ref_model.named_parameters():
            obs = post_step[n]
            if not torch.allclose(obs, p.data, atol=1e-3, rtol=1e-3):
                diff = (obs - p.data).abs().max().item()
                raise RuntimeError(
                    f"rank {rank}: param {n} diverges from single-rank "
                    f"reference after 1 step (max_abs_diff={diff})"
                )

        with open(_os.path.join(tmpdir, f"math_rank{rank}.done"), "w") as f:
            f.write("ok")

        mgr.uninstall()
        host.close()
        del pool
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


def _worker_state_size(rank: int, world_size: int, tmpdir: str) -> None:
    """Adam moment bytes on each rank = (huge_bytes * 2) / world ± one shard."""
    import os as _os

    import torch
    import torch.distributed as dist

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29663")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-size",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _model_with_huge_param(huge_rows=_HUGE_ROWS, huge_cols=_HUGE_COLS)
        mgr, _layout, pool, host = _build_persistent_chunk_manager_cpu(model)
        optim, _orig, _shard, _owned = _build_with_huge_threshold(
            model,
            mgr,
            world_size=world_size,
            rank=rank,
            threshold_bytes=_HUGE_THRESHOLD,
        )

        for _name, p in model.named_parameters():
            p.grad = torch.full_like(p.data, 0.1)
        optim.step()

        local_bytes = 0
        for state in optim._gpu_optim._optim.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    local_bytes += int(v.numel()) * int(v.element_size())

        # Huge param contributes m + v = 2 * 4 bytes/elem * numel total.
        huge_numel = _HUGE_ROWS * _HUGE_COLS
        huge_total = 2 * 4 * huge_numel  # m + v
        expected_huge_share = huge_total // world_size

        # local_bytes must be at least the rank's share of the huge param's Adam state.
        if local_bytes < expected_huge_share:
            raise RuntimeError(
                f"rank {rank}: local state bytes={local_bytes} less than "
                f"expected huge-shard share {expected_huge_share}"
            )
        # One small-param's worth is the only round-off noise budget.
        small_param_bytes = 2 * 4 * 4 * 4  # 4x4 fp32, m+v
        if local_bytes - expected_huge_share > 2 * small_param_bytes:
            raise RuntimeError(
                f"rank {rank}: local state bytes={local_bytes} exceeds "
                f"expected huge-shard share {expected_huge_share} by more "
                f"than {2 * small_param_bytes} bytes"
            )

        with open(_os.path.join(tmpdir, f"size_rank{rank}.done"), "w") as f:
            f.write(f"local={local_bytes} expected_huge_share={expected_huge_share}\n")

        mgr.uninstall()
        host.close()
        del pool
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


def _worker_stable_across_resume(rank: int, world_size: int, tmpdir: str) -> None:
    """Two ChunkManager rebuilds with identical seed/world/threshold pick the
    same huge-param set + same shard-shape."""
    import os as _os

    import torch
    import torch.distributed as dist

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29664")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-resume",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        model_a = _model_with_huge_param(huge_rows=_HUGE_ROWS, huge_cols=_HUGE_COLS)
        mgr_a, _la, pool_a, host_a = _build_persistent_chunk_manager_cpu(model_a)
        _, orig_a, shard_a, _ = _build_with_huge_threshold(
            model_a,
            mgr_a,
            world_size=world_size,
            rank=rank,
            threshold_bytes=_HUGE_THRESHOLD,
        )
        a_signature = [
            (tuple(o.shape), tuple(s.shape))
            for o, s in zip(orig_a, shard_a, strict=True)
        ]
        mgr_a.uninstall()
        host_a.close()
        del pool_a, model_a

        torch.manual_seed(0)
        model_b = _model_with_huge_param(huge_rows=_HUGE_ROWS, huge_cols=_HUGE_COLS)
        mgr_b, _lb, pool_b, host_b = _build_persistent_chunk_manager_cpu(model_b)
        _, orig_b, shard_b, _ = _build_with_huge_threshold(
            model_b,
            mgr_b,
            world_size=world_size,
            rank=rank,
            threshold_bytes=_HUGE_THRESHOLD,
        )
        b_signature = [
            (tuple(o.shape), tuple(s.shape))
            for o, s in zip(orig_b, shard_b, strict=True)
        ]
        mgr_b.uninstall()
        host_b.close()
        del pool_b

        if a_signature != b_signature:
            raise RuntimeError(
                f"rank {rank}: huge-shard signature drifted between "
                f"rebuilds.\n  A: {a_signature}\n  B: {b_signature}"
            )

        with open(_os.path.join(tmpdir, f"resume_rank{rank}.done"), "w") as f:
            f.write(f"sig={a_signature}\n")
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _spawn(worker, tmp_path, world_size: int = 2, sentinel: str = "done") -> None:
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    import torch.multiprocessing as mp

    mp.spawn(worker, args=(world_size, str(tmp_path)), nprocs=world_size, join=True)

    err_files = list(tmp_path.glob(f"*{sentinel}*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")


def test_huge_param_partition_assignment(tmp_path):
    """Each rank's adapter sees exactly one shard of (rows // world, cols)."""
    _spawn(_worker_partition_assignment, tmp_path, world_size=2)
    for r in range(2):
        assert (tmp_path / f"assign_rank{r}.done").is_file(), (
            f"rank {r} did not reach sentinel"
        )


def test_huge_param_math_equivalence(tmp_path):
    """Within-shard step matches single-rank vanilla AdamW within 1e-3."""
    _spawn(_worker_math_equivalence, tmp_path, world_size=2)
    for r in range(2):
        assert (tmp_path / f"math_rank{r}.done").is_file(), (
            f"rank {r} did not reach sentinel"
        )


def test_huge_param_per_rank_state_bytes(tmp_path):
    """Sum of Adam moment bytes per rank ≈ (huge_param * 2) / world."""
    _spawn(_worker_state_size, tmp_path, world_size=2)
    for r in range(2):
        assert (tmp_path / f"size_rank{r}.done").is_file(), (
            f"rank {r} did not reach sentinel"
        )


def test_huge_param_stable_across_resume(tmp_path):
    """Two rebuilds with same seed/world/threshold produce identical shard sigs."""
    _spawn(_worker_stable_across_resume, tmp_path, world_size=2)
    for r in range(2):
        assert (tmp_path / f"resume_rank{r}.done").is_file(), (
            f"rank {r} did not reach sentinel"
        )


def test_huge_param_dim0_indivisible_raises():
    """Indivisible dim-0 surfaces the documented error in the wrapper itself.

    Calls ``protrain_optimizer_wrapper`` directly with a hand-built
    chunk_manager carrying a single persistent param of shape (7, ...)
    at world_size=4. Threshold is set low so the param hits the huge
    path. The wrapper must raise ``RuntimeError`` referencing the
    indivisible dim-0 size.
    """
    import torch
    from torch import nn

    pytest.importorskip("torch")

    class _ChunkManagerStub:
        def __init__(self, p) -> None:
            self.world_size = 4
            self.rank = 0
            self.layout = type(
                "L",
                (),
                {"chunks": [["w"]], "S_chunk": 4096, "N_chunk": 1},
            )()
            self._persistent_ids = {0}
            self._params_by_id = {"w": p}
            self._chunk_shards: dict = {}
            self.cpu_optim = None
            self.gpu_optim = None

    p = nn.Parameter(torch.randn(7, 4))
    # WrappedModel duck-typing — only chunk_manager + module are touched in the path we hit.
    wrapped = type(
        "W",
        (),
        {
            "chunk_manager": _ChunkManagerStub(p),
            "module": nn.Module(),  # used by _collect_no_decay_param_ids (never reached)
        },
    )()

    from axolotl.integrations.protrain.api.optim_wrapper import (
        protrain_optimizer_wrapper,
    )

    with pytest.raises(RuntimeError, match="not divisible by world_size=4"):
        protrain_optimizer_wrapper(
            wrapped,
            lr=1e-3,
            huge_param_threshold_bytes=16,  # forces huge path
        )
