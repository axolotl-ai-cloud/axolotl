"""Round-robin persistent-param partition tests (v3 schema).

Covers the post-rejection v2 plan for arch #8: every rank only steps
its slice ``persistent_params_full[rank::world]`` of the persistent set,
then ``_sync_persistent_params_after_step`` broadcasts each owner's
update back to peers via an all-reduce(SUM)-over-zeros.

mp.spawn workers use the gloo backend (CPU collectives) so the tests
don't require multi-GPU rigs. The reduction math is identical for the
gloo and NCCL backends, so single-host gloo coverage is sufficient for
correctness assertions.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from axolotl.integrations.protrain.types import BlockId, ParamId


def _tiny_cpu_model(n_params: int = 8, hidden: int = 4):
    """Tiny model with ``n_params`` independently-shaped linear weights.

    Each weight is small enough that ``persistent_params_full`` has
    enough entries for round-robin to do something visible at world=2.
    """
    import torch
    from torch import nn

    torch.manual_seed(0)
    layers = []
    for _ in range(n_params):
        layers.append(nn.Linear(hidden, hidden, bias=False))
    model = nn.Module()
    model.h = nn.ModuleList(layers)  # type: ignore[attr-defined]
    return model


def _build_persistent_chunk_manager_cpu(model):
    """Build a CPU ChunkManager with every chunk persistent."""
    import torch

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    block_spans: dict[BlockId, list[ParamId]] = {}
    for idx, (name, _) in enumerate(model.named_parameters()):
        block_spans.setdefault(cast(BlockId, idx), []).append(cast(ParamId, name))

    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    S_chunk = 1 << 12  # small chunk size so each param lands in its own chunk
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


def _build_persistent_optim(
    model,
    mgr,
    *,
    lr: float = 1e-3,
    world_size: int = 1,
    rank: int = 0,
):
    """Build a _ProTrainOptimizer with only persistent params, mirroring optim_wrapper logic.

    Uses :class:`torch.optim.AdamW` as the inner optimizer (avoids the
    Apex / DeepSpeed dependency in the gloo CPU workers).
    """
    import torch

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

    persistent_params_full = list(persistent_params)
    owner_rank = [i % world_size for i in range(len(persistent_params_full))]
    if world_size > 1:
        owned = persistent_params_full[rank::world_size]
    else:
        owned = persistent_params_full

    inner = torch.optim.AdamW(
        owned, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
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

    gpu_optim = _FakeGpuAdapter(inner) if owned else None
    mgr.gpu_optim = gpu_optim
    mgr.cpu_optim = None

    optim = _ProTrainOptimizer(
        gpu_optim=cast("Any", gpu_optim),
        cpu_optim=None,
        params=persistent_params_full or [torch.nn.Parameter(torch.zeros(1))],
        defaults={"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        chunk_manager=mgr,
        persistent_params_full=persistent_params_full,
        persistent_owner_rank=owner_rank,
        persistent_world_size=world_size,
    )
    return optim, persistent_params_full, owner_rank


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def _worker_assignment_stable(rank: int, world_size: int, tmpdir: str) -> None:
    import os as _os

    import torch.distributed as dist

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29551")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-assign",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _tiny_cpu_model(n_params=8)
        mgr, layout, pool, host = _build_persistent_chunk_manager_cpu(model)
        optim, persistent_full, owner_rank = _build_persistent_optim(
            model, mgr, world_size=world_size, rank=rank
        )

        # Owned subset on this rank = persistent_full[rank::world]
        expected_owned = [
            persistent_full[i] for i in range(rank, len(persistent_full), world_size)
        ]
        observed = optim._gpu_optim._optim.param_groups[0]["params"]
        if [id(p) for p in observed] != [id(p) for p in expected_owned]:
            raise RuntimeError(
                f"rank {rank}: owned params mismatch. "
                f"expected ids={[id(p) for p in expected_owned]} "
                f"observed ids={[id(p) for p in observed]}"
            )

        # Sentinel file for the driver.
        with open(_os.path.join(tmpdir, f"assign_rank{rank}.done"), "w") as f:
            f.write(
                f"owned={len(observed)} total={len(persistent_full)} "
                f"owner_rank={owner_rank}\n"
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
    """Each rank steps the partitioned optimizer; after sync every rank's
    full param tensors must match a single-rank reference within tol."""
    import os as _os

    import torch
    import torch.distributed as dist

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29552")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-math",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)
        model = _tiny_cpu_model(n_params=8)
        # Snapshot pre-step params for the reference.
        init_state = {n: p.detach().clone() for n, p in model.named_parameters()}

        mgr, layout, pool, host = _build_persistent_chunk_manager_cpu(model)
        optim, persistent_full, _ = _build_persistent_optim(
            model, mgr, world_size=world_size, rank=rank
        )

        # Plant rank-local grads; _ProTrainOptimizer.step must AVG-reduce
        # persistent grads before each rank steps its owned partition.
        local_grad = float(rank + 1)
        for _name, p in model.named_parameters():
            p.grad = torch.full_like(p.data, local_grad)

        optim.step()

        # After step + sync, every rank should see the same post-step params.
        post_step = {n: p.detach().clone() for n, p in model.named_parameters()}

        # Reference: build a fresh vanilla AdamW on the SAME initial state,
        # plant the same grads, step once.
        torch.manual_seed(0)
        ref_model = _tiny_cpu_model(n_params=8)
        for n, p in ref_model.named_parameters():
            p.data.copy_(init_state[n])
        expected_mean_grad = sum(float(r + 1) for r in range(world_size)) / float(
            world_size
        )
        for _name, p in ref_model.named_parameters():
            p.grad = torch.full_like(p.data, expected_mean_grad)
        ref_optim = torch.optim.AdamW(
            list(ref_model.parameters()),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        ref_optim.step()

        # Iter-0 tolerance: 1e-3.
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
    """Sum of state-tensor bytes on this rank should be ~total / world."""
    import os as _os

    import torch
    import torch.distributed as dist

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29553")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-size",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _tiny_cpu_model(n_params=8)
        mgr, layout, pool, host = _build_persistent_chunk_manager_cpu(model)
        optim, persistent_full, _ = _build_persistent_optim(
            model, mgr, world_size=world_size, rank=rank
        )

        # Plant grads, step once so inner state is populated.
        for _name, p in model.named_parameters():
            p.grad = torch.full_like(p.data, 0.1)
        optim.step()

        local_bytes = 0
        for state in optim._gpu_optim._optim.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    local_bytes += int(v.numel()) * int(v.element_size())

        # Reduce SUM to derive total cluster bytes.
        local_t = torch.tensor([local_bytes], dtype=torch.long)
        total_t = local_t.clone()
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        total = int(total_t.item())

        # Per-rank slice — within +/- one param's worth of bytes.
        # Largest param: 4x4 f32 = 64 bytes m + 64 bytes v = 128 bytes.
        # Adam also tracks step (scalar) per param — small noise budget.
        expected = total // world_size
        tolerance = 4 * 4 * 4 * 2  # one Linear's optim state in bytes
        if abs(local_bytes - expected) > tolerance:
            raise RuntimeError(
                f"rank {rank}: per-rank state bytes={local_bytes} differs "
                f"from total/world={expected} (total={total}, "
                f"world={world_size}) by more than tolerance={tolerance}"
            )

        with open(_os.path.join(tmpdir, f"size_rank{rank}.done"), "w") as f:
            f.write(f"local={local_bytes} expected~={expected} total={total}\n")

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
    """Two ChunkManager builds with identical seed/world produce the same
    persistent_params_full sequence (by param NAME) — so owner_rank is
    deterministic and the partition is reproducible across resumes."""
    import os as _os

    import torch.distributed as dist

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29554")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-resume",
        rank=rank,
        world_size=world_size,
    )
    try:
        import torch

        torch.manual_seed(0)
        model_a = _tiny_cpu_model(n_params=8)
        mgr_a, layout_a, pool_a, host_a = _build_persistent_chunk_manager_cpu(model_a)
        names_a = [n for n, _ in model_a.named_parameters()]
        persistent_ids_a = set(mgr_a._persistent_ids)
        persistent_names_a: list[str] = []
        for cid, chunk_pids in enumerate(mgr_a.layout.chunks):
            if cid in persistent_ids_a:
                for pid in chunk_pids:
                    if str(pid) in names_a:
                        persistent_names_a.append(str(pid))

        mgr_a.uninstall()
        host_a.close()
        del pool_a, model_a

        torch.manual_seed(0)
        model_b = _tiny_cpu_model(n_params=8)
        mgr_b, layout_b, pool_b, host_b = _build_persistent_chunk_manager_cpu(model_b)
        names_b = [n for n, _ in model_b.named_parameters()]
        persistent_ids_b = set(mgr_b._persistent_ids)
        persistent_names_b: list[str] = []
        for cid, chunk_pids in enumerate(mgr_b.layout.chunks):
            if cid in persistent_ids_b:
                for pid in chunk_pids:
                    if str(pid) in names_b:
                        persistent_names_b.append(str(pid))

        mgr_b.uninstall()
        host_b.close()
        del pool_b

        if persistent_names_a != persistent_names_b:
            raise RuntimeError(
                f"rank {rank}: persistent name sequence differs between "
                f"ChunkManager rebuilds; partition would shift on resume.\n"
                f"  A: {persistent_names_a}\n  B: {persistent_names_b}"
            )

        # Owner-rank sequence (by name) must also be stable.
        owner_a = [(n, i % world_size) for i, n in enumerate(persistent_names_a)]
        owner_b = [(n, i % world_size) for i, n in enumerate(persistent_names_b)]
        if owner_a != owner_b:
            raise RuntimeError(
                f"rank {rank}: owner_rank assignment drifted between "
                f"rebuilds.\n  A: {owner_a}\n  B: {owner_b}"
            )

        with open(_os.path.join(tmpdir, f"resume_rank{rank}.done"), "w") as f:
            f.write(f"names={persistent_names_a}\n")
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


def test_persistent_partition_assignment_stable(tmp_path):
    """Each rank's inner-optim params list is the round-robin slice."""
    _spawn(_worker_assignment_stable, tmp_path, world_size=2)
    for r in range(2):
        assert (tmp_path / f"assign_rank{r}.done").is_file(), (
            f"rank {r} did not reach sentinel"
        )


def test_math_equivalence_partitioned_vs_single_rank(tmp_path):
    """One step under w=2 partition matches one step of vanilla AdamW (single-rank)."""
    _spawn(_worker_math_equivalence, tmp_path, world_size=2)
    for r in range(2):
        assert (tmp_path / f"math_rank{r}.done").is_file(), (
            f"rank {r} did not reach sentinel"
        )


def test_persistent_optim_state_size_per_rank(tmp_path):
    """Per-rank state bytes ~= total / world."""
    _spawn(_worker_state_size, tmp_path, world_size=2)
    for r in range(2):
        assert (tmp_path / f"size_rank{r}.done").is_file(), (
            f"rank {r} did not reach sentinel"
        )


def test_persistent_partition_stable_across_resume(tmp_path):
    """Two ChunkManager builds with same seed/world produce the same partition."""
    _spawn(_worker_stable_across_resume, tmp_path, world_size=2)
    for r in range(2):
        assert (tmp_path / f"resume_rank{r}.done").is_file(), (
            f"rank {r} did not reach sentinel"
        )
