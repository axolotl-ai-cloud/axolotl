"""Optimizer-boundary regression tests for ProTrain offload paths."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from axolotl.integrations.protrain.api.optim_wrapper import _ProTrainOptimizer
from axolotl.integrations.protrain.chunk.manager import ChunkManager, _CpuParamSlot
from axolotl.integrations.protrain.runtime.scheduler import Scheduler
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    ChunkId,
    ChunkLayout,
    ParamId,
)


class _FakeCpuOptim:
    def __init__(self) -> None:
        self.step_calls: list[ChunkId] = []

    def step_async(self, chunk_id, *, d2h_event=None, post_step=None):  # noqa: ARG002
        self.step_calls.append(chunk_id)


def _bare_manager() -> ChunkManager:
    mgr = ChunkManager.__new__(ChunkManager)
    mgr._chunk_shards = {}
    mgr._grad_remaining = {}
    mgr._grad_initial = {}
    mgr._cpu_step_ready_chunks = set()
    mgr._cpu_step_events = {}
    mgr._cpu_step_post_steps = {}
    mgr._persistent_grads_synced = set()
    mgr.skip_internal_grad_reduce = True
    mgr.cpu_optim = cast(Any, _FakeCpuOptim())
    mgr.device = torch.device("cpu")
    return mgr


def _cpu_optim(mgr: ChunkManager) -> _FakeCpuOptim:
    return cast(_FakeCpuOptim, mgr.cpu_optim)


def _optim_defaults() -> dict[str, Any]:
    return {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}


def _slot(name: str = "w") -> _CpuParamSlot:
    return _CpuParamSlot(
        param_id=ParamId(name),
        cpu_data=torch.zeros(2),
        cpu_grad=torch.zeros(2),
        shape=torch.Size([2]),
        dtype=torch.float32,
        byte_offset=0,
        numel=2,
        element_size=4,
    )


def test_replicated_grad_hook_defers_cpu_step_until_optimizer_boundary() -> None:
    cid = ChunkId(0)
    slot = _slot()
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.full_like(param, 3.0)

    mgr = _bare_manager()
    mgr._cpu_slots = {cid: [slot]}
    mgr._params_by_id = {slot.param_id: param}
    mgr._grad_remaining = {cid: 1}
    mgr._grad_initial = {cid: 1}

    hook = mgr._make_grad_offload_hook(cid, slot)
    hook(param)

    assert slot.cpu_grad is not None
    assert slot.cpu_grad.tolist() == [3.0, 3.0]
    assert param.grad is None
    assert cid in mgr._cpu_step_ready_chunks
    assert _cpu_optim(mgr).step_calls == []

    mgr.step_ready_cpu_chunks()

    assert _cpu_optim(mgr).step_calls == [cid]
    assert cid not in mgr._cpu_step_ready_chunks


def test_sharded_reduce_waits_for_all_param_hooks_before_step_ready() -> None:
    cid = ChunkId(0)
    mgr = _bare_manager()
    mgr._cpu_slots = {cid: [_slot()]}
    mgr._grad_remaining = {cid: 1}

    finalized = mgr._reduce_scatter_and_offload_shard(cid, cast(Any, object()))

    assert finalized is False
    assert cid not in mgr._cpu_step_ready_chunks
    assert _cpu_optim(mgr).step_calls == []


def test_backward_finalizer_defers_persistent_grad_sync_to_optimizer_step() -> None:
    cid = ChunkId(0)
    calls: list[ChunkId] = []
    mgr = _bare_manager()
    mgr._persistent_ids = {cid}
    mgr._chunk_shards = {}

    def _unexpected_reduce(chunk_id: ChunkId, *, force: bool = False) -> None:  # noqa: ARG001
        calls.append(chunk_id)

    cast(Any, mgr).reduce_grads_and_offload = _unexpected_reduce

    mgr.reduce_grads_and_offload_from_backward(cid)

    assert calls == []


def test_backward_finalize_for_sharded_chunk_only_releases_storage() -> None:
    cid = ChunkId(0)
    slot = _slot()
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.full_like(param, 3.0)
    offload_calls: list[ChunkId] = []

    mgr = _bare_manager()
    mgr._persistent_ids = set()
    mgr._chunk_shards = cast(Any, {cid: object()})
    mgr._cpu_slots = {cid: [slot]}
    mgr._params_by_id = {slot.param_id: param}
    cast(Any, mgr).offload = offload_calls.append

    mgr.reduce_grads_and_offload_from_backward(cid)

    assert offload_calls == [cid]
    assert param.grad is not None
    assert cid not in mgr._cpu_step_ready_chunks
    assert _cpu_optim(mgr).step_calls == []


def test_force_sweep_offloads_sharded_chunk_without_grads() -> None:
    cid = ChunkId(0)
    slot = _slot()
    param = torch.nn.Parameter(torch.ones(2))
    offload_calls: list[ChunkId] = []

    mgr = _bare_manager()
    mgr._persistent_ids = set()
    mgr._chunk_shards = cast(Any, {cid: object()})
    mgr._cpu_slots = {cid: [slot]}
    mgr._params_by_id = {slot.param_id: param}
    cast(Any, mgr).offload = offload_calls.append

    mgr.reduce_grads_and_offload(cid, force=True)

    assert offload_calls == [cid]
    assert cid not in mgr._cpu_step_ready_chunks
    assert _cpu_optim(mgr).step_calls == []


def test_optimizer_clips_replicated_offloaded_grads_before_cpu_step() -> None:
    cid = ChunkId(0)
    slot = _slot()
    assert slot.cpu_grad is not None
    slot.cpu_grad.copy_(torch.tensor([3.0, 4.0]))
    grad_seen_by_step: list[torch.Tensor] = []

    class _Event:
        def __init__(self) -> None:
            self.synchronized = False

        def synchronize(self) -> None:
            self.synchronized = True

    event = _Event()

    class _Manager:
        def __init__(self) -> None:
            self._cpu_step_ready_chunks = {cid}
            self._cpu_step_events = {cid: event}
            self._cpu_slots = {cid: [slot]}
            self._chunk_shards: dict[ChunkId, object] = {}

        def step_ready_cpu_chunks(self) -> None:
            assert slot.cpu_grad is not None
            grad_seen_by_step.append(slot.cpu_grad.clone())
            self._cpu_step_ready_chunks.clear()

        def wait_cpu_optim_all(self) -> None:
            return None

        def reset_optimizer_step_tracking(self) -> None:
            return None

    optim = _ProTrainOptimizer(
        gpu_optim=None,
        cpu_optim=None,
        params=[torch.nn.Parameter(torch.zeros(1))],
        defaults=_optim_defaults(),
        chunk_manager=_Manager(),
        max_grad_norm=1.0,
    )

    optim.step()

    assert event.synchronized is True
    assert len(grad_seen_by_step) == 1
    assert torch.allclose(grad_seen_by_step[0], torch.tensor([0.6, 0.8]))


def test_optimizer_clips_sharded_offloaded_grads_before_cpu_step() -> None:
    cid = ChunkId(0)
    shard_param = torch.nn.Parameter(torch.zeros(2))
    shard_param.grad = torch.tensor([6.0, 8.0])
    grad_seen_by_step: list[torch.Tensor] = []

    class _Manager:
        def __init__(self) -> None:
            self._cpu_step_ready_chunks = {cid}
            self._cpu_step_events: dict[ChunkId, object] = {}
            self._cpu_slots: dict[ChunkId, list[_CpuParamSlot]] = {}
            self._chunk_shards = {
                cid: SimpleNamespace(regions=[SimpleNamespace(shard_param=shard_param)])
            }

        def step_ready_cpu_chunks(self) -> None:
            assert shard_param.grad is not None
            grad_seen_by_step.append(shard_param.grad.clone())
            self._cpu_step_ready_chunks.clear()

        def wait_cpu_optim_all(self) -> None:
            return None

        def reset_optimizer_step_tracking(self) -> None:
            return None

    optim = _ProTrainOptimizer(
        gpu_optim=None,
        cpu_optim=None,
        params=[torch.nn.Parameter(torch.zeros(1))],
        defaults=_optim_defaults(),
        chunk_manager=_Manager(),
        max_grad_norm=1.0,
    )

    optim.step()

    assert len(grad_seen_by_step) == 1
    assert torch.allclose(grad_seen_by_step[0], torch.tensor([0.6, 0.8]))


def test_optimizer_clips_gpu_and_hidden_cpu_targets_together() -> None:
    cid = ChunkId(0)
    slot = _slot()
    assert slot.cpu_grad is not None
    slot.cpu_grad.copy_(torch.tensor([3.0, 4.0]))
    gpu_param = torch.nn.Parameter(torch.zeros(2))
    gpu_param.grad = torch.tensor([12.0, 16.0])
    grad_seen_by_cpu_step: list[torch.Tensor] = []

    class _Gpu:
        underlying = SimpleNamespace(param_groups=[{"params": [gpu_param]}])

        def step(self) -> None:
            return None

    class _Manager:
        def __init__(self) -> None:
            self._cpu_step_ready_chunks = {cid}
            self._cpu_step_events: dict[ChunkId, object] = {}
            self._cpu_slots = {cid: [slot]}
            self._chunk_shards: dict[ChunkId, object] = {}

        def step_ready_cpu_chunks(self) -> None:
            assert slot.cpu_grad is not None
            grad_seen_by_cpu_step.append(slot.cpu_grad.clone())
            self._cpu_step_ready_chunks.clear()

        def wait_cpu_optim_all(self) -> None:
            return None

        def reset_optimizer_step_tracking(self) -> None:
            return None

    optim = _ProTrainOptimizer(
        gpu_optim=cast(Any, _Gpu()),
        cpu_optim=None,
        params=[gpu_param],
        defaults=_optim_defaults(),
        chunk_manager=_Manager(),
        max_grad_norm=1.0,
    )

    optim.step()

    assert len(grad_seen_by_cpu_step) == 1
    assert gpu_param.grad is not None
    combined_norm = torch.linalg.vector_norm(
        torch.cat([grad_seen_by_cpu_step[0], gpu_param.grad])
    )
    assert combined_norm.item() == pytest.approx(1.0, rel=1e-6)


def test_optimizer_rejects_nonfinite_hidden_cpu_grad_before_step() -> None:
    cid = ChunkId(0)
    slot = _slot()
    assert slot.cpu_grad is not None
    slot.cpu_grad.copy_(torch.tensor([float("nan"), 1.0]))
    step_called = False

    class _Manager:
        def __init__(self) -> None:
            self._cpu_step_ready_chunks = {cid}
            self._cpu_step_events: dict[ChunkId, object] = {}
            self._cpu_slots = {cid: [slot]}
            self._chunk_shards: dict[ChunkId, object] = {}

        def step_ready_cpu_chunks(self) -> None:
            nonlocal step_called
            step_called = True

        def wait_cpu_optim_all(self) -> None:
            return None

    optim = _ProTrainOptimizer(
        gpu_optim=None,
        cpu_optim=None,
        params=[torch.nn.Parameter(torch.zeros(1))],
        defaults=_optim_defaults(),
        chunk_manager=_Manager(),
        max_grad_norm=1.0,
    )

    with pytest.raises(RuntimeError, match="non-finite gradient norm"):
        optim.step()
    assert step_called is False


def test_scheduler_uses_backward_finalize_hook_when_available() -> None:
    class _Manager:
        buffer_pool = None

        def __init__(self) -> None:
            self.calls: list[tuple[str, ChunkId]] = []

        def reduce_grads_and_offload(self, chunk_id: ChunkId) -> None:
            self.calls.append(("legacy", chunk_id))

        def reduce_grads_and_offload_from_backward(self, chunk_id: ChunkId) -> None:
            self.calls.append(("backward", chunk_id))

    layout = ChunkLayout(
        S_chunk=1 << 12,
        N_chunk=1,
        chunks=((ParamId("p"),),),
        param_to_chunk={ParamId("p"): ChunkId(0)},
        block_to_chunks={BlockId(0): (ChunkId(0),)},
    )
    mgr = _Manager()
    scheduler = Scheduler(
        chunk_manager=cast("object", mgr),  # type: ignore[arg-type]
        block_map={BlockId(0): BlockMode.NONE},
        layout=layout,
        effective_h2d_bps=1.0,
        effective_d2h_bps=1.0,
    )

    scheduler.post_block_backward(BlockId(0))

    assert mgr.calls == [("backward", ChunkId(0))]


def test_persistent_step_sync_skips_already_reduced_chunks(monkeypatch) -> None:
    import torch.distributed as dist

    cid = ChunkId(0)
    pid = ParamId("tail")
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones_like(param.data)
    layout = ChunkLayout(
        S_chunk=1 << 12,
        N_chunk=1,
        chunks=((pid,),),
        param_to_chunk={pid: cid},
        block_to_chunks={},
        mandatory_persistent=frozenset({cid}),
    )

    mgr = _bare_manager()
    mgr._persistent_ids = {cid}
    mgr.skip_internal_grad_reduce = False
    mgr.layout = layout
    mgr._params_by_id = {pid: param}

    calls = 0

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def _fake_all_reduce(tensor, op=None):  # noqa: ANN001, ARG001
        nonlocal calls
        calls += 1
        tensor.fill_(float(calls))

    monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)

    mgr.sync_persistent_grads_for_step()
    mgr.sync_persistent_grads_for_step()

    assert calls == 1
    assert torch.equal(param.grad, torch.ones_like(param.grad))

    mgr.reset_optimizer_step_tracking()
    mgr.sync_persistent_grads_for_step()

    assert calls == 2
    assert torch.equal(param.grad, torch.full_like(param.grad, 2.0))
