"""Tests for ProTrain optimizer checkpoint/resume (CHECKPOINT_DESIGN.md Phase 1).

Covers the save/load helpers in ``api/checkpoint.py`` plus the gating,
validation, and round-trip behaviour required by the design note's
ship gate (§7).

CPU-only tests use mocked chunk managers; GPU tests share a
module-scoped chunk manager + optimizer (see :func:`saved_checkpoint`)
so we don't allocate a fresh pinned-host region per test — that
exhausts the pinned-memory budget on the test rig and crashes the
process. Tests that need their own teardown (e.g. continued-training
correctness) are explicitly marked ``slow`` so the auto-cleanup
fixture runs between them.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from typing import cast
from unittest import mock

import pytest

from axolotl.integrations.protrain.api.checkpoint import (
    CPU_OPTIM_DIRNAME,
    DEFAULT_SAVE_MAX_BYTES,
    GPU_OPTIM_FILENAME,
    METADATA_FILENAME,
    PROTRAIN_OPTIM_DIRNAME,
    SAVE_MODE_REPLICATED,
    SAVE_MODE_SHARDED,
    SCHEMA_FORMAT_VERSION,
    _build_regions_per_chunk,
    _effective_persistent_ids,
    _estimate_optim_state_bytes,
    _hash_state_dict,
    _is_protrain_optimizer,
    _is_raw_protrain_optimizer,
    _layout_signature,
    _load_protrain_optim_dir,
    _save_protrain_optim_dir,
    _unwrap_protrain_optim,
    _validate_regions_match,
    install_load_hook,
    make_checkpoint_callback,
)
from axolotl.integrations.protrain.types import BlockId, ChunkId, ParamId

# ---------------------------------------------------------------------------
# Helpers — mirror test_chunk_manager_offload.py's fixture style
# ---------------------------------------------------------------------------


def _tiny_model(hidden: int = 64, n_layers: int = 4):
    """Tiny 4-layer "transformer-ish" model identical to the offload tests'."""
    import torch
    from torch import nn

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(hidden, hidden, bias=False)
            self.h = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)]
            )
            self.head = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.embed(x)
            for layer in self.h:
                x = layer(x)
            return self.head(x)

    torch.manual_seed(0)
    return TinyTransformer()


def _build_layout_for(model, S_chunk: int):
    from axolotl.integrations.protrain.chunk.layout import build_layout

    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _ in model.named_parameters():
        if name.startswith("h."):
            idx = int(name.split(".")[1])
            block_spans.setdefault(cast(BlockId, idx), []).append(cast(ParamId, name))
    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    return build_layout(model, exec_order, S_chunk, block_spans)


def _build_chunk_manager(model, n_persist: int, S_chunk: int):
    import torch

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

    layout = _build_layout_for(model, S_chunk)
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
    return mgr, host  # keep host alive — see fixture teardown


def _build_optim_pair(model, mgr, *, lr: float = 1e-3):
    """Build the (gpu_optim, cpu_optim, _ProTrainOptimizer) triple by hand.

    Mirrors what protrain_optimizer_wrapper does, minus the wrapper's
    DeepSpeed-failure fallback path. Calling this requires
    materialize_offload to have run on the manager so the per-chunk
    shard_params (sharded mode) or the cpu_params (replicated mode)
    are reachable.
    """
    import torch

    from axolotl.integrations.protrain.api.optim_wrapper import _ProTrainOptimizer
    from axolotl.integrations.protrain.chunk import (
        CpuFusedAdamAdapter,
        GpuFusedAdamAdapter,
    )

    layout = mgr.layout
    persistent_ids = set(mgr._persistent_ids)
    params_by_name = dict(model.named_parameters())

    persistent_params: list = []
    cpu_params_per_chunk: dict = {}
    for cid, chunk_param_ids in enumerate(layout.chunks):
        chunk_params = [
            params_by_name[str(pid)]
            for pid in chunk_param_ids
            if str(pid) in params_by_name
        ]
        if cid in persistent_ids:
            persistent_params.extend(chunk_params)
        else:
            cpu_params_per_chunk[ChunkId(cid)] = chunk_params

    gpu_optim = None
    if persistent_params:
        gpu_optim = GpuFusedAdamAdapter(params=persistent_params, lr=lr)

    cpu_optim = None
    cpu_params_for_optim: dict = {}
    for cid, ps in cpu_params_per_chunk.items():
        shard_state = mgr._chunk_shards.get(cid)
        if shard_state is not None and shard_state.regions:
            cpu_params_for_optim[cid] = [r.shard_param for r in shard_state.regions]
        else:
            cpu_params_for_optim[cid] = ps

    if any(cpu_params_for_optim.values()):
        cpu_optim = CpuFusedAdamAdapter(params_per_chunk=cpu_params_for_optim, lr=lr)

    mgr.cpu_optim = cpu_optim
    mgr.gpu_optim = gpu_optim

    all_params: list = list(persistent_params)
    for ps in cpu_params_per_chunk.values():
        all_params.extend(ps)
    seen: set[int] = set()
    unique = []
    for p in all_params:
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(p)
    if not unique:
        unique = [torch.nn.Parameter(torch.zeros(1, device="cuda"))]

    optim = _ProTrainOptimizer(
        gpu_optim=gpu_optim,
        cpu_optim=cpu_optim,
        params=unique,
        defaults={"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
        chunk_manager=mgr,
    )
    return gpu_optim, cpu_optim, optim


def _step_once(model, mgr, optim, device):
    """One fwd+bwd+step cycle. Manually gathers offloaded chunks first."""
    import torch

    for cid in list(mgr._non_persistent_ids):
        mgr.gather(cid)
    optim.zero_grad()
    x = torch.randn(2, model.embed.in_features, device=device)
    out = model(x)
    out.sum().backward()
    optim.step()


def _teardown_mgr(mgr, optim) -> None:
    import torch

    try:
        mgr.restore_to_gpu()
    except Exception:
        pass
    if optim is not None and getattr(optim, "_cpu_optim", None) is not None:
        try:
            optim._cpu_optim.shutdown()
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Pure helpers (CPU only)
# ---------------------------------------------------------------------------


def test_estimate_optim_state_bytes_walks_inner_state():
    """Estimator sums tensor bytes from inner adapter state dicts.

    Walking outer optim.param_groups would miss offloaded state (the
    user-facing param.data is replaced with an empty placeholder by
    materialize_offload — manager.py:706 / :1494). The fix walks the
    inner adapters' state directly, where tensors are real.
    """
    import torch

    fake_inner_gpu = mock.MagicMock()
    fake_inner_gpu.state = {
        0: {
            "exp_avg": torch.zeros(10, dtype=torch.float32),  # 10 * 4 = 40 bytes
            "exp_avg_sq": torch.zeros(10, dtype=torch.float32),  # 40 bytes
            "step": 1,  # int — not counted
        },
    }
    fake_inner_cpu_chunk_0 = mock.MagicMock()
    fake_inner_cpu_chunk_0.state = {
        0: {
            "exp_avg": torch.zeros(20, dtype=torch.float32),  # 80 bytes
            "exp_avg_sq": torch.zeros(20, dtype=torch.float32),  # 80 bytes
        },
    }

    fake_optim = mock.MagicMock()
    fake_optim._gpu_optim = mock.MagicMock(_optim=fake_inner_gpu)
    fake_optim._cpu_optim = mock.MagicMock(_optims={0: fake_inner_cpu_chunk_0})

    # 40 + 40 + 80 + 80 = 240 bytes
    assert _estimate_optim_state_bytes(fake_optim) == 240


def test_estimate_optim_state_bytes_pre_step_returns_zero():
    """Pre-first-step the inner state is empty → estimate is 0.

    This is correct: there is no Adam state to save yet. Any save
    attempt would produce small placeholder files that legitimately
    pass the gate.
    """
    fake_inner_gpu = mock.MagicMock()
    fake_inner_gpu.state = {}
    fake_optim = mock.MagicMock()
    fake_optim._gpu_optim = mock.MagicMock(_optim=fake_inner_gpu)
    fake_optim._cpu_optim = None

    assert _estimate_optim_state_bytes(fake_optim) == 0


def test_estimate_optim_state_bytes_handles_none_adapters():
    """Both adapters absent → 0. Either present alone → counted."""
    fake_optim = mock.MagicMock()
    fake_optim._gpu_optim = None
    fake_optim._cpu_optim = None

    assert _estimate_optim_state_bytes(fake_optim) == 0


def test_estimate_optim_state_bytes_partitioned_local_shard():
    """When partition is active, GPU adapter state counts as local_shard
    (multiplied by world_size at cluster aggregation), not replicated."""
    import torch

    # Mock inner GPU optim with 100 bytes of state, on a "world=4" partition.
    fake_inner_gpu = mock.MagicMock()
    fake_inner_gpu.state = {
        0: {"exp_avg": torch.zeros(25, dtype=torch.float32)},  # 100 bytes
    }
    fake_optim = mock.MagicMock(spec=[
        "_gpu_optim",
        "_cpu_optim",
        "_persistent_world_size",
        "_persistent_params_full",
    ])
    fake_optim._gpu_optim = mock.MagicMock(_optim=fake_inner_gpu)
    fake_optim._cpu_optim = None
    fake_optim._persistent_world_size = 4
    # Non-empty list signals "partition active" path.
    fake_optim._persistent_params_full = [object(), object(), object(), object()]

    # No distributed group active → global_sharded_bytes == local_bytes.
    # Estimator returns 100 (replicated=0, local_shard=100, global=100).
    assert _estimate_optim_state_bytes(fake_optim) == 100

    # Sanity: when partition is OFF (single-rank world), the path treats
    # GPU state as replicated and still returns 100.
    fake_optim._persistent_world_size = 1
    fake_optim._persistent_params_full = []
    assert _estimate_optim_state_bytes(fake_optim) == 100


def test_layout_signature_stable_across_calls():
    fake_layout = mock.MagicMock(
        S_chunk=1024, N_chunk=3, chunks=(("a",), ("b", "c"), ("d",))
    )
    fake_mgr = mock.MagicMock(layout=fake_layout, _persistent_ids={0, 1})
    h1 = _layout_signature(fake_mgr, world_size=1, zero3_shard=False)
    h2 = _layout_signature(fake_mgr, world_size=1, zero3_shard=False)
    assert h1 == h2
    assert len(h1) == 64


def test_layout_signature_changes_with_persistent_ids():
    fake_layout = mock.MagicMock(
        S_chunk=1024, N_chunk=3, chunks=(("a",), ("b",), ("c",))
    )
    mgr_a = mock.MagicMock(layout=fake_layout, _persistent_ids={0})
    mgr_b = mock.MagicMock(layout=fake_layout, _persistent_ids={0, 1})
    assert _layout_signature(
        mgr_a, world_size=1, zero3_shard=False
    ) != _layout_signature(mgr_b, world_size=1, zero3_shard=False)


def test_layout_signature_changes_with_world_size_or_zero3():
    """Mode-aware ``world_size`` semantics:

    * Mode-B (``zero3_shard=False``, replicated): ``world_size`` is
      IGNORED — replicated state survives cross-world resume so the
      signature is rank-count-independent.
    * Mode-C (``zero3_shard=True``, sharded): ``world_size`` IS part
      of the hash — different ranks hold different shards, and
      cross-world resume requires the offline reshard tool.
    """
    fake_layout = mock.MagicMock(S_chunk=1024, N_chunk=2, chunks=(("a",), ("b",)))
    fake_mgr = mock.MagicMock(layout=fake_layout, _persistent_ids={0})
    base = _layout_signature(fake_mgr, world_size=1, zero3_shard=False)
    same_ws_replicated = _layout_signature(fake_mgr, world_size=2, zero3_shard=False)
    z3_ws1 = _layout_signature(fake_mgr, world_size=1, zero3_shard=True)
    z3_ws2 = _layout_signature(fake_mgr, world_size=2, zero3_shard=True)
    # Mode-B: world_size delta does NOT change signature (Phase-2 fix).
    assert base == same_ws_replicated
    # Mode flip changes signature.
    assert base != z3_ws1
    # Mode-C: world_size delta DOES change signature.
    assert z3_ws1 != z3_ws2


def test_effective_persistent_ids_returns_sorted_list():
    fake_mgr = mock.MagicMock(_persistent_ids={5, 1, 3, 0})
    assert _effective_persistent_ids(fake_mgr) == [0, 1, 3, 5]


def test_hash_state_dict_handles_bf16_tensor():
    """Direct ``t.numpy()`` rejects bf16 (and other torch-only dtypes);
    the hash path goes through a uint8 view so storage bytes always
    work. Regression: prior implementation crashed with
    ``TypeError: Got unsupported ScalarType BFloat16`` for any
    optimizer state holding bf16 momentum (custom optimizers, future
    mixed-precision configs)."""
    import torch

    sd = {"x": torch.zeros(2, dtype=torch.bfloat16)}
    digest = _hash_state_dict(sd)
    assert isinstance(digest, bytes)
    assert len(digest) == 32  # SHA-256

    # Different bf16 contents → different hash. Confirms the byte view
    # actually round-trips storage, not just shape/dtype.
    sd_other = {"x": torch.ones(2, dtype=torch.bfloat16)}
    assert _hash_state_dict(sd_other) != digest


def test_hash_state_dict_handles_empty_tensor():
    """Empty tensors must not break the hash path. The numpy() byte
    view path skips the body for numel()==0 to avoid edge-case behavior
    of ``view(torch.uint8)`` on zero-element storage."""
    import torch

    sd = {"x": torch.empty(0, dtype=torch.bfloat16)}
    digest = _hash_state_dict(sd)
    assert isinstance(digest, bytes)
    assert len(digest) == 32


def test_is_protrain_optimizer_duck_types():
    assert _is_protrain_optimizer(mock.MagicMock(spec=[])) is False
    has_all = mock.MagicMock(spec=["_gpu_optim", "_cpu_optim", "_chunk_manager"])
    assert _is_protrain_optimizer(has_all) is True
    assert _is_raw_protrain_optimizer(has_all) is True


def test_unwrap_protrain_optim_handles_raw_and_wrapped():
    """Without the unwrap, AcceleratedOptimizer wrapping silently
    no-ops the callback in real Trainer saves (HF replaces
    trainer.optimizer with AcceleratedOptimizer post-prepare; the raw
    ProTrain attrs are only reachable via .optimizer)."""
    raw = mock.MagicMock(spec=["_gpu_optim", "_cpu_optim", "_chunk_manager"])
    # Direct case
    assert _unwrap_protrain_optim(raw) is raw

    # Wrapped case — anything with .optimizer pointing at raw
    class _WrapperLike:
        def __init__(self, inner):
            self.optimizer = inner

    wrapper = _WrapperLike(raw)
    assert _unwrap_protrain_optim(wrapper) is raw
    assert _is_protrain_optimizer(wrapper) is True
    # Raw-only check rejects the wrapper
    assert _is_raw_protrain_optimizer(wrapper) is False

    # Non-ProTrain optimizer wrapped or otherwise: returns None
    not_protrain = mock.MagicMock(spec=[])
    assert _unwrap_protrain_optim(not_protrain) is None
    assert _unwrap_protrain_optim(_WrapperLike(not_protrain)) is None
    assert _unwrap_protrain_optim(None) is None


def test_unwrap_real_accelerated_optimizer():
    """AcceleratedOptimizer (the actual class HF Trainer wraps with) is
    correctly unwrapped. Catches the silent-no-op bug where the
    callback receives the wrapped form post-prepare and the duck-type
    check fails on the wrapper.
    """
    pytest.importorskip("accelerate")
    from accelerate import Accelerator
    from accelerate.optimizer import AcceleratedOptimizer

    # AcceleratedOptimizer.__init__ touches the accelerator state
    # singleton. Initialize one (idempotent across tests).
    Accelerator()

    raw_protrain = mock.MagicMock(
        spec=[
            "_gpu_optim",
            "_cpu_optim",
            "_chunk_manager",
            "state_dict",
            "load_state_dict",
            "param_groups",
            "state",
            "defaults",
        ]
    )
    raw_protrain.state_dict.return_value = {"state": {}, "param_groups": []}
    raw_protrain.load_state_dict.return_value = None

    wrapped = AcceleratedOptimizer(raw_protrain, device_placement=False)

    assert wrapped.optimizer is raw_protrain
    assert _unwrap_protrain_optim(wrapped) is raw_protrain


def test_save_skipped_when_estimate_exceeds_threshold(tmp_path, caplog):
    """Gate trips on the inner-state size, not outer param_groups."""
    import logging

    import torch

    fake_inner_gpu = mock.MagicMock()
    fake_inner_gpu.state = {
        0: {
            "exp_avg": torch.zeros(10**5, dtype=torch.float32),  # 400 KB
            "exp_avg_sq": torch.zeros(10**5, dtype=torch.float32),
        }
    }
    fake_optim = mock.MagicMock()
    fake_optim._gpu_optim = mock.MagicMock(_optim=fake_inner_gpu)
    fake_optim._cpu_optim = None
    fake_optim._chunk_manager = mock.MagicMock(zero3_shard=False)
    fake_optim._chunk_manager.layout = mock.MagicMock(
        S_chunk=1024, N_chunk=1, chunks=(("a",),)
    )
    fake_optim._chunk_manager._persistent_ids = {0}

    # ``axolotl.logging_config.configure_logging()`` (run at axolotl.cli
    # import time, which CI hits) sets ``propagate=False`` on the
    # ``axolotl`` logger. Pytest's ``caplog`` installs its handler at the
    # root, so non-propagating records never reach it and the assertion
    # below sees an empty ``caplog.records``. Force propagation for the
    # duration of the test (and restore on exit) so caplog deterministically
    # sees the production WARN.
    ax_logger = logging.getLogger("axolotl")
    prev_propagate = ax_logger.propagate
    ax_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            wrote = _save_protrain_optim_dir(
                fake_optim, str(tmp_path), step=1, save_max_bytes=1024
            )
    finally:
        ax_logger.propagate = prev_propagate
    assert wrote is False
    assert any(
        "skipping save" in rec.message and "exceeds" in rec.message
        for rec in caplog.records
    )
    assert not (tmp_path / PROTRAIN_OPTIM_DIRNAME).exists()


def test_save_skipped_when_offloaded_state_exceeds_threshold(tmp_path, caplog):
    """Regression for the param_groups-walking bug: offloaded state's
    user-facing params have empty .data after materialize_offload, so
    walking outer param_groups returned 0 bytes for offloaded state and
    let arbitrarily large saves through. Verify the fix counts the
    actual inner-state bytes regardless of outer placeholder shapes.
    """
    import logging

    import torch

    # Simulate the post-materialize_offload state: outer param_groups
    # have empty placeholders (would have summed to 0 under the old
    # estimator), but the inner CPU adam owns real state tensors.
    empty_placeholder = torch.nn.Parameter(torch.empty(0))
    fake_inner_cpu_chunk_0 = mock.MagicMock()
    fake_inner_cpu_chunk_0.state = {
        0: {
            "exp_avg": torch.zeros(10**5, dtype=torch.float32),  # 400 KB real
            "exp_avg_sq": torch.zeros(10**5, dtype=torch.float32),
        }
    }
    fake_optim = mock.MagicMock()
    fake_optim.param_groups = [{"params": [empty_placeholder]}]  # red herring
    fake_optim._gpu_optim = None
    fake_optim._cpu_optim = mock.MagicMock(_optims={0: fake_inner_cpu_chunk_0})
    fake_optim._chunk_manager = mock.MagicMock(zero3_shard=False)
    fake_optim._chunk_manager.layout = mock.MagicMock(
        S_chunk=1024, N_chunk=1, chunks=(("a",),)
    )
    fake_optim._chunk_manager._persistent_ids = set()

    with caplog.at_level(logging.WARNING):
        wrote = _save_protrain_optim_dir(
            fake_optim, str(tmp_path), step=1, save_max_bytes=1024
        )
    assert wrote is False, (
        "estimator must count offloaded inner state, not outer placeholders"
    )
    assert not (tmp_path / PROTRAIN_OPTIM_DIRNAME).exists()


def test_load_returns_false_when_dir_absent(tmp_path):
    fake_optim = mock.MagicMock()
    assert _load_protrain_optim_dir(fake_optim, str(tmp_path)) is False


def test_install_load_hook_wraps_trainer_method():
    fake_trainer = mock.MagicMock()
    original = mock.MagicMock()
    fake_trainer._load_optimizer_and_scheduler = original
    fake_optim = mock.MagicMock(spec=["_gpu_optim", "_cpu_optim", "_chunk_manager"])

    install_load_hook(fake_trainer, fake_optim)
    assert fake_trainer._load_optimizer_and_scheduler is not original

    fake_trainer._load_optimizer_and_scheduler(None)
    original.assert_called_once_with(None)


def test_callback_skips_when_optim_is_not_protrain(tmp_path):
    """Callback no-ops when trainer.optimizer is a vanilla torch optimizer."""
    import torch

    cb = make_checkpoint_callback(save_max_bytes=DEFAULT_SAVE_MAX_BYTES)
    fake_args = mock.MagicMock(output_dir=str(tmp_path), process_index=0, world_size=1)
    fake_state = mock.MagicMock(global_step=1)
    fake_control = mock.MagicMock()

    plain = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    cb.on_save(fake_args, fake_state, fake_control, optimizer=plain)
    assert not (tmp_path / PROTRAIN_OPTIM_DIRNAME).exists()


# ---------------------------------------------------------------------------
# GPU tests — share one chunk_manager across the validation tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def saved_checkpoint(tmp_path_factory):
    """Build mgr+optim once, do one step, save once. Module-scoped — most
    tests just inspect or mutate the saved directory + optim, no need to
    re-run the expensive setup.

    Yields ``(saved_dir, mgr, optim)``. Teardown restores the manager
    and shuts down the CPU adam thread pool.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    saved_dir = tmp_path_factory.mktemp("protrain_save")
    model = _tiny_model().to("cuda")
    mgr, host = _build_chunk_manager(model, n_persist=1, S_chunk=64 * 1024)
    mgr.materialize_offload()
    _, _, optim = _build_optim_pair(model, mgr)
    _step_once(model, mgr, optim, "cuda")

    wrote = _save_protrain_optim_dir(
        optim,
        str(saved_dir),
        step=42,
        save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
    )
    assert wrote is True

    try:
        yield saved_dir, mgr, optim
    finally:
        _teardown_mgr(mgr, optim)
        del model, optim, mgr, host
        gc.collect()


@pytest.fixture
def fresh_checkpoint_dir(tmp_path, saved_checkpoint):
    """Per-test copy of the shared saved directory. Mutation tests use this
    so they don't contaminate the module-scoped original."""
    saved_dir, _, _ = saved_checkpoint
    target = tmp_path / "protrain_checkpoint"
    shutil.copytree(saved_dir, target)
    return target


@pytest.mark.gpu
def test_save_writes_expected_directory_layout(saved_checkpoint):
    saved_dir, _, optim = saved_checkpoint
    proot = saved_dir / PROTRAIN_OPTIM_DIRNAME
    assert (proot / "metadata.json").is_file()
    assert (proot / "gpu_optim.pt").is_file()

    if optim._cpu_optim is not None and optim._cpu_optim._optims:
        cpu_dir = proot / "cpu_optim"
        assert cpu_dir.is_dir()
        files = sorted(p.name for p in cpu_dir.iterdir())
        assert all(f.startswith("chunk_") and f.endswith(".pt") for f in files)
        assert len(files) == len(optim._cpu_optim._optims)


@pytest.mark.gpu
def test_save_metadata_contains_expected_fields(saved_checkpoint):
    saved_dir, mgr, _ = saved_checkpoint
    with open(saved_dir / PROTRAIN_OPTIM_DIRNAME / "metadata.json") as f:
        meta = json.load(f)

    assert meta["format_version"] == SCHEMA_FORMAT_VERSION
    assert SCHEMA_FORMAT_VERSION == 3
    assert isinstance(meta["protrain_layout_signature"], str)
    assert len(meta["protrain_layout_signature"]) == 64
    assert meta["protrain_persistent_ids"] == sorted(
        int(x) for x in mgr._persistent_ids
    )
    assert meta["protrain_world_size"] == 1
    assert meta["protrain_zero3_shard"] is False
    # Phase 2 schema additions:
    assert meta["protrain_save_mode"] == "replicated"
    assert meta["saving_rank"] == 0
    assert meta["saved_at_step"] == 42
    assert isinstance(meta["estimated_optim_state_bytes"], int)


@pytest.mark.gpu
def test_save_drains_cpu_optim_before_snapshot(tmp_path, saved_checkpoint):
    """Save calls wait_cpu_optim_all() so we don't snapshot mid-step."""
    _, mgr, optim = saved_checkpoint
    target = tmp_path / "spy_save"
    target.mkdir()

    with mock.patch.object(
        mgr, "wait_cpu_optim_all", wraps=mgr.wait_cpu_optim_all
    ) as spy:
        _save_protrain_optim_dir(
            optim,
            str(target),
            step=99,
            save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
        )
        assert spy.called


@pytest.mark.gpu
def test_load_succeeds_from_pristine_checkpoint(fresh_checkpoint_dir, saved_checkpoint):
    """Sanity: a clean copy of the saved dir loads without error."""
    _, _, optim = saved_checkpoint
    assert _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir)) is True


@pytest.mark.gpu
def test_load_actually_restores_inner_state(fresh_checkpoint_dir, saved_checkpoint):
    """Load overwrites in-memory state with disk state.

    Stronger than test_load_succeeds_from_pristine_checkpoint: snapshot
    the inner adapters' state, mutate the in-memory tensors, load from
    disk, and verify state matches the snapshot bit-identical. The
    earlier "load returned True" assertion proved the function ran but
    not that it restored anything.
    """
    import copy

    import torch

    _, _, optim = saved_checkpoint

    def _snapshot_inner_states():
        snap = {}
        if optim._gpu_optim is not None:
            snap["gpu"] = copy.deepcopy(optim._gpu_optim._optim.state_dict())
        if optim._cpu_optim is not None:
            snap["cpu"] = {
                cid: copy.deepcopy(inner.state_dict())
                for cid, inner in optim._cpu_optim._optims.items()
            }
        return snap

    pre_load = _snapshot_inner_states()

    # Mutate every state tensor in-memory so a no-op load would be visible.
    def _mutate_inner_states(by: float):
        if optim._gpu_optim is not None:
            for s in optim._gpu_optim._optim.state.values():
                for v in s.values():
                    if isinstance(v, torch.Tensor):
                        v.add_(by)
        if optim._cpu_optim is not None:
            for inner in optim._cpu_optim._optims.values():
                for s in inner.state.values():
                    for v in s.values():
                        if isinstance(v, torch.Tensor):
                            v.add_(by)

    _mutate_inner_states(by=1.0)
    # Sanity: the mutation actually changed state vs the snapshot.
    mutated = _snapshot_inner_states()
    assert mutated != pre_load, (
        "test setup failure: mutation didn't change state — "
        "the load assertion below would be vacuous"
    )

    # Load from the on-disk copy.
    assert _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir)) is True

    post_load = _snapshot_inner_states()

    # Compare every tensor value
    def _states_match(a, b) -> bool:
        if set(a) != set(b):
            return False
        for k in a:
            sa, sb = a[k], b[k]
            if isinstance(sa, dict) and isinstance(sb, dict):
                if not _states_match(sa, sb):
                    return False
            elif isinstance(sa, torch.Tensor) and isinstance(sb, torch.Tensor):
                if not torch.equal(sa, sb):
                    return False
            else:
                if sa != sb:
                    return False
        return True

    assert _states_match(post_load, pre_load), (
        "load did not restore inner state to pre-mutation snapshot"
    )


@pytest.mark.gpu
def test_callback_unwraps_accelerated_optimizer(tmp_path, saved_checkpoint):
    """Callback fires through Accelerate's AcceleratedOptimizer wrapper.

    Regression for the bug where Trainer.optimizer is replaced by
    AcceleratedOptimizer post-prepare; without unwrap, the callback's
    duck-type check fails on the wrapper and protrain_optim/ is never
    written in real Trainer runs.
    """
    pytest.importorskip("accelerate")
    from accelerate.optimizer import AcceleratedOptimizer

    _, _, raw_optim = saved_checkpoint

    # Construct the wrapper. We disable device_placement to avoid the
    # prepare round-trip's extra state_dict/load_state_dict pass —
    # those work via the no-op patches in real Trainer runs but we
    # don't need them for this regression test.
    try:
        wrapped = AcceleratedOptimizer(raw_optim, device_placement=False)
    except Exception as e:
        pytest.skip(f"AcceleratedOptimizer needs accelerate state init: {e}")

    # Build a checkpoint dir per HF's convention.
    output_dir = tmp_path / "trainer_out"
    output_dir.mkdir()
    step = 7
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir()

    cb = make_checkpoint_callback(save_max_bytes=DEFAULT_SAVE_MAX_BYTES)
    # process_index/world_size must be real ints — Phase 2 Mode-B
    # orchestration uses HF's args.process_index / args.world_size to
    # decide who writes.
    fake_args = mock.MagicMock(
        output_dir=str(output_dir), process_index=0, world_size=1
    )
    fake_state = mock.MagicMock(global_step=step)
    fake_control = mock.MagicMock()

    # The callback receives the wrapped optimizer (mimics HF's
    # callback_handler.on_save signature).
    cb.on_save(fake_args, fake_state, fake_control, optimizer=wrapped)

    # Verify the ProTrain shard was actually written.
    assert (ckpt_dir / PROTRAIN_OPTIM_DIRNAME / "metadata.json").is_file(), (
        "callback failed to write protrain_optim/ when handed an "
        "AcceleratedOptimizer wrapper — the unwrap path is broken"
    )


@pytest.mark.gpu
def test_load_uses_map_location_cpu(fresh_checkpoint_dir, saved_checkpoint):
    """Every torch.load call uses map_location='cpu' (defeats HF's hostile default)."""
    import torch

    _, _, optim = saved_checkpoint
    seen: list = []
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    with mock.patch("axolotl.integrations.protrain.api.checkpoint.torch.load", spy):
        _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir))

    assert seen, "no torch.load calls observed"
    assert all(loc == "cpu" for loc in seen), seen


@pytest.mark.gpu
def test_load_rejects_layout_signature_mismatch(fresh_checkpoint_dir, saved_checkpoint):
    _, _, optim = saved_checkpoint
    meta_path = fresh_checkpoint_dir / PROTRAIN_OPTIM_DIRNAME / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["protrain_layout_signature"] = "deadbeef" * 8
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match="layout signature mismatch"):
        _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir))


@pytest.mark.gpu
def test_load_rejects_unknown_format_version(fresh_checkpoint_dir, saved_checkpoint):
    _, _, optim = saved_checkpoint
    meta_path = fresh_checkpoint_dir / PROTRAIN_OPTIM_DIRNAME / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["format_version"] = 99
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match="format_version"):
        _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir))


@pytest.mark.gpu
def test_load_accepts_world_size_change_for_replicated(
    fresh_checkpoint_dir, saved_checkpoint
):
    """Phase 2 Option B: replicated checkpoints saved with world_size=N
    can load into world_size=M (state shape is rank-independent).

    Tampering metadata to claim a different saved world_size + matching
    layout signature must load cleanly. The Phase 1 test that asserted
    the inverse was a Phase-1 hard-guard artifact.
    """
    _, _, optim = saved_checkpoint
    meta_path = fresh_checkpoint_dir / PROTRAIN_OPTIM_DIRNAME / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["protrain_world_size"] = 4
    # Layout signature embeds world_size; recompute it for the saved
    # value so the only difference is world_size itself.
    chunk_manager = optim._chunk_manager
    meta["protrain_layout_signature"] = _layout_signature(
        chunk_manager,
        world_size=4,
        zero3_shard=bool(getattr(chunk_manager, "zero3_shard", False)),
    )
    meta_path.write_text(json.dumps(meta))

    assert _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir)) is True


@pytest.mark.gpu
def test_load_rejects_zero3_mismatch(fresh_checkpoint_dir, saved_checkpoint):
    _, _, optim = saved_checkpoint
    meta_path = fresh_checkpoint_dir / PROTRAIN_OPTIM_DIRNAME / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["protrain_zero3_shard"] = True
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match="zero3_shard mismatch"):
        _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir))


@pytest.mark.gpu
def test_load_rejects_missing_chunk_file(fresh_checkpoint_dir, saved_checkpoint):
    _, _, optim = saved_checkpoint
    if optim._cpu_optim is None or not optim._cpu_optim._optims:
        pytest.skip("test requires at least one non-persistent CPU chunk")

    cpu_dir = fresh_checkpoint_dir / PROTRAIN_OPTIM_DIRNAME / "cpu_optim"
    sorted(cpu_dir.iterdir())[0].unlink()

    with pytest.raises(RuntimeError, match="CPU chunk set mismatch"):
        _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir))


@pytest.mark.gpu
def test_load_rejects_missing_metadata(fresh_checkpoint_dir, saved_checkpoint):
    _, _, optim = saved_checkpoint
    (fresh_checkpoint_dir / PROTRAIN_OPTIM_DIRNAME / "metadata.json").unlink()
    with pytest.raises(RuntimeError, match="lacks metadata.json"):
        _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir))


# ---------------------------------------------------------------------------
# Functional-equivalence-under-resume — separate-process verification
# ---------------------------------------------------------------------------
# The single-process version of this test segfaults on the rig because
# two ChunkManager instantiations exhaust the pinned-host allocator
# even with explicit restore_to_gpu / shutdown / gc. Workaround: run
# each arm of the experiment in a fresh subprocess via
# ``multiprocessing.Process`` with the ``spawn`` start method. Process
# teardown reclaims pinned host memory cleanly.
#
# Three arms:
#   * Reference: 4 continuous steps from scratch → final params
#   * Save:      2 steps from scratch → save state to disk
#   * Resume:    load state from save → 2 more steps → final params
#
# Each arm is its own subprocess. Driver compares the reference's
# final params to the resume's final params with torch.allclose.


def _arm_continuous_training(
    start_step: int,
    end_step: int,
    load_dir: str | None,
    save_dir: str | None,
    output_path: str | None,
    error_path: str,
) -> None:
    """One arm of the continued-training experiment, run inside a
    fresh subprocess.

    Half-open step range ``[start_step, end_step)``. If ``load_dir``
    is set, load BOTH model weights (model_state.pt) AND optimizer
    state (protrain_optim/) before the loop — mirrors HF Trainer's
    real resume flow where model weights and optimizer state both
    live in the checkpoint dir. If ``save_dir`` is set, save both.
    If ``output_path`` is set, write a snapshot of model params there.

    Errors are captured to ``error_path`` so the parent process can
    surface them after seeing a non-zero exitcode.
    """
    import os
    import traceback

    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    try:
        import torch

        torch.manual_seed(0)
        model = _tiny_model().to("cuda")
        mgr, _host = _build_chunk_manager(model, n_persist=1, S_chunk=64 * 1024)
        mgr.materialize_offload()
        _, _, optim = _build_optim_pair(model, mgr)

        if load_dir is not None:
            from axolotl.integrations.protrain.api.checkpoint import (
                _load_protrain_optim_dir,
            )

            # Load model weights into the gathered (on-GPU) chunks.
            # Gather every non-persistent chunk first so param.data is
            # real GPU storage (otherwise load_state_dict's tensor
            # copy would write into the empty placeholder).
            for cid in list(mgr._non_persistent_ids):
                mgr.gather(cid)
            saved_model_state = torch.load(
                os.path.join(load_dir, "model_state.pt"),
                map_location="cuda",
                weights_only=False,
            )
            model.load_state_dict(saved_model_state)

            ok = _load_protrain_optim_dir(optim, load_dir)
            assert ok, "load_protrain_optim_dir returned False unexpectedly"

        for step_idx in range(start_step, end_step):
            # Deterministic batch RNG keyed on absolute step idx so
            # reference and resume see identical batches at the same
            # step idx regardless of how they got there.
            torch.manual_seed(100 + step_idx)
            for cid in list(mgr._non_persistent_ids):
                mgr.gather(cid)
            optim.zero_grad()
            x = torch.randn(2, model.embed.in_features, device="cuda")
            out = model(x)
            out.sum().backward()
            optim.step()

        if save_dir is not None:
            from axolotl.integrations.protrain.api.checkpoint import (
                DEFAULT_SAVE_MAX_BYTES,
                _save_protrain_optim_dir,
            )

            # Save model weights AND optimizer state. Mirrors HF
            # Trainer's behavior: checkpoint dir contains both.
            # Gather every chunk before snapshotting weights so all
            # param.data tensors hold real values.
            for cid in list(mgr._non_persistent_ids):
                mgr.gather(cid)
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "model_state.pt"),
            )

            wrote = _save_protrain_optim_dir(
                optim,
                save_dir,
                step=end_step,
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
            )
            assert wrote is True, "save returned False unexpectedly"

        if output_path is not None:
            # Gather every chunk so each param.data is real GPU
            # storage (post-step, offloaded params have empty
            # placeholders again).
            for cid in list(mgr._non_persistent_ids):
                mgr.gather(cid)
            snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            torch.save(snap, output_path)

    except BaseException:
        with open(error_path, "w") as f:
            traceback.print_exc(file=f)
        raise


@pytest.mark.gpu
@pytest.mark.slow
def test_continued_training_matches_continuous_via_subprocess(tmp_path):
    """Functional equivalence: N save+load+M matches N+M continuous.

    Three subprocess arms (reference, save-half, resume-half), spawn
    start method, fresh CUDA state per arm. Final params from the
    resume arm must match the reference within tight tol — proves the
    saved optimizer state is functionally meaningful, not just
    syntactically equal to its source.
    """
    import multiprocessing as mp

    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    ctx = mp.get_context("spawn")

    ref_out = tmp_path / "ref_params.pt"
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    resume_out = tmp_path / "resume_params.pt"

    def _spawn_arm(
        start: int,
        end: int,
        load_d: str | None,
        save_d: str | None,
        out: str | None,
        tag: str,
    ) -> None:
        err = tmp_path / f"err_{tag}.txt"
        p = ctx.Process(
            target=_arm_continuous_training,
            args=(start, end, load_d, save_d, out, str(err)),
        )
        p.start()
        p.join(timeout=180)
        if p.is_alive():
            p.terminate()
            pytest.fail(f"arm {tag!r} timed out after 180s")
        if p.exitcode != 0:
            err_text = err.read_text() if err.exists() else "(no traceback captured)"
            pytest.fail(f"arm {tag!r} exited with code {p.exitcode}:\n{err_text}")

    # Reference: 4 continuous steps from scratch
    _spawn_arm(0, 4, None, None, str(ref_out), tag="reference")

    # Save arm: 2 steps from scratch, save state
    _spawn_arm(0, 2, None, str(save_dir), None, tag="save")

    # Resume arm: load state, run steps 2 and 3
    _spawn_arm(2, 4, str(save_dir), None, str(resume_out), tag="resume")

    ref = torch.load(ref_out, map_location="cpu", weights_only=False)
    resume = torch.load(resume_out, map_location="cpu", weights_only=False)

    assert set(ref) == set(resume), (
        f"param name sets differ: "
        f"only_ref={set(ref) - set(resume)}, only_resume={set(resume) - set(ref)}"
    )
    for name, ref_t in ref.items():
        cur_t = resume[name]
        assert ref_t.shape == cur_t.shape, (
            f"shape mismatch on {name!r}: ref={ref_t.shape} resume={cur_t.shape}"
        )
        assert torch.allclose(cur_t, ref_t, atol=1e-5, rtol=1e-4), (
            f"param {name!r} diverged after subprocess resume: "
            f"max_abs_diff={(cur_t - ref_t).abs().max().item():.3e}, "
            f"max_rel_diff={((cur_t - ref_t).abs() / ref_t.abs().clamp(min=1e-8)).max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# Phase 2 Mode-B (DDP-replicated) — schema, forward compat, dispatcher
# ---------------------------------------------------------------------------


def test_load_rejects_v2_metadata_missing_save_mode(tmp_path):
    """v2 saves MUST carry protrain_save_mode; missing it is a hard error.

    The forward-compat path applies only to v1 saves; v2 saves with
    incomplete metadata indicate corruption or a pre-release schema.
    """
    proot = tmp_path / PROTRAIN_OPTIM_DIRNAME
    proot.mkdir()
    bad_meta = {
        "format_version": 2,
        "protrain_layout_signature": "0" * 64,
        "protrain_persistent_ids": [],
        "protrain_n_buffer": 1,
        "protrain_world_size": 1,
        "protrain_zero3_shard": False,
        "saving_rank": 0,
        # protrain_save_mode is missing on purpose
        "param_groups_meta": [],
        "saved_at_step": 0,
        "torch_version": "x",
        "estimated_optim_state_bytes": 0,
    }
    (proot / "metadata.json").write_text(json.dumps(bad_meta))
    fake_optim = mock.MagicMock(spec=["_gpu_optim", "_cpu_optim", "_chunk_manager"])
    fake_optim._chunk_manager = mock.MagicMock(zero3_shard=False)
    with pytest.raises(RuntimeError, match="protrain_save_mode"):
        _load_protrain_optim_dir(fake_optim, str(tmp_path))


def test_load_rejects_save_mode_mismatch_replicated_to_sharded(tmp_path):
    """Saved replicated, current sharded → hard error pointing at Mode-C.

    Catches the user trying to resume a Phase-1 / Mode-B replicated
    save into a ZeRO-3 sharded run. The on-disk shape doesn't match
    what the current run needs.
    """
    proot = tmp_path / PROTRAIN_OPTIM_DIRNAME
    proot.mkdir()
    meta = {
        "format_version": 2,
        "protrain_layout_signature": "0" * 64,
        "protrain_persistent_ids": [],
        "protrain_n_buffer": 1,
        "protrain_world_size": 1,
        "protrain_zero3_shard": False,
        "protrain_save_mode": "replicated",
        "saving_rank": 0,
        "param_groups_meta": [],
        "saved_at_step": 0,
        "torch_version": "x",
        "estimated_optim_state_bytes": 0,
    }
    (proot / "metadata.json").write_text(json.dumps(meta))
    fake_optim = mock.MagicMock(spec=["_gpu_optim", "_cpu_optim", "_chunk_manager"])
    fake_optim._chunk_manager = mock.MagicMock(zero3_shard=True)
    with pytest.raises(RuntimeError, match="save_mode mismatch"):
        _load_protrain_optim_dir(fake_optim, str(tmp_path))


def test_load_rejects_save_mode_mismatch_sharded_to_replicated(tmp_path):
    """Saved sharded, current replicated → hard error.

    Inverse of the above: rank-0 of a replicated run can't reconstruct
    full state from sharded files without a re-shard step (out of scope).
    """
    proot = tmp_path / PROTRAIN_OPTIM_DIRNAME
    proot.mkdir()
    meta = {
        "format_version": 2,
        "protrain_layout_signature": "0" * 64,
        "protrain_persistent_ids": [],
        "protrain_n_buffer": 1,
        "protrain_world_size": 2,
        "protrain_zero3_shard": True,
        "protrain_save_mode": "sharded",
        "saving_rank": 0,
        "param_groups_meta": [],
        "saved_at_step": 0,
        "torch_version": "x",
        "estimated_optim_state_bytes": 0,
    }
    (proot / "metadata.json").write_text(json.dumps(meta))
    fake_optim = mock.MagicMock(spec=["_gpu_optim", "_cpu_optim", "_chunk_manager"])
    fake_optim._chunk_manager = mock.MagicMock(zero3_shard=False)
    with pytest.raises(RuntimeError, match="save_mode mismatch"):
        _load_protrain_optim_dir(fake_optim, str(tmp_path))


# ---------------------------------------------------------------------------
# Phase 2 Mode-C (ZeRO-3 sharded) — CPU-only unit tests for helpers
# ---------------------------------------------------------------------------


def _make_region_dict(
    chunk_offset: int = 0,
    region_bytes: int = 1024,
    region_bytes_padded: int = 1024,
    shard_bytes: int = 512,
    dtype: str = "torch.float16",
) -> dict:
    return {
        "chunk_offset": chunk_offset,
        "region_bytes": region_bytes,
        "region_bytes_padded": region_bytes_padded,
        "shard_bytes": shard_bytes,
        "dtype": dtype,
    }


def test_validate_regions_match_passes_on_identical_layout():
    """Identical region descriptors round-trip cleanly."""
    a = {"0": [_make_region_dict()], "1": [_make_region_dict(chunk_offset=2048)]}
    b = {"0": [_make_region_dict()], "1": [_make_region_dict(chunk_offset=2048)]}
    _validate_regions_match(a, b)  # no raise


def test_validate_regions_match_rejects_chunk_id_mismatch():
    a = {"0": [_make_region_dict()], "1": [_make_region_dict()]}
    b = {"0": [_make_region_dict()], "2": [_make_region_dict()]}
    with pytest.raises(RuntimeError, match="chunk-id mismatch"):
        _validate_regions_match(a, b)


def test_validate_regions_match_rejects_region_count_mismatch():
    a = {"0": [_make_region_dict()]}
    b = {"0": [_make_region_dict(), _make_region_dict(chunk_offset=2048)]}
    with pytest.raises(RuntimeError, match="region count mismatch.*chunk 0"):
        _validate_regions_match(a, b)


def test_validate_regions_match_rejects_dtype_mismatch():
    a = {"0": [_make_region_dict(dtype="torch.float16")]}
    b = {"0": [_make_region_dict(dtype="torch.bfloat16")]}
    with pytest.raises(RuntimeError, match="field 'dtype'"):
        _validate_regions_match(a, b)


def test_validate_regions_match_rejects_shard_bytes_mismatch():
    """world_size change typically manifests as a shard_bytes drift."""
    a = {"0": [_make_region_dict(shard_bytes=512)]}
    b = {"0": [_make_region_dict(shard_bytes=256)]}
    with pytest.raises(RuntimeError, match="field 'shard_bytes'"):
        _validate_regions_match(a, b)


def test_validate_regions_match_rejects_chunk_offset_mismatch():
    a = {"0": [_make_region_dict(chunk_offset=0)]}
    b = {"0": [_make_region_dict(chunk_offset=64)]}
    with pytest.raises(RuntimeError, match="field 'chunk_offset'"):
        _validate_regions_match(a, b)


def test_build_regions_per_chunk_emits_expected_descriptors():
    """`_build_regions_per_chunk` walks `_chunk_shards` and serializes
    every region's load-bearing fields."""
    import torch

    fake_region_a = mock.MagicMock(
        chunk_offset=0,
        region_bytes=1000,
        region_bytes_padded=1024,
        shard_bytes=512,
        dtype=torch.float16,
    )
    fake_region_b = mock.MagicMock(
        chunk_offset=1024,
        region_bytes=128,
        region_bytes_padded=128,
        shard_bytes=64,
        dtype=torch.float32,
    )
    fake_shard_state = mock.MagicMock(regions=[fake_region_a, fake_region_b])
    chunk_manager = mock.MagicMock()
    chunk_manager._chunk_shards = {ChunkId(0): fake_shard_state}

    out = _build_regions_per_chunk(chunk_manager)
    assert "0" in out
    assert len(out["0"]) == 2
    assert out["0"][0]["chunk_offset"] == 0
    assert out["0"][0]["region_bytes"] == 1000
    assert out["0"][0]["region_bytes_padded"] == 1024
    assert out["0"][0]["shard_bytes"] == 512
    assert out["0"][0]["dtype"] == "torch.float16"
    assert out["0"][1]["chunk_offset"] == 1024
    assert out["0"][1]["dtype"] == "torch.float32"


def test_build_regions_per_chunk_empty_when_no_chunk_shards():
    """Replicated-mode managers have an empty `_chunk_shards`."""
    chunk_manager = mock.MagicMock()
    chunk_manager._chunk_shards = {}
    assert _build_regions_per_chunk(chunk_manager) == {}


def test_load_rejects_sharded_metadata_missing_regions_per_chunk(tmp_path):
    """A v2 sharded save that lacks regions_per_chunk is rejected.

    Catches a corrupt file or a forward-incompat producer; the loader
    needs the descriptors to validate the rebuilt shard_param fits the
    saved bytes.
    """
    proot = tmp_path / PROTRAIN_OPTIM_DIRNAME
    proot.mkdir()
    meta = {
        "format_version": 2,
        "protrain_layout_signature": "0" * 64,
        "protrain_persistent_ids": [],
        "protrain_n_buffer": 1,
        "protrain_world_size": 2,
        "protrain_zero3_shard": True,
        "protrain_save_mode": "sharded",
        "saving_rank": 0,
        "param_groups_meta": [],
        "saved_at_step": 0,
        "torch_version": "x",
        "estimated_optim_state_bytes": 0,
        # regions_per_chunk missing on purpose
    }
    (proot / "metadata.json").write_text(json.dumps(meta))
    fake_optim = mock.MagicMock(spec=["_gpu_optim", "_cpu_optim", "_chunk_manager"])
    # Pretend we're in a 2-rank sharded run so we get past the
    # save_mode/world_size guards and reach the regions check.
    fake_optim._chunk_manager = mock.MagicMock(zero3_shard=True)
    fake_optim._chunk_manager._chunk_shards = {}
    with mock.patch(
        "axolotl.integrations.protrain.api.checkpoint._current_world_size",
        return_value=2,
    ):
        with pytest.raises(RuntimeError, match="regions_per_chunk"):
            _load_protrain_optim_dir(fake_optim, str(tmp_path))


def test_load_rejects_sharded_world_size_change(tmp_path):
    """Mode-C resume requires identical world_size; mismatch hard-errors.

    Sharded shard arithmetic depends on world_size — cross-world-size
    resume is out of scope for Phase 2.
    """
    proot = tmp_path / PROTRAIN_OPTIM_DIRNAME
    proot.mkdir()
    meta = {
        "format_version": 2,
        "protrain_layout_signature": "0" * 64,
        "protrain_persistent_ids": [],
        "protrain_n_buffer": 1,
        "protrain_world_size": 2,
        "protrain_zero3_shard": True,
        "protrain_save_mode": "sharded",
        "saving_rank": 0,
        "param_groups_meta": [],
        "saved_at_step": 0,
        "torch_version": "x",
        "estimated_optim_state_bytes": 0,
        "regions_per_chunk": {"0": [_make_region_dict()]},
    }
    (proot / "metadata.json").write_text(json.dumps(meta))
    fake_optim = mock.MagicMock(spec=["_gpu_optim", "_cpu_optim", "_chunk_manager"])
    fake_optim._chunk_manager = mock.MagicMock(zero3_shard=True)
    fake_optim._chunk_manager._chunk_shards = {}
    # Saved world=2; pretend current world=4 → must error.
    with mock.patch(
        "axolotl.integrations.protrain.api.checkpoint._current_world_size",
        return_value=4,
    ):
        with pytest.raises(RuntimeError, match="world_size"):
            _load_protrain_optim_dir(fake_optim, str(tmp_path))


# ---------------------------------------------------------------------------
# v1 forward-compat — write a Phase-1 layout, load it under Phase-2 code
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_replicated_load_v1_checkpoint_is_forward_compat(
    fresh_checkpoint_dir, saved_checkpoint
):
    """v1 saves load cleanly under v2 code as Mode-B replicated, ws=1.

    Mutates the saved metadata to look like a Phase-1 (v1) save: drops
    the v2-only fields and renames format_version to 1. Phase-2 loader
    must infer save_mode=replicated, saving_rank=0, world_size=1 and
    proceed without error.
    """
    _, _, optim = saved_checkpoint
    meta_path = fresh_checkpoint_dir / PROTRAIN_OPTIM_DIRNAME / "metadata.json"
    meta = json.loads(meta_path.read_text())
    # Strip the v2-only fields so the metadata looks like a v1 save.
    meta.pop("protrain_save_mode", None)
    meta.pop("saving_rank", None)
    meta["format_version"] = 1
    meta_path.write_text(json.dumps(meta))

    # Loader must accept this without raising.
    assert _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir)) is True


@pytest.mark.gpu
def test_resume_from_legacy_v2_into_v3(fresh_checkpoint_dir, saved_checkpoint):
    """Legacy v2 metadata + gpu_optim.pt loads into a v3 build cleanly.

    Mutates the saved metadata to look like a v2 save (drops the
    partition_version field and rewrites format_version=2). The legacy
    ``gpu_optim.pt`` file stays in place. Loader must accept this on
    rank-0 (broadcast disabled in single-process) and the next save
    activates the partition.
    """
    _, _, optim = saved_checkpoint
    meta_path = fresh_checkpoint_dir / PROTRAIN_OPTIM_DIRNAME / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("protrain_persistent_partition_version", None)
    meta.pop("protrain_persistent_owner_world_size", None)
    meta["format_version"] = 2
    meta_path.write_text(json.dumps(meta))

    # The on-disk file is gpu_optim.pt — the v3 build (with partition
    # disabled at world=1) must load it without raising.
    assert _load_protrain_optim_dir(optim, str(fresh_checkpoint_dir)) is True


@pytest.mark.gpu
def test_partitioned_persistent_save_load_roundtrip(tmp_path, saved_checkpoint):
    """Manually craft a partition-active save (w=2) and verify the
    loader rejects mismatched world_size with the documented error.

    The full multi-rank live round-trip is covered by the mp.spawn test
    in test_world_size_reshard.py. This test exercises the
    single-process schema-check path on a hand-rolled directory.
    """
    _, _, optim = saved_checkpoint
    target = tmp_path / "partitioned_save" / PROTRAIN_OPTIM_DIRNAME
    target.mkdir(parents=True)

    chunk_manager = optim._chunk_manager
    saved_world = 2  # claim a partitioned w=2 save

    import torch as _torch

    meta = {
        "format_version": SCHEMA_FORMAT_VERSION,
        "protrain_layout_signature": _layout_signature(
            chunk_manager,
            world_size=saved_world,
            zero3_shard=bool(getattr(chunk_manager, "zero3_shard", False)),
        ),
        "protrain_persistent_ids": sorted(int(x) for x in chunk_manager._persistent_ids),
        "protrain_n_buffer": int(getattr(chunk_manager, "n_buffer", 0)),
        "protrain_world_size": saved_world,
        "protrain_zero3_shard": False,
        "protrain_save_mode": SAVE_MODE_REPLICATED,
        "saving_rank": 0,
        "param_groups_meta": [],
        "saved_at_step": 1,
        "torch_version": str(_torch.__version__),
        "estimated_optim_state_bytes": 0,
        "protrain_persistent_partition_version": 1,
        "protrain_persistent_owner_world_size": saved_world,
    }
    (target / METADATA_FILENAME).write_text(json.dumps(meta))
    # Two per-rank gpu_optim files — proof the writer would have laid them down.
    for r in range(saved_world):
        _torch.save(
            optim._gpu_optim._optim.state_dict() if optim._gpu_optim else {},
            target / f"gpu_optim_rank_{r}.pt",
        )

    # Single-rank load attempts to resume a w=2 partitioned save → reject.
    with pytest.raises(RuntimeError, match="world_size mismatch on resume"):
        _load_protrain_optim_dir(optim, str(tmp_path / "partitioned_save"))


# ---------------------------------------------------------------------------
# Mode-B multi-rank (gloo + mp.spawn) — slow lane
# ---------------------------------------------------------------------------
# The pattern here mirrors test_chunk_manager_offload.py:875
# (_worker_sharded_restore_round_trip): each rank initializes a gloo
# process group via a file:// rendezvous in tmpdir, runs its body, and
# tears down the group. Tests downgrade to skip if a required gloo
# collective isn't available on this build (rank{N}.skip files).


def _common_worker_setup(rank: int, world_size: int, tmpdir: str, tag: str):
    """Init gloo process group + return ``(model, mgr, optim, host)``.

    The chunk_manager is built with the same seed across ranks so every
    rank holds the same starting weights — the Mode-B replication
    invariant. After one fwd+bwd+step every rank's optimizer state is
    identical.
    """
    import os as _os

    import torch
    import torch.distributed as dist

    _os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("worker: CUDA not available")

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-{tag}",
        rank=rank,
        world_size=world_size,
    )

    torch.manual_seed(0)  # identical init across ranks
    model = _tiny_model().to("cuda")
    mgr, host = _build_chunk_manager(model, n_persist=1, S_chunk=64 * 1024)
    mgr.materialize_offload()
    _, _, optim = _build_optim_pair(model, mgr)
    # Replicate one fwd+bwd+step with a DETERMINISTIC batch — torch.randn
    # advances per-process CUDA RNG that may diverge between mp.spawn
    # workers (deepspeed/apex import side effects can consume RNG
    # unequally). Build the input on CPU from a fresh-seeded generator
    # then copy to GPU so the byte values are identical across ranks.
    import torch as _torch  # local alias to satisfy linters

    cpu_gen = _torch.Generator(device="cpu")
    cpu_gen.manual_seed(123)
    x = _torch.randn(2, model.embed.in_features, generator=cpu_gen).to("cuda")
    for cid in list(mgr._non_persistent_ids):
        mgr.gather(cid)
    optim.zero_grad()
    out = model(x)
    out.sum().backward()
    optim.step()
    return model, mgr, optim, host


def _force_identical_inner_state(optim) -> None:
    """Zero every inner-state tensor — guarantees byte-identical state
    across ranks regardless of step-time numerical noise.

    The cross-rank verify and the load tests exercise the
    save/load/verify *mechanisms*, not DDP-determinism (which is the
    framework's contract, verified elsewhere). Forcing zeros eliminates
    non-determinism from CPU-adam threading, async copies, or
    per-process RNG drift between mp.spawn workers.
    """
    import torch as _torch

    if optim._gpu_optim is not None:
        for s in optim._gpu_optim._optim.state.values():
            for k, v in s.items():
                if isinstance(v, _torch.Tensor):
                    v.zero_()
                elif isinstance(v, int):
                    s[k] = 0
    if optim._cpu_optim is not None:
        for inner in optim._cpu_optim._optims.values():
            for s in inner.state.values():
                for k, v in s.items():
                    if isinstance(v, _torch.Tensor):
                        v.zero_()
                    elif isinstance(v, int):
                        s[k] = 0


def _worker_replicated_save_only_rank_0_writes(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Rank-0 writes; rank-1 must NOT create any extra files.

    Drives the callback through a fake HF args object (output_dir +
    process_index + world_size). Rank-1 writes a sentinel file
    naming itself; the parent test asserts there are no rank-suffix
    files in protrain_optim/ and that rank-1 reached the post-save
    point (so the callback didn't deadlock).
    """
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_worker_setup(
            rank, world_size, tmpdir, tag="r0only"
        )
        try:
            output_dir = _os.path.join(tmpdir, "trainer_out")
            if rank == 0:
                _os.makedirs(output_dir, exist_ok=True)
            dist.barrier()  # output_dir must exist before any rank's callback
            ckpt_dir = _os.path.join(output_dir, "checkpoint-1")
            if rank == 0:
                _os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()

            cb = make_checkpoint_callback(save_max_bytes=DEFAULT_SAVE_MAX_BYTES)
            fake_args = mock.MagicMock(
                output_dir=output_dir,
                process_index=rank,
                world_size=world_size,
            )
            fake_state = mock.MagicMock(global_step=1)
            fake_control = mock.MagicMock()

            cb.on_save(fake_args, fake_state, fake_control, optimizer=optim)

            # Both ranks reach this point — sentinel for liveness.
            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_replicated_save_only_rank_0_writes(tmp_path):
    """mp.spawn 2 gloo ranks: only rank-0's files appear on disk.

    The on-disk layout in Mode-B has no per-rank suffix
    (CHECKPOINT_DESIGN_PHASE2.md §2.1). Both ranks call the callback
    but only rank-0 actually writes; rank-1 must reach the
    post-callback point (sentinel rank1.done) without creating extra
    files.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_replicated_save_only_rank_0_writes,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")

    # Both ranks must have reached the post-save sentinel.
    for r in range(world_size):
        assert (tmp_path / f"rank{r}.done").is_file(), (
            f"rank {r} did not reach post-callback point"
        )

    # Verify the directory layout has no rank suffix.
    proot = tmp_path / "trainer_out" / "checkpoint-1" / PROTRAIN_OPTIM_DIRNAME
    assert (proot / METADATA_FILENAME).is_file()
    assert (proot / GPU_OPTIM_FILENAME).is_file()

    cpu_dir = proot / CPU_OPTIM_DIRNAME
    if cpu_dir.is_dir():
        for entry in cpu_dir.iterdir():
            # Must match chunk_<N>.pt — no rank suffix in Mode-B.
            assert "_rank_" not in entry.name, (
                f"Mode-B file has unexpected rank suffix: {entry.name}"
            )

    # The metadata records save_mode=replicated, saving_rank=0,
    # protrain_world_size=2.
    meta = json.loads((proot / METADATA_FILENAME).read_text())
    assert meta["protrain_save_mode"] == SAVE_MODE_REPLICATED
    assert meta["saving_rank"] == 0
    assert meta["protrain_world_size"] == 2


def _worker_replicated_load_succeeds_on_all_ranks(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Each rank loads from the same path, verifies state matches pre-save.

    Step 1: every rank builds a fresh chunk_manager, takes one step
    (state X). Rank-0 saves. All ranks barrier.
    Step 2: every rank mutates its in-memory state, then loads from
    the saved dir. Loaded state must match pre-mutation snapshot
    (== state X), proving the load actually reads files (and that
    rank-1 finds the file rank-0 wrote).
    """
    import copy
    import os as _os

    import torch
    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_worker_setup(
            rank, world_size, tmpdir, tag="loadall"
        )
        try:
            save_dir = _os.path.join(tmpdir, "save_root")
            if rank == 0:
                _os.makedirs(save_dir, exist_ok=True)
            dist.barrier()

            # Force byte-identical state across ranks. Mode-B's
            # contract is that DDP makes this true at runtime, but for
            # the load test we just need a known-equal baseline so the
            # post-load comparison is deterministic regardless of
            # CPU-adam threading or per-process RNG drift.
            _force_identical_inner_state(optim)

            # Snapshot inner state pre-save.
            def _snap():
                snap = {}
                if optim._gpu_optim is not None:
                    snap["gpu"] = copy.deepcopy(optim._gpu_optim._optim.state_dict())
                if optim._cpu_optim is not None:
                    snap["cpu"] = {
                        cid: copy.deepcopy(inner.state_dict())
                        for cid, inner in optim._cpu_optim._optims.items()
                    }
                return snap

            pre_save = _snap()

            # _save_protrain_optim_dir is collective (lockstep broadcast in
            # its finally — see api/checkpoint.py:_broadcast_status_or_raise);
            # every rank must call it. Only rank-0 actually writes (gated
            # internally), but every rank must reach the broadcast so a rank-0
            # write failure raises in lockstep instead of deadlocking the
            # trailing barrier.
            wrote = _save_protrain_optim_dir(
                optim,
                save_dir,
                step=1,
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                rank=rank,
                world_size=world_size,
            )
            assert wrote is True, f"rank {rank}: save returned False"
            dist.barrier()

            # Mutate every state tensor on every rank so a no-op load
            # would be visible.
            if optim._gpu_optim is not None:
                for s in optim._gpu_optim._optim.state.values():
                    for v in s.values():
                        if isinstance(v, torch.Tensor):
                            v.add_(7.0)
            if optim._cpu_optim is not None:
                for inner in optim._cpu_optim._optims.values():
                    for s in inner.state.values():
                        for v in s.values():
                            if isinstance(v, torch.Tensor):
                                v.add_(7.0)

            # Load from the same path on every rank.
            loaded = _load_protrain_optim_dir(optim, save_dir)
            assert loaded is True, f"rank {rank}: load returned False"

            post_load = _snap()

            def _states_match(a, b) -> bool:
                if set(a) != set(b):
                    return False
                for k in a:
                    sa, sb = a[k], b[k]
                    if isinstance(sa, dict) and isinstance(sb, dict):
                        if not _states_match(sa, sb):
                            return False
                    elif isinstance(sa, torch.Tensor) and isinstance(sb, torch.Tensor):
                        if not torch.equal(sa, sb):
                            return False
                    else:
                        if sa != sb:
                            return False
                return True

            assert _states_match(post_load, pre_save), (
                f"rank {rank}: load did not restore inner state"
            )

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_replicated_load_succeeds_on_all_ranks(tmp_path):
    """2 ranks load from rank-0's saved dir; loaded state matches pre-save.

    Verifies the Mode-B load contract (CHECKPOINT_DESIGN_PHASE2.md §2.5):
    every rank reads the same files into its own optimizer.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_replicated_load_succeeds_on_all_ranks,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")

    for r in range(world_size):
        assert (tmp_path / f"rank{r}.done").is_file(), (
            f"rank {r} did not reach post-load point"
        )


def _worker_estimate_gate_broadcast(rank: int, world_size: int, tmpdir: str) -> None:
    """Rank-0's estimate trips the threshold; rank-1's wouldn't on its own.

    Mocks ``_estimate_optim_state_bytes`` per-rank: rank-0 returns
    ``threshold + 1``; rank-1 returns 0. Without the broadcast,
    rank-0 would skip but rank-1 would write — partial save.
    With the broadcast, all ranks must skip together.
    """
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_worker_setup(
            rank, world_size, tmpdir, tag="gate"
        )
        try:
            output_dir = _os.path.join(tmpdir, "trainer_out")
            if rank == 0:
                _os.makedirs(output_dir, exist_ok=True)
            dist.barrier()
            ckpt_dir = _os.path.join(output_dir, "checkpoint-1")
            if rank == 0:
                _os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()

            small_threshold = 64
            # Per-rank patch: rank-0's estimate exceeds; rank-1's fits.
            per_rank_estimate = (small_threshold + 1) if rank == 0 else 0

            cb = make_checkpoint_callback(save_max_bytes=small_threshold)
            fake_args = mock.MagicMock(
                output_dir=output_dir,
                process_index=rank,
                world_size=world_size,
            )
            fake_state = mock.MagicMock(global_step=1)
            fake_control = mock.MagicMock()

            with mock.patch(
                "axolotl.integrations.protrain.api.checkpoint."
                "_estimate_optim_state_bytes",
                return_value=per_rank_estimate,
            ):
                cb.on_save(fake_args, fake_state, fake_control, optimizer=optim)

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_save_estimate_gate_broadcast_from_rank_0(tmp_path):
    """Rank-0's gate decision is broadcast; all ranks skip together.

    Without the broadcast (per-rank decide), rank-0 would skip but
    rank-1 would write — partial save → broken checkpoint
    (CHECKPOINT_DESIGN_PHASE2.md §4.4). Verifies no protrain_optim/
    files end up on disk despite rank-1's "would-fit" estimate.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_estimate_gate_broadcast,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")

    proot = tmp_path / "trainer_out" / "checkpoint-1" / PROTRAIN_OPTIM_DIRNAME
    assert not proot.exists() or not (proot / METADATA_FILENAME).exists(), (
        "Mode-B estimate gate failed: some rank wrote despite rank-0's "
        "skip decision — partial save means broken checkpoint."
    )


def _worker_verify_replicated_clean(rank: int, world_size: int, tmpdir: str) -> None:
    """Verify flag ON, identical state across ranks → save proceeds."""
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_worker_setup(
            rank, world_size, tmpdir, tag="verifyok"
        )
        try:
            output_dir = _os.path.join(tmpdir, "trainer_out")
            if rank == 0:
                _os.makedirs(output_dir, exist_ok=True)
            dist.barrier()
            ckpt_dir = _os.path.join(output_dir, "checkpoint-1")
            if rank == 0:
                _os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()

            # Force byte-identical state across ranks. The clean-run
            # test exercises the verify *mechanism*, not DDP
            # determinism (which is a different invariant).
            _force_identical_inner_state(optim)

            cb = make_checkpoint_callback(
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                verify_replicated=True,
            )
            fake_args = mock.MagicMock(
                output_dir=output_dir,
                process_index=rank,
                world_size=world_size,
            )
            fake_state = mock.MagicMock(global_step=1)
            fake_control = mock.MagicMock()

            cb.on_save(fake_args, fake_state, fake_control, optimizer=optim)

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_replicated_save_with_verify_flag_passes_on_clean_run(tmp_path):
    """Verify flag ON, identical state across ranks → save proceeds, no error."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_verify_replicated_clean,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")

    for r in range(world_size):
        assert (tmp_path / f"rank{r}.done").is_file()
    proot = tmp_path / "trainer_out" / "checkpoint-1" / PROTRAIN_OPTIM_DIRNAME
    assert (proot / METADATA_FILENAME).is_file(), (
        "verify-on clean-run did not produce a checkpoint"
    )


def _worker_verify_replicated_divergent(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Verify flag ON, mutate rank-1's state pre-save → expect RuntimeError."""
    import os as _os

    import torch
    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_worker_setup(
            rank, world_size, tmpdir, tag="verifybad"
        )
        try:
            output_dir = _os.path.join(tmpdir, "trainer_out")
            if rank == 0:
                _os.makedirs(output_dir, exist_ok=True)
            dist.barrier()
            ckpt_dir = _os.path.join(output_dir, "checkpoint-1")
            if rank == 0:
                _os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()

            # Force identical state on both ranks first, then mutate
            # rank-1's only — this isolates the verify path from any
            # incidental determinism issues in the chunk_manager
            # plumbing.
            _force_identical_inner_state(optim)

            # Tamper rank-1's state so the cross-rank hash compare fails.
            if rank == 1 and optim._cpu_optim is not None:
                for inner in optim._cpu_optim._optims.values():
                    for s in inner.state.values():
                        for v in s.values():
                            if isinstance(v, torch.Tensor):
                                v.add_(13.0)
            if rank == 1 and optim._gpu_optim is not None:
                for s in optim._gpu_optim._optim.state.values():
                    for v in s.values():
                        if isinstance(v, torch.Tensor):
                            v.add_(13.0)

            cb = make_checkpoint_callback(
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                verify_replicated=True,
            )
            fake_args = mock.MagicMock(
                output_dir=output_dir,
                process_index=rank,
                world_size=world_size,
            )
            fake_state = mock.MagicMock(global_step=1)
            fake_control = mock.MagicMock()

            try:
                cb.on_save(fake_args, fake_state, fake_control, optimizer=optim)
            except RuntimeError as exc:
                if "Mode-B precondition violated" in str(exc):
                    with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                        f.write(str(exc))
                else:
                    raise
            else:
                # No raise == bug. Mark sentinel so the parent test
                # fails loudly.
                with open(_os.path.join(tmpdir, f"rank{rank}.no_raise"), "w") as f:
                    f.write("verify did not raise on divergent state")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        if isinstance(exc, RuntimeError) and "Mode-B precondition violated" in str(exc):
            with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                f.write(str(exc))
            return
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_replicated_save_with_verify_flag_catches_divergence(tmp_path):
    """Verify flag ON, divergent state → RuntimeError naming the divergence.

    Mutates rank-1's state pre-save; the all_gather_object hash compare
    must trip. Both ranks raise (the all_gather is collective).
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    # mp.spawn re-raises the worker exception in the parent, but the
    # workers also write a "caught" sentinel so we can verify the
    # message regardless of how mp.spawn surfaces it.
    try:
        mp.spawn(
            _worker_verify_replicated_divergent,
            args=(world_size, str(tmp_path)),
            nprocs=world_size,
            join=True,
        )
    except Exception:
        # Expected: at least one rank raised RuntimeError. The
        # sentinel files distinguish "verify caught divergence" from
        # "actual unexpected error".
        pass

    no_raise = list(tmp_path.glob("rank*.no_raise"))
    if no_raise:
        bodies = "\n---\n".join(f.read_text() for f in no_raise)
        pytest.fail(f"verify did not raise on divergent state:\n{bodies}")

    caught = list(tmp_path.glob("rank*.caught"))
    assert caught, "no rank caught the verify-flag RuntimeError"
    # The error message names the divergent ranks.
    msgs = [f.read_text() for f in caught]
    assert any(
        "divergent ranks" in m and "Mode-B precondition violated" in m for m in msgs
    ), f"verify error did not mention divergent ranks: {msgs}"

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"unexpected worker errors:\n{bodies}")


# ---------------------------------------------------------------------------
# Phase 2 Mode-C (ZeRO-3 sharded) — multi-rank gloo + mp.spawn
# ---------------------------------------------------------------------------
# Mode-C writes per-rank chunk shards (chunk_<N>_rank_<R>.pt) so we
# need real distributed init even on a single-GPU box. Gloo's CPU
# collectives suffice for the file-bookkeeping path. The mixed-dtype
# model below produces multiple dtype regions per chunk — exercises
# the multi-region branch of regions_per_chunk.


def _build_sharded_chunk_manager_mixed_dtype(rank: int, world_size: int):
    """Mixed-dtype 1-block model + sharded ChunkManager for Mode-C tests.

    Uses an fp16 Linear + fp32 LayerNorm (mirrors
    test_chunk_manager_offload.py:875's ``_MixedLayer``) so the chunk
    contains multiple dtype regions and the regions_per_chunk path
    exercises real multi-region descriptors. Returns
    ``(model, mgr, host)``. Caller builds the optim via
    :func:`_build_optim_pair`.

    The chunk manager is built with the supplied ``rank`` /
    ``world_size`` and ``zero3_shard=True``; ``materialize_offload``
    runs against gloo's CPU collective for the ``broadcast``-style
    payload assembly. The model lives on CUDA so the optim adapters
    that follow `_build_optim_pair` can use the existing
    DeepSpeedCPUAdam plumbing without forking onto a CPU-only path.
    """
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import (
        PinnedHostMemory,
    )

    class _MixedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(32, 32, bias=True).to(torch.float16)
            self.norm = nn.LayerNorm(32).to(torch.float32)

    class _MixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.h = nn.ModuleList([_MixedLayer()])

    torch.manual_seed(0)  # identical init across ranks
    model = _MixedModel().to("cuda")

    block_spans: dict[BlockId, list[ParamId]] = {}
    for name, _p in model.named_parameters():
        if name.startswith("h."):
            idx = int(name.split(".")[1])
            block_spans.setdefault(cast(BlockId, idx), []).append(cast(ParamId, name))
    exec_order = [cast(ParamId, n) for n, _ in model.named_parameters()]
    S_chunk = 1 << 14  # plenty for the tiny mixed layer
    layout = build_layout(model, exec_order, S_chunk, block_spans)

    n_buffer = 2
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
        n_persist=0,  # everything offloaded -> sharded path
        buffer_pool=pool,
        cpu_optim=None,
        gpu_optim=None,
        device=torch.device("cuda"),
        world_size=world_size,
        rank=rank,
        zero3_shard=True,
    )
    return model, mgr, host


def _common_sharded_worker_setup(rank: int, world_size: int, tmpdir: str, tag: str):
    """Init gloo + build mixed-dtype sharded chunk_manager + optim.

    Mode-C analog of :func:`_common_worker_setup`. Returns
    ``(model, mgr, optim, host)``. Each rank takes one fwd+bwd+step
    so the optimizer state is non-trivial.
    """
    import os as _os

    import torch
    import torch.distributed as dist

    _os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("worker: CUDA not available")

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-{tag}",
        rank=rank,
        world_size=world_size,
    )

    model, mgr, host = _build_sharded_chunk_manager_mixed_dtype(rank, world_size)
    mgr.materialize_offload()
    _, _, optim = _build_optim_pair(model, mgr)
    # Take one step against a deterministic batch so the inner state
    # has real exp_avg / exp_avg_sq tensors. Identical inputs across
    # ranks; with gloo all-reduce hooks elsewhere DDP would replicate
    # grads, but here we just want non-empty state — the test bodies
    # zero state where strict cross-rank equality is needed.
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(123)
    x = torch.randn(2, 32, generator=cpu_gen).to("cuda").to(torch.float16)
    for cid in list(mgr._non_persistent_ids):
        mgr.gather(cid)
    optim.zero_grad()
    out = model.h[0].proj(x)
    out = model.h[0].norm(out.to(torch.float32))
    out.sum().backward()
    optim.step()
    return model, mgr, optim, host


def _worker_sharded_save_writes_per_rank_files(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Each rank's callback writes its own chunk_<N>_rank_<R>.pt.

    Drives the callback with a fake HF args. Verifies post-callback
    that on rank-0 all expected files exist and metadata declares
    ``protrain_save_mode="sharded"``. Both ranks write a sentinel
    ``rank<R>.done`` so the parent test can confirm liveness.
    """
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_sharded_worker_setup(
            rank, world_size, tmpdir, tag="shardsave"
        )
        try:
            output_dir = _os.path.join(tmpdir, "trainer_out")
            if rank == 0:
                _os.makedirs(output_dir, exist_ok=True)
            dist.barrier()
            ckpt_dir = _os.path.join(output_dir, "checkpoint-1")
            if rank == 0:
                _os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()

            cb = make_checkpoint_callback(save_max_bytes=DEFAULT_SAVE_MAX_BYTES)
            fake_args = mock.MagicMock(
                output_dir=output_dir,
                process_index=rank,
                world_size=world_size,
            )
            fake_state = mock.MagicMock(global_step=1)
            fake_control = mock.MagicMock()

            cb.on_save(fake_args, fake_state, fake_control, optimizer=optim)

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_save_writes_per_rank_shard_files(tmp_path):
    """2-rank gloo: each rank writes its own chunk_<N>_rank_<R>.pt files.

    Verifies the Mode-C save layout (CHECKPOINT_DESIGN_PHASE2.md §3.1):
    rank-0 writes metadata + gpu_optim.pt (none here since n_persist=0);
    every rank writes chunk_<N>_rank_<R>.pt. Metadata records
    ``protrain_save_mode="sharded"`` and a non-empty
    ``regions_per_chunk``.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_sharded_save_writes_per_rank_files,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")
    for r in range(world_size):
        assert (tmp_path / f"rank{r}.done").is_file()

    proot = tmp_path / "trainer_out" / "checkpoint-1" / PROTRAIN_OPTIM_DIRNAME
    assert (proot / METADATA_FILENAME).is_file()

    meta = json.loads((proot / METADATA_FILENAME).read_text())
    assert meta["protrain_save_mode"] == SAVE_MODE_SHARDED
    assert meta["protrain_zero3_shard"] is True
    assert meta["protrain_world_size"] == 2
    assert "regions_per_chunk" in meta
    assert meta["regions_per_chunk"], (
        "regions_per_chunk should be non-empty (mixed-dtype chunk has "
        "at least one region)"
    )

    cpu_dir = proot / CPU_OPTIM_DIRNAME
    assert cpu_dir.is_dir(), "cpu_optim/ must exist"

    # Every chunk in regions_per_chunk must have a per-rank file from
    # every rank.
    for cid in meta["regions_per_chunk"]:
        for r in range(world_size):
            shard_path = cpu_dir / f"chunk_{int(cid)}_rank_{r}.pt"
            assert shard_path.is_file(), f"missing per-rank shard {shard_path.name}"

    # No unsuffixed Mode-B-style chunk_<N>.pt files in this dir.
    for entry in cpu_dir.iterdir():
        assert "_rank_" in entry.name, (
            f"Mode-C cpu_optim/ contains a non-rank-suffixed file: {entry.name}"
        )


def _worker_sharded_metadata_contains_regions(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Save and verify ``regions_per_chunk`` matches runtime descriptors."""
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_sharded_worker_setup(
            rank, world_size, tmpdir, tag="shardmeta"
        )
        try:
            save_dir = _os.path.join(tmpdir, "save_root")
            if rank == 0:
                _os.makedirs(save_dir, exist_ok=True)
            dist.barrier()

            wrote = _save_protrain_optim_dir(
                optim,
                save_dir,
                step=1,
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                rank=rank,
                world_size=world_size,
            )
            assert wrote is True, f"rank {rank}: save returned False"
            dist.barrier()

            # Snapshot expected regions on every rank from the live
            # chunk_manager — rank-0 wrote them; non-zero ranks didn't.
            # Either way the in-memory descriptors are the source of
            # truth.
            current_regions = _build_regions_per_chunk(mgr)

            if rank == 0:
                meta_path = _os.path.join(
                    save_dir, PROTRAIN_OPTIM_DIRNAME, METADATA_FILENAME
                )
                meta = json.loads(open(meta_path).read())
                saved_regions = meta["regions_per_chunk"]
                assert set(saved_regions.keys()) == set(current_regions.keys()), (
                    f"rank 0: saved chunk-id set {set(saved_regions)} "
                    f"!= current {set(current_regions)}"
                )
                for cid in saved_regions:
                    s = saved_regions[cid]
                    c = current_regions[cid]
                    assert len(s) == len(c), (
                        f"rank 0: chunk {cid} region count diff: {len(s)} vs {len(c)}"
                    )
                    for i, (sr, cr) in enumerate(zip(s, c, strict=True)):
                        for k in (
                            "chunk_offset",
                            "region_bytes",
                            "region_bytes_padded",
                            "shard_bytes",
                            "dtype",
                        ):
                            assert sr[k] == cr[k], (
                                f"rank 0: chunk {cid} region {i} "
                                f"field {k}: saved={sr[k]!r} "
                                f"current={cr[k]!r}"
                            )

                # Multi-region invariant: the mixed-dtype layer
                # produces at least 2 regions (fp16 weights + fp32
                # layernorm). Be tolerant of different layout decisions
                # but assert at least one chunk has > 1 region so the
                # multi-region branch is genuinely exercised.
                multi_region_chunks = [
                    cid for cid, regs in saved_regions.items() if len(regs) > 1
                ]
                assert multi_region_chunks, (
                    "rank 0: expected at least one multi-region chunk "
                    f"in regions_per_chunk; got {saved_regions}"
                )

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_metadata_contains_regions_per_chunk(tmp_path):
    """metadata.json's regions_per_chunk matches runtime DtypeRegion records."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_sharded_metadata_contains_regions,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")
    for r in range(world_size):
        assert (tmp_path / f"rank{r}.done").is_file()


def _worker_sharded_load_round_trip(rank: int, world_size: int, tmpdir: str) -> None:
    """Save, mutate state, load, verify state matches pre-save snapshot."""
    import copy
    import os as _os

    import torch
    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_sharded_worker_setup(
            rank, world_size, tmpdir, tag="shardload"
        )
        try:
            save_dir = _os.path.join(tmpdir, "save_root")
            if rank == 0:
                _os.makedirs(save_dir, exist_ok=True)
            dist.barrier()

            # Force byte-identical state structure across ranks; the
            # actual values may differ per rank in Mode-C (each rank
            # owns its own slice), but zeroing keeps the test focused
            # on the load round-trip rather than on cpu-adam threading
            # noise.
            _force_identical_inner_state(optim)

            def _snap():
                snap = {}
                if optim._gpu_optim is not None:
                    snap["gpu"] = copy.deepcopy(optim._gpu_optim._optim.state_dict())
                if optim._cpu_optim is not None:
                    snap["cpu"] = {
                        cid: copy.deepcopy(inner.state_dict())
                        for cid, inner in optim._cpu_optim._optims.items()
                    }
                return snap

            pre_save = _snap()

            wrote = _save_protrain_optim_dir(
                optim,
                save_dir,
                step=1,
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                rank=rank,
                world_size=world_size,
            )
            assert wrote is True
            dist.barrier()

            # Mutate state on every rank so a no-op load would be
            # visible.
            if optim._cpu_optim is not None:
                for inner in optim._cpu_optim._optims.values():
                    for s in inner.state.values():
                        for v in s.values():
                            if isinstance(v, torch.Tensor):
                                v.add_(7.0)

            # Load: each rank reads its own per-rank shard.
            loaded = _load_protrain_optim_dir(optim, save_dir)
            assert loaded is True

            post_load = _snap()

            def _states_match(a, b) -> bool:
                if set(a) != set(b):
                    return False
                for k in a:
                    sa, sb = a[k], b[k]
                    if isinstance(sa, dict) and isinstance(sb, dict):
                        if not _states_match(sa, sb):
                            return False
                    elif isinstance(sa, torch.Tensor) and isinstance(sb, torch.Tensor):
                        if not torch.equal(sa, sb):
                            return False
                    else:
                        if sa != sb:
                            return False
                return True

            assert _states_match(post_load, pre_save), (
                f"rank {rank}: load did not restore inner state"
            )

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_load_reads_per_rank_shard_files(tmp_path):
    """2-rank gloo: each rank loads its own per-rank shard.

    Verifies the Mode-C load contract (CHECKPOINT_DESIGN_PHASE2.md
    §3.4): every rank reads ``chunk_<N>_rank_<R>.pt`` for its own
    ordinal and the resulting inner CPU optim state matches what the
    rank had pre-save.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_sharded_load_round_trip,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")
    for r in range(world_size):
        assert (tmp_path / f"rank{r}.done").is_file()


def _worker_sharded_load_rejects(
    rank: int, world_size: int, tmpdir: str, mode: str
) -> None:
    """Save, then tamper the saved metadata/files per ``mode``, expect
    RuntimeError on load.

    ``mode``:
      - "region_count": rank-0 appends a fake region to chunk-0 metadata
      - "region_dtype": rank-0 flips a region's dtype string
      - "missing_shard": rank-0 deletes rank-1's chunk-0 shard file
    """
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_sharded_worker_setup(
            rank, world_size, tmpdir, tag=f"shardrej-{mode}"
        )
        try:
            save_dir = _os.path.join(tmpdir, "save_root")
            if rank == 0:
                _os.makedirs(save_dir, exist_ok=True)
            dist.barrier()

            wrote = _save_protrain_optim_dir(
                optim,
                save_dir,
                step=1,
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                rank=rank,
                world_size=world_size,
            )
            assert wrote is True
            dist.barrier()

            # Rank-0 mutates the saved layout to provoke the failure
            # the test is checking.
            if rank == 0:
                meta_path = _os.path.join(
                    save_dir, PROTRAIN_OPTIM_DIRNAME, METADATA_FILENAME
                )
                meta = json.loads(open(meta_path).read())
                first_cid = sorted(meta["regions_per_chunk"], key=int)[0]
                if mode == "region_count":
                    # Append a fake region so the count drifts.
                    fake_region = dict(meta["regions_per_chunk"][first_cid][0])
                    meta["regions_per_chunk"][first_cid].append(fake_region)
                    open(meta_path, "w").write(json.dumps(meta))
                elif mode == "region_dtype":
                    # Flip the first region's dtype to something that
                    # won't match the runtime.
                    meta["regions_per_chunk"][first_cid][0]["dtype"] = "torch.float64"
                    open(meta_path, "w").write(json.dumps(meta))
                elif mode == "missing_shard":
                    # Delete rank-1's chunk-0 shard.
                    target_rank = 1
                    cpu_dir = _os.path.join(
                        save_dir, PROTRAIN_OPTIM_DIRNAME, CPU_OPTIM_DIRNAME
                    )
                    victim = _os.path.join(
                        cpu_dir,
                        f"chunk_{int(first_cid)}_rank_{target_rank}.pt",
                    )
                    _os.remove(victim)
                else:
                    raise ValueError(f"unknown mode {mode!r}")
            dist.barrier()

            # Both ranks attempt to load. The error mode determines which
            # rank raises:
            #   - region_count / region_dtype: every rank validates
            #     metadata first → both raise.
            #   - missing_shard: only rank-1's file is gone → only
            #     rank-1 raises; rank-0 loads successfully.
            try:
                _load_protrain_optim_dir(optim, save_dir)
            except RuntimeError as exc:
                msg = str(exc)
                with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                    f.write(msg)
            else:
                # Some ranks legitimately don't error in missing_shard
                # mode (only rank-1 does). Mark a sentinel so we can
                # tell "load succeeded on this rank" from "load
                # silently skipped".
                with open(_os.path.join(tmpdir, f"rank{rank}.no_raise"), "w") as f:
                    f.write("load did not raise on this rank")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                f.write(str(exc))
            return
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def _spawn_sharded_load_rejects(tmp_path, mode: str) -> list[str]:
    """Run the ``_worker_sharded_load_rejects`` body and return ``caught`` msgs."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    try:
        mp.spawn(
            _worker_sharded_load_rejects,
            args=(world_size, str(tmp_path), mode),
            nprocs=world_size,
            join=True,
        )
    except Exception:
        # mp.spawn re-raises any worker exception; the workers also
        # write sentinel files so the parent test can inspect details.
        pass

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"unexpected worker errors:\n{bodies}")

    caught_files = sorted(tmp_path.glob("rank*.caught"))
    return [f.read_text() for f in caught_files]


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_load_rejects_region_count_mismatch(tmp_path):
    """Tamper saved metadata to add a fake region → load hard-errors.

    A region-count drift means the saved per-rank shards won't match
    the rebuilt shard_param. Loader must raise pointing at the
    differing chunk + region index instead of letting torch's
    load_state_dict crash with a shape error.
    """
    msgs = _spawn_sharded_load_rejects(tmp_path, mode="region_count")
    assert msgs, "no rank caught the region-count-mismatch RuntimeError"
    assert any("region count mismatch" in m for m in msgs), (
        f"region count error did not name the mismatch: {msgs}"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_load_rejects_region_dtype_mismatch(tmp_path):
    """Tamper saved region dtype → load hard-errors naming the field."""
    msgs = _spawn_sharded_load_rejects(tmp_path, mode="region_dtype")
    assert msgs, "no rank caught the region-dtype-mismatch RuntimeError"
    assert any("field 'dtype'" in m for m in msgs), (
        f"region dtype error did not name the field: {msgs}"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_load_rejects_missing_rank_shard(tmp_path):
    """Delete a per-rank shard → that rank's load hard-errors naming the file.

    The missing file must be flagged by name so an operator reading
    the trace can map it to the worker that failed to write.
    """
    msgs = _spawn_sharded_load_rejects(tmp_path, mode="missing_shard")
    assert msgs, "no rank caught the missing-shard RuntimeError"
    assert any("missing rank shard" in m and "rank_1.pt" in m for m in msgs), (
        f"missing-shard error did not name the file: {msgs}"
    )


# ---------------------------------------------------------------------------
# Mode-C regression tests for the verify-gate and inner-size-gate fixes
# ---------------------------------------------------------------------------


def _worker_sharded_verify_replicated_is_noop(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Mode-C with ``verify_replicated=True`` must NOT call the cross-rank
    state-equality check. In Mode-C each rank's inner state is sharded
    per-rank, so the check would falsely raise. The schema documents
    "Has no effect on single-rank or ZeRO-3 sharded runs".

    We patch ``_verify_replicated_state_across_ranks`` to write a
    sentinel file on entry; the parent test asserts the file was
    never created.
    """
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_sharded_worker_setup(
            rank, world_size, tmpdir, tag="verifynoop"
        )
        try:
            output_dir = _os.path.join(tmpdir, "trainer_out")
            if rank == 0:
                _os.makedirs(output_dir, exist_ok=True)
            dist.barrier()
            ckpt_dir = _os.path.join(output_dir, "checkpoint-1")
            if rank == 0:
                _os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()

            # Sentinel: any call raises so the fixture sees the
            # symptom even if mp.spawn swallows the patch context.
            sentinel_path = _os.path.join(tmpdir, f"verify_called_rank{rank}")

            def _tripwire(*args, **kwargs):
                with open(sentinel_path, "w") as f:
                    f.write("called")
                raise RuntimeError("verify_replicated should be a no-op in Mode-C")

            cb = make_checkpoint_callback(
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                verify_replicated=True,
            )
            fake_args = mock.MagicMock(
                output_dir=output_dir,
                process_index=rank,
                world_size=world_size,
            )
            fake_state = mock.MagicMock(global_step=1)
            fake_control = mock.MagicMock()

            with mock.patch(
                "axolotl.integrations.protrain.api.checkpoint."
                "_verify_replicated_state_across_ranks",
                side_effect=_tripwire,
            ):
                cb.on_save(fake_args, fake_state, fake_control, optimizer=optim)

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_save_with_verify_flag_skips_cross_rank_check(tmp_path):
    """Mode-C + ``verify_replicated=True`` → save proceeds; the cross-rank
    hash check is NOT invoked.

    Regression: prior gate checked only ``world_size > 1``, so a Mode-C
    user who left the Mode-B verify flag enabled would see a spurious
    RuntimeError on save (per-rank shards intentionally diverge).
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_sharded_verify_replicated_is_noop,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")

    for r in range(world_size):
        assert (tmp_path / f"rank{r}.done").is_file(), (
            f"rank {r} did not reach post-save sentinel"
        )
        assert not (tmp_path / f"verify_called_rank{r}").exists(), (
            f"_verify_replicated_state_across_ranks fired on rank {r} "
            f"in Mode-C — gate must exclude zero3_shard"
        )

    proot = tmp_path / "trainer_out" / "checkpoint-1" / PROTRAIN_OPTIM_DIRNAME
    assert (proot / METADATA_FILENAME).is_file(), (
        "save did not produce a Mode-C checkpoint"
    )


def _worker_sharded_inverted_gate_writes_all_shards(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Mode-C: rank-0 estimate fits; rank-1 estimate would trip the cap
    if recomputed locally inside ``_save_protrain_optim_dir``. After
    rank-0's broadcast says proceed, every rank must still write its
    shards — the inner per-rank gate must be suppressed via
    ``_skip_size_gate=True``.

    Regression: prior code re-ran the gate per-rank, so rank-1 would
    silently return False without writing ``chunk_<N>_rank_1.pt`` and
    leave a partial Mode-C checkpoint.
    """
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_sharded_worker_setup(
            rank, world_size, tmpdir, tag="invgate"
        )
        try:
            output_dir = _os.path.join(tmpdir, "trainer_out")
            if rank == 0:
                _os.makedirs(output_dir, exist_ok=True)
            dist.barrier()
            ckpt_dir = _os.path.join(output_dir, "checkpoint-1")
            if rank == 0:
                _os.makedirs(ckpt_dir, exist_ok=True)
            dist.barrier()

            small_threshold = 64
            # Per-rank patch: rank-0's local estimate fits (skip=False
            # broadcast), but rank-1's would trip the cap if the inner
            # gate fired. With the fix the inner gate is suppressed by
            # the callback so rank-1 still writes.
            per_rank_estimate = 0 if rank == 0 else (small_threshold + 1)

            cb = make_checkpoint_callback(save_max_bytes=small_threshold)
            fake_args = mock.MagicMock(
                output_dir=output_dir,
                process_index=rank,
                world_size=world_size,
            )
            fake_state = mock.MagicMock(global_step=1)
            fake_control = mock.MagicMock()

            with mock.patch(
                "axolotl.integrations.protrain.api.checkpoint."
                "_estimate_optim_state_bytes",
                return_value=per_rank_estimate,
            ):
                cb.on_save(fake_args, fake_state, fake_control, optimizer=optim)

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_save_inner_gate_does_not_drop_rank_n_shards(tmp_path):
    """Mode-C: rank-N's local estimate must NOT independently trip the
    inner save-size gate after rank-0's broadcast said proceed.

    Without the ``_skip_size_gate=True`` plumbing in the callback,
    rank-1 would silently bail inside ``_save_protrain_optim_dir`` and
    the on-disk Mode-C checkpoint would be missing every
    ``chunk_<N>_rank_1.pt`` shard.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    mp.spawn(
        _worker_sharded_inverted_gate_writes_all_shards,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"worker errors:\n{bodies}")

    for r in range(world_size):
        assert (tmp_path / f"rank{r}.done").is_file(), (
            f"rank {r} did not reach post-save sentinel"
        )

    proot = tmp_path / "trainer_out" / "checkpoint-1" / PROTRAIN_OPTIM_DIRNAME
    assert (proot / METADATA_FILENAME).is_file(), "metadata missing"
    meta = json.loads((proot / METADATA_FILENAME).read_text())
    assert meta["protrain_save_mode"] == SAVE_MODE_SHARDED

    cpu_dir = proot / CPU_OPTIM_DIRNAME
    assert cpu_dir.is_dir(), "cpu_optim/ missing"

    # Every chunk in regions_per_chunk must have a per-rank file from
    # *every* rank. The bug this guards against: rank-1 tripping the
    # inner gate and silently skipping its writes.
    for cid in meta["regions_per_chunk"]:
        for r in range(world_size):
            shard_path = cpu_dir / f"chunk_{int(cid)}_rank_{r}.pt"
            assert shard_path.is_file(), (
                f"missing per-rank shard {shard_path.name} — inner "
                f"size-gate likely fired on rank {r} after rank-0's "
                f"broadcast said proceed"
            )


# ---------------------------------------------------------------------------
# Mode-C lockstep-failure regressions (Findings 1, 2, 3)
# ---------------------------------------------------------------------------
# Pre-fix, three deadlock/silent-accept paths existed in the Mode-C save/load
# barriers:
#   F1: rank-0 raises mid-write on save -> non-zero ranks block forever on
#       the trailing barrier (NCCL/gloo barriers have no timeout).
#   F2: a single rank's shard load raises on resume -> surviving ranks
#       block on the load hook's trailing barrier.
#   F3: extra files left behind from a higher-world_size save in cpu_optim/
#       were silently accepted; only the per-rank expected file existence
#       was checked.
# Fix: status broadcast / all_reduce around the barriers + Mode-B-style
# stray-file rejection mirroring CHUNK_FILE_RE on the Mode-C path.


def _worker_sharded_save_rank0_failure_lockstep(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Rank-0 fails mid-write during ``_save_protrain_optim_dir``; every
    rank must raise (no deadlock on the post-rank-0-write barrier).

    The forced failure: monkey-patch ``json.dump`` on rank-0 only so the
    metadata write raises a synthetic ``RuntimeError``. Without the
    Finding-1 fix, rank-1 would block forever on the barrier inside
    ``_save_protrain_optim_dir``; with the fix, rank-0's status flag is
    broadcast in a ``finally`` block and rank-1 raises a synthetic
    "rank 0 failed" error.
    """
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_sharded_worker_setup(
            rank, world_size, tmpdir, tag="f1save"
        )
        try:
            save_dir = _os.path.join(tmpdir, "save_root")
            if rank == 0:
                _os.makedirs(save_dir, exist_ok=True)
            dist.barrier()

            import json as _json

            class _BoomError(RuntimeError):
                pass

            real_dump = _json.dump

            def _maybe_boom(obj, fp, *args, **kwargs):
                if rank == 0:
                    raise _BoomError("synthetic ENOSPC during metadata write")
                return real_dump(obj, fp, *args, **kwargs)

            try:
                with mock.patch("json.dump", side_effect=_maybe_boom):
                    _save_protrain_optim_dir(
                        optim,
                        save_dir,
                        step=1,
                        save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                        rank=rank,
                        world_size=world_size,
                    )
            except RuntimeError as exc:
                with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                    f.write(f"{type(exc).__name__}: {exc}")
            else:
                with open(_os.path.join(tmpdir, f"rank{rank}.no_raise"), "w") as f:
                    f.write("save did not raise on this rank")

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                f.write(f"{type(exc).__name__}: {exc}")
            return
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_save_rank0_failure_propagates_lockstep(tmp_path):
    """Mode-C: rank-0 raises mid-write -> every rank raises (no deadlock).

    Regression for Finding 1. Pre-fix, rank-1 would block forever on the
    barrier between the rank-0 metadata write and the per-rank shard
    write. With the fix, rank-0's failure is broadcast as a status flag
    before the barrier so every rank raises in lockstep.

    Liveness witness: ``mp.spawn`` joins. If either rank deadlocked, the
    spawn would hang and pytest's per-test timeout would fail the test.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    try:
        mp.spawn(
            _worker_sharded_save_rank0_failure_lockstep,
            args=(world_size, str(tmp_path)),
            nprocs=world_size,
            join=True,
        )
    except Exception:
        # mp.spawn re-raises worker exceptions; the workers also write
        # caught/err sentinels we inspect below.
        pass

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"unexpected worker errors:\n{bodies}")

    # Every rank must have caught a RuntimeError (either rank-0's
    # synthetic _BoomError or the synthetic "rank 0 failed" raised on
    # rank-1 after the status broadcast).
    caught = sorted(tmp_path.glob("rank*.caught"))
    assert len(caught) == world_size, (
        f"expected every rank to raise; got {[c.name for c in caught]}. "
        f"no_raise sentinels: {[p.name for p in tmp_path.glob('rank*.no_raise')]}"
    )

    bodies = [c.read_text() for c in caught]
    # rank-0's exception is the original synthetic ENOSPC; non-rank-0
    # ranks see the synthetic "rank 0 failed" error.
    assert any("ENOSPC" in b for b in bodies), (
        f"rank-0's original exception was lost: {bodies}"
    )
    assert any("rank 0 failed" in b for b in bodies), (
        f"non-rank-0 ranks did not synthesize the lockstep error: {bodies}"
    )


def _worker_sharded_load_rank_failure_lockstep(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Save normally, then corrupt rank-1's shard so its torch.load raises;
    every rank must raise (no deadlock on the trailing load barrier).

    The corruption: rank-0 truncates rank-1's chunk-0 shard to a few
    junk bytes after the normal save. On load, rank-1's torch.load
    raises an UnpicklingError; rank-0's load would otherwise succeed
    and block on the trailing barrier — with the fix, the all-reduce
    SUM of statuses raises on rank-0 too.
    """
    import os as _os

    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_sharded_worker_setup(
            rank, world_size, tmpdir, tag="f2load"
        )
        try:
            save_dir = _os.path.join(tmpdir, "save_root")
            if rank == 0:
                _os.makedirs(save_dir, exist_ok=True)
            dist.barrier()

            wrote = _save_protrain_optim_dir(
                optim,
                save_dir,
                step=1,
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                rank=rank,
                world_size=world_size,
            )
            assert wrote is True
            dist.barrier()

            # Rank-0 corrupts rank-1's shard so rank-1's torch.load
            # raises while rank-0's would succeed.
            if rank == 0:
                cpu_dir = _os.path.join(
                    save_dir, PROTRAIN_OPTIM_DIRNAME, CPU_OPTIM_DIRNAME
                )
                # Find any chunk-id; corrupt rank 1's file.
                victim_name = None
                for name in _os.listdir(cpu_dir):
                    if name.endswith("_rank_1.pt"):
                        victim_name = name
                        break
                assert victim_name is not None, "no rank-1 shard found to corrupt"
                with open(_os.path.join(cpu_dir, victim_name), "wb") as f:
                    f.write(b"\x00garbage_not_a_pickle\x00")
            dist.barrier()

            try:
                _load_protrain_optim_dir(optim, save_dir)
            except Exception as exc:
                with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                    f.write(f"{type(exc).__name__}: {exc}")
            else:
                with open(_os.path.join(tmpdir, f"rank{rank}.no_raise"), "w") as f:
                    f.write("load did not raise on this rank")

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        if isinstance(exc, (RuntimeError, Exception)):
            with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                f.write(f"{type(exc).__name__}: {exc}")
            return
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_load_single_rank_failure_propagates_lockstep(tmp_path):
    """Mode-C: rank-1's shard is corrupt -> every rank raises (no deadlock).

    Regression for Finding 2. Pre-fix, rank-0 would silently load and
    block forever on the trailing load barrier. With the fix, the
    all-reduce SUM of per-rank load statuses raises on every rank.

    Liveness witness: ``mp.spawn`` joins. If either rank deadlocked, the
    spawn would hang and the per-test timeout would fail the test.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    try:
        mp.spawn(
            _worker_sharded_load_rank_failure_lockstep,
            args=(world_size, str(tmp_path)),
            nprocs=world_size,
            join=True,
        )
    except Exception:
        pass

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"unexpected worker errors:\n{bodies}")

    caught = sorted(tmp_path.glob("rank*.caught"))
    assert len(caught) == world_size, (
        f"expected every rank to raise; got {[c.name for c in caught]}. "
        f"no_raise sentinels: {[p.name for p in tmp_path.glob('rank*.no_raise')]}"
    )
    bodies = [c.read_text() for c in caught]
    # At least one rank surfaces the synthetic "rank(s) failed during the
    # per-rank phase" error from the all_reduce path; the originating
    # rank surfaces the real torch.load error.
    assert any("per-rank phase" in b or "rank(s) failed" in b for b in bodies), (
        f"no rank reported the lockstep all_reduce error: {bodies}"
    )


def _worker_sharded_load_rejects_stray_file(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """Save normally, drop a stray ``chunk_X_rank_99.pt`` into cpu_optim/,
    then assert load rejects on every rank.

    Mirror of Mode-B's ``CHUNK_FILE_RE`` enforcement. Pre-fix, Mode-C
    silently accepted extras (e.g. left-over shards from a higher-
    world_size save). Post-fix, the loader enumerates cpu_optim/ and
    rejects anything that doesn't match ``CHUNK_SHARD_FILE_RE``.
    """
    import os as _os

    import torch
    import torch.distributed as dist

    try:
        model, mgr, optim, host = _common_sharded_worker_setup(
            rank, world_size, tmpdir, tag="f3stray"
        )
        try:
            save_dir = _os.path.join(tmpdir, "save_root")
            if rank == 0:
                _os.makedirs(save_dir, exist_ok=True)
            dist.barrier()

            wrote = _save_protrain_optim_dir(
                optim,
                save_dir,
                step=1,
                save_max_bytes=DEFAULT_SAVE_MAX_BYTES,
                rank=rank,
                world_size=world_size,
            )
            assert wrote is True
            dist.barrier()

            # Rank-0 plants a stray shard from a phantom higher-
            # world_size save. The filename matches the shape of a
            # legitimate Mode-C shard but the rank ordinal is impossible
            # for this 2-rank load.
            if rank == 0:
                cpu_dir = _os.path.join(
                    save_dir, PROTRAIN_OPTIM_DIRNAME, CPU_OPTIM_DIRNAME
                )
                # Pick any chunk id from the on-disk shards.
                some_cid = None
                for name in _os.listdir(cpu_dir):
                    if name.endswith("_rank_0.pt"):
                        some_cid = name.split("_")[1]
                        break
                assert some_cid is not None, "no rank-0 shard found to clone"
                stray = _os.path.join(cpu_dir, f"chunk_{int(some_cid)}_rank_99.pt")
                # Make it a valid pickle so the loader can't reject on
                # corruption — we want the regex check to be the gate,
                # not torch.load.
                torch.save({"state": {}, "param_groups": []}, stray)
            dist.barrier()

            # Every rank attempts the load. With the fix, every rank's
            # listdir scan trips on the stray file and raises BEFORE
            # any torch.load runs. The all_reduce then propagates so the
            # cluster fails in lockstep.
            try:
                _load_protrain_optim_dir(optim, save_dir)
            except Exception as exc:
                with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                    f.write(f"{type(exc).__name__}: {exc}")
            else:
                with open(_os.path.join(tmpdir, f"rank{rank}.no_raise"), "w") as f:
                    f.write("load did not raise")

            with open(_os.path.join(tmpdir, f"rank{rank}.done"), "w") as f:
                f.write("ok")
        finally:
            _teardown_mgr(mgr, optim)
            host.close()
            del model, optim, mgr
    except Exception as exc:
        if isinstance(exc, (RuntimeError, Exception)):
            with open(_os.path.join(tmpdir, f"rank{rank}.caught"), "w") as f:
                f.write(f"{type(exc).__name__}: {exc}")
            return
        import traceback as _tb

        with open(_os.path.join(tmpdir, f"rank{rank}.err"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
            _tb.print_exc(file=f)
        raise
    finally:
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.gpu
@pytest.mark.slow
def test_sharded_load_rejects_stray_file_in_cpu_optim(tmp_path):
    """Mode-C: a stray ``chunk_X_rank_99.pt`` file makes load hard-error.

    Regression for Finding 3. Mode-B already rejects unknown files via
    ``CHUNK_FILE_RE``; Mode-C must mirror with ``CHUNK_SHARD_FILE_RE``.
    Pre-fix, the stray file was silently tolerated because Mode-C only
    checked "my rank's expected files exist".
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    try:
        mp.spawn(
            _worker_sharded_load_rejects_stray_file,
            args=(world_size, str(tmp_path)),
            nprocs=world_size,
            join=True,
        )
    except Exception:
        pass

    err_files = list(tmp_path.glob("rank*.err"))
    if err_files:
        bodies = "\n---\n".join(f.read_text() for f in err_files)
        pytest.fail(f"unexpected worker errors:\n{bodies}")

    caught = sorted(tmp_path.glob("rank*.caught"))
    assert len(caught) == world_size, (
        f"expected every rank to raise; got {[c.name for c in caught]}. "
        f"no_raise sentinels: {[p.name for p in tmp_path.glob('rank*.no_raise')]}"
    )
    bodies = [c.read_text() for c in caught]
    assert any("unexpected file" in b and "rank_99.pt" in b for b in bodies), (
        f"stray-file rejection error did not name the offending file: {bodies}"
    )
