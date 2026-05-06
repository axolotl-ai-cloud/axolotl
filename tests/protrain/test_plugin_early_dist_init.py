"""Tests for ``plugin._early_init_dist_for_nccl`` and the
``post_model_load`` early-init wiring (Item 6 — Preflight NCCL
measurement).

The helper brings ``torch.distributed`` up via
``init_process_group(backend="nccl")`` *before* the model wrapper runs,
so the profiler trace captures real NCCL gather/reduce times on the
live process group instead of recording empty tables. Real NCCL collectives
require a multi-rank rendezvous, so these tests exercise the *wiring* —
when the helper fires, what env it consults, when it skips — with
``torch.distributed.init_process_group`` mocked out. Measurement
correctness itself is covered by ``scripts/protrain/measure_nccl.py``
under torchrun.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@contextmanager
def _multi_rank_env(world_size: int = 2, local_rank: int = 0, rank: int = 0):
    """Set the env vars torchrun / Accelerate would set, restore on exit."""
    keys = {
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(local_rank),
        "RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update(keys)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def _single_rank_env():
    """Clear all torchrun env so we look like a non-launcher process."""
    keys = ("WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _FakeCfg:
    """Stand-in for the merged plugin cfg DictDefault."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _patch_dist_module(*, available=True, initialized=False, world_size=2):
    """Patch ``torch.distributed`` to a non-initialised state we can drive.

    Returns a list of ``unittest.mock`` patcher contexts. The caller
    starts/stops them and inspects the live mocks via the ``patcher.start()``
    return values (collected from ``_start_all``) — once stopped the
    attribute reverts to the real function and ``.called`` is gone.
    """
    import torch.distributed as dist

    return [
        patch.object(dist, "is_available", return_value=available),
        patch.object(dist, "is_initialized", return_value=initialized),
        patch.object(dist, "get_world_size", return_value=world_size),
        patch.object(dist, "init_process_group"),
    ]


def _patch_cuda(*, available=True):
    """Patch ``torch.cuda.is_available`` + ``set_device`` for early-init tests."""
    import torch

    return [
        patch.object(torch.cuda, "is_available", return_value=available),
        patch.object(torch.cuda, "set_device"),
    ]


def _start_all(patches):
    return [p.start() for p in patches]


def _stop_all(patches):
    for p in patches:
        p.stop()


# ---------------------------------------------------------------------------
# _early_init_dist_for_nccl — direct unit coverage
# ---------------------------------------------------------------------------


def test_early_init_skips_on_single_rank():
    """WORLD_SIZE unset / 1 → no init attempt, returns 1."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _early_init_dist_for_nccl

    with _single_rank_env():
        # Even if the user accidentally toggles a fake-initialised dist,
        # WORLD_SIZE=1 short-circuits before we touch torch.distributed.
        patches = _patch_dist_module(initialized=False, world_size=1)
        mocks = _start_all(patches)
        init_pg_mock = mocks[3]  # init_process_group is index 3
        try:
            result = _early_init_dist_for_nccl(_FakeCfg())
            assert not init_pg_mock.called
        finally:
            _stop_all(patches)

    assert result == 1


def test_early_init_invokes_init_process_group_when_multi_rank():
    """WORLD_SIZE=4, dist not init, default backend → call init_process_group(nccl)."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _early_init_dist_for_nccl

    cfg = _FakeCfg()  # ddp_backend unset

    with _multi_rank_env(world_size=4):
        patches = _patch_dist_module(initialized=False, world_size=4) + _patch_cuda(
            available=True
        )
        mocks = _start_all(patches)
        init_pg_mock = mocks[3]
        try:
            result = _early_init_dist_for_nccl(cfg)
            assert init_pg_mock.called
            # We must have asked for nccl explicitly (not allowed to drift).
            call_kwargs = init_pg_mock.call_args.kwargs
            assert call_kwargs.get("backend") == "nccl"
        finally:
            _stop_all(patches)

    assert result == 4


def test_early_init_idempotent_when_already_initialized():
    """If dist.is_initialized() is True on entry, do not re-init."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _early_init_dist_for_nccl

    with _multi_rank_env(world_size=2):
        patches = _patch_dist_module(initialized=True, world_size=2) + _patch_cuda(
            available=True
        )
        mocks = _start_all(patches)
        init_pg_mock = mocks[3]
        try:
            result = _early_init_dist_for_nccl(_FakeCfg())
            assert not init_pg_mock.called
        finally:
            _stop_all(patches)

    # Live world size returned.
    assert result == 2


def test_early_init_skips_on_custom_ddp_backend():
    """A non-default ``cfg.ddp_backend`` defers init to Accelerate / HF."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _early_init_dist_for_nccl

    cfg = _FakeCfg(ddp_backend="gloo")

    with _multi_rank_env(world_size=4):
        patches = _patch_dist_module(initialized=False, world_size=4) + _patch_cuda(
            available=True
        )
        mocks = _start_all(patches)
        init_pg_mock = mocks[3]
        try:
            result = _early_init_dist_for_nccl(cfg)
            assert not init_pg_mock.called
        finally:
            _stop_all(patches)

    assert result == 1  # treated as single-rank for the early-init path


def test_early_init_accepts_explicit_nccl_backend():
    """``ddp_backend='nccl'`` matches our default — proceed with init."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _early_init_dist_for_nccl

    cfg = _FakeCfg(ddp_backend="nccl")

    with _multi_rank_env(world_size=2):
        patches = _patch_dist_module(initialized=False, world_size=2) + _patch_cuda(
            available=True
        )
        mocks = _start_all(patches)
        init_pg_mock = mocks[3]
        try:
            result = _early_init_dist_for_nccl(cfg)
            assert init_pg_mock.called
        finally:
            _stop_all(patches)

    assert result == 2


def test_early_init_skips_when_local_rank_unset():
    """``WORLD_SIZE`` set but ``LOCAL_RANK`` missing → bail (not under launcher)."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _early_init_dist_for_nccl

    keys = ("WORLD_SIZE", "LOCAL_RANK", "RANK", "MASTER_ADDR", "MASTER_PORT")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ["WORLD_SIZE"] = "4"
        # Deliberately leave LOCAL_RANK / RANK / MASTER_* unset.

        patches = _patch_dist_module(initialized=False, world_size=4) + _patch_cuda(
            available=True
        )
        mocks = _start_all(patches)
        init_pg_mock = mocks[3]
        try:
            result = _early_init_dist_for_nccl(_FakeCfg())
            assert not init_pg_mock.called
        finally:
            _stop_all(patches)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    assert result == 1


def test_early_init_skips_without_cuda():
    """No CUDA → cannot bring up NCCL; defer to late-bind path."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _early_init_dist_for_nccl

    with _multi_rank_env(world_size=2):
        patches = _patch_dist_module(initialized=False, world_size=2) + _patch_cuda(
            available=False
        )
        mocks = _start_all(patches)
        init_pg_mock = mocks[3]
        try:
            result = _early_init_dist_for_nccl(_FakeCfg())
            assert not init_pg_mock.called
        finally:
            _stop_all(patches)

    assert result == 1


def test_early_init_swallows_init_failure():
    """If ``init_process_group`` raises, fall back gracefully without crashing."""
    pytest.importorskip("torch")

    import torch.distributed as dist

    from axolotl.integrations.protrain.plugin import _early_init_dist_for_nccl

    with _multi_rank_env(world_size=2):
        patches = [
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=False),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(
                dist,
                "init_process_group",
                side_effect=RuntimeError("rendezvous timeout"),
            ),
        ] + _patch_cuda(available=True)
        _start_all(patches)
        try:
            result = _early_init_dist_for_nccl(_FakeCfg())
        finally:
            _stop_all(patches)

    assert result == 1


# ---------------------------------------------------------------------------
# post_model_load wiring — verify the helper is invoked at the right moment
# ---------------------------------------------------------------------------


def test_post_model_load_calls_early_init_before_wrapper():
    """``post_model_load`` must call ``_early_init_dist_for_nccl`` *before*
    invoking the wrapper, so the wrapper's profiler trace sees the live PG.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torch.cuda")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("post_model_load builds a HardwareProfile from a real CUDA device.")

    from axolotl.integrations.protrain import plugin as plugin_mod

    # Track call ordering: early-init then wrapper.
    call_log: list[str] = []

    def fake_early_init(cfg):
        call_log.append("early_init")
        return 4  # pretend WORLD_SIZE=4

    def fake_wrapper(*args, **kwargs):
        call_log.append("wrapper")
        # Build a minimal fake WrappedModel — only the attrs
        # post_model_load reads (search_result.cfg, chunk_manager,
        # _trace) need to exist.
        from types import SimpleNamespace

        return SimpleNamespace(
            search_result=SimpleNamespace(
                cfg=SimpleNamespace(n_persist=1, n_buffer=1, n_swap=0, n_checkpoint=0),
                block_map={},
            ),
            chunk_manager=SimpleNamespace(
                layout=SimpleNamespace(N_chunk=2),
                zero3_shard=False,
            ),
        )

    cfg = _FakeCfg(
        protrain_auto_memory=True,
        plugins=["axolotl.integrations.protrain.ProTrainPlugin"],
        micro_batch_size=1,
        sequence_len=128,
        protrain_capacity_bytes=None,
        protrain_cpu_capacity_bytes=None,
        protrain_cache_dir=None,
        protrain_force_all_persistent=True,
        protrain_n_persist_override=None,
        protrain_n_buffer_override=None,
        protrain_n_swap_override=None,
        protrain_n_checkpoint_override=None,
        protrain_zero3_shard=None,
        protrain_auto_mode=False,
    )
    fake_model = torch.nn.Linear(4, 4)

    patches = [
        patch.object(
            plugin_mod, "_early_init_dist_for_nccl", side_effect=fake_early_init
        ),
        patch(
            "axolotl.integrations.protrain.api.protrain_model_wrapper",
            side_effect=fake_wrapper,
        ),
    ]
    _start_all(patches)
    try:
        plugin_mod.ProTrainPlugin().post_model_load(cfg, fake_model)
    finally:
        _stop_all(patches)

    assert call_log == ["early_init", "wrapper"], (
        f"early init must precede wrapper; saw {call_log!r}"
    )
    # The wrapper handle was stashed back on cfg as expected.
    assert getattr(cfg, "_protrain_wrapped", None) is not None


def test_post_model_load_idempotent_when_already_wrapped():
    """If ``cfg._protrain_wrapped`` is already set, skip both init + wrap."""
    pytest.importorskip("torch")

    from types import SimpleNamespace

    import torch

    if not torch.cuda.is_available():
        pytest.skip("post_model_load builds a HardwareProfile from a real CUDA device.")

    from axolotl.integrations.protrain import plugin as plugin_mod

    fake_model = torch.nn.Linear(4, 4)
    # The idempotency guard checks ``existing._protrain_wrapped.model is
    # model``: only the SAME model instance reuses the cached wrapper.
    # Make the sentinel match that contract (a namespace carrying the
    # incoming model under ``.model``) so the test exercises the
    # same-model fast-path. A bare ``object()`` would trigger the
    # different-model warn-and-rebuild branch instead.
    sentinel = SimpleNamespace(model=fake_model)
    cfg = _FakeCfg(
        protrain_auto_memory=True,
        plugins=["axolotl.integrations.protrain.ProTrainPlugin"],
        _protrain_wrapped=sentinel,
    )

    early_init_calls = []
    wrapper_calls = []

    patches = [
        patch.object(
            plugin_mod,
            "_early_init_dist_for_nccl",
            side_effect=lambda c: early_init_calls.append(c) or 1,
        ),
        patch(
            "axolotl.integrations.protrain.api.protrain_model_wrapper",
            side_effect=lambda *a, **kw: wrapper_calls.append((a, kw)),
        ),
    ]
    _start_all(patches)
    try:
        plugin_mod.ProTrainPlugin().post_model_load(cfg, fake_model)
    finally:
        _stop_all(patches)

    assert early_init_calls == [], "idempotent path must not re-init dist"
    assert wrapper_calls == [], "idempotent path must not re-run the wrapper"
    # The pre-existing wrapped reference is preserved.
    assert cfg._protrain_wrapped is sentinel


def test_post_model_load_skips_when_plugin_inactive():
    """Plugin off → no early init, no wrap, no crash."""
    pytest.importorskip("torch")

    import torch

    from axolotl.integrations.protrain import plugin as plugin_mod

    # protrain_auto_memory False → _is_plugin_active returns False.
    cfg = _FakeCfg(
        protrain_auto_memory=False,
        plugins=["axolotl.integrations.protrain.ProTrainPlugin"],
    )
    fake_model = torch.nn.Linear(4, 4)

    early_init_calls = []
    patches = [
        patch.object(
            plugin_mod,
            "_early_init_dist_for_nccl",
            side_effect=lambda c: early_init_calls.append(c) or 1,
        ),
    ]
    _start_all(patches)
    try:
        plugin_mod.ProTrainPlugin().post_model_load(cfg, fake_model)
    finally:
        _stop_all(patches)

    assert early_init_calls == []
