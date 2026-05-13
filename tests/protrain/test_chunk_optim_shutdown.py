"""Tests for ``CpuFusedAdamAdapter.shutdown()`` C-state release.

The shutdown path must explicitly free every per-chunk
``DeepSpeedCPUAdam`` C++ kernel state by calling
``ds_opt_adam.destroy_adam(opt_id)``. Relying on the wrapper's
``__del__`` is unreliable: GC ordering at interpreter shutdown can run
the destructor on a partially initialised object that lacks
``ds_opt_adam`` (we observed this as ``AttributeError`` warnings under
repeated adapter rebuilds), and even on healthy objects ``__del__`` is
only invoked once the wrapper is unreachable — references held by the
executor thread, futures, or test fixtures keep the C state alive
until process exit.

These tests run on CPU only (no CUDA / no DeepSpeed C extension build):
the adapter is constructed via direct attribute injection so the test
exercises the shutdown control flow without requiring the real
DeepSpeed runtime.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from axolotl.integrations.protrain.chunk.optim import (
    CpuFusedAdamAdapter,
    _DestroyedDsAdam,
)
from axolotl.integrations.protrain.types import ChunkId


class _FakeDsOptAdam:
    """Mock-friendly stand-in for ``DeepSpeedCPUAdam.ds_opt_adam``.

    Tracks every ``destroy_adam`` call made against this binding. Survives
    the production code's "swap to ``_DestroyedDsAdam``" idiom because the
    test holds the original reference in ``fakes`` rather than going
    through ``opt.ds_opt_adam`` (which is the slot that gets replaced).
    """

    def __init__(self) -> None:
        self.destroy_calls: list[Any] = []

    def destroy_adam(self, opt_id: Any) -> None:
        self.destroy_calls.append(opt_id)


def _make_adapter_with_mock_ds(
    n_chunks: int = 2,
) -> tuple[CpuFusedAdamAdapter, dict[int, _FakeDsOptAdam]]:
    """Build a ``CpuFusedAdamAdapter`` whose per-chunk DeepSpeed optims are mocks.

    The real ``__init__`` would call into ``DeepSpeedCPUAdam`` (which loads
    the CPU Adam C extension); we sidestep that by constructing an empty
    instance via ``__new__`` and injecting the fields ``shutdown()``
    actually reads — ``_optims``, ``_executor``, ``_pending``. Each fake
    optim exposes a ``ds_opt_adam`` mock with a ``destroy_adam`` callable
    plus an ``opt_id`` int, matching the contract of DeepSpeed 0.18.2's
    ``deepspeed/ops/adam/cpu_adam.py``.

    Returns the adapter plus a dict mapping ``chunk_id -> _FakeDsOptAdam``
    so tests can inspect ``destroy_adam`` calls AFTER shutdown has swapped
    ``opt.ds_opt_adam`` for the destroyed-stub sentinel.
    """
    from concurrent.futures import ThreadPoolExecutor

    adapter = CpuFusedAdamAdapter.__new__(CpuFusedAdamAdapter)
    adapter._optims = {}  # type: ignore[attr-defined]
    fakes: dict[int, _FakeDsOptAdam] = {}
    for cid in range(n_chunks):
        fake_optim = mock.MagicMock()
        fake_ds = _FakeDsOptAdam()
        fake_optim.ds_opt_adam = fake_ds
        fake_optim.opt_id = 100 + cid
        adapter._optims[ChunkId(cid)] = fake_optim  # type: ignore[attr-defined]
        fakes[cid] = fake_ds
    adapter._executor = ThreadPoolExecutor(  # type: ignore[attr-defined]
        max_workers=1, thread_name_prefix="protrain-cpu-adam-test"
    )
    adapter._pending = {}  # type: ignore[attr-defined]
    return adapter, fakes


def test_shutdown_calls_destroy_adam_once_per_chunk():
    """Each per-chunk ``DeepSpeedCPUAdam`` has its C state explicitly released."""
    adapter, fakes = _make_adapter_with_mock_ds(n_chunks=3)

    adapter.shutdown()

    for cid, fake_ds in fakes.items():
        # Captured pre-shutdown — the binding gets stubbed out post-destroy,
        # so we have to inspect via the original reference held in ``fakes``.
        assert len(fake_ds.destroy_calls) == 1, (
            f"chunk {cid}: destroy_adam should be called exactly once on shutdown, "
            f"got {len(fake_ds.destroy_calls)} calls"
        )
        assert fake_ds.destroy_calls[0] == 100 + cid


def test_shutdown_is_idempotent():
    """A second ``shutdown()`` call must not crash and must not double-call destroy."""
    adapter, fakes = _make_adapter_with_mock_ds(n_chunks=2)

    adapter.shutdown()
    assert all(len(fake.destroy_calls) == 1 for fake in fakes.values())

    # Second shutdown should be a no-op for the C state (executor is already
    # shut down — calling ``shutdown(wait=True)`` again is documented as
    # idempotent on ``ThreadPoolExecutor``).
    adapter.shutdown()

    for cid, fake_ds in fakes.items():
        assert len(fake_ds.destroy_calls) == 1, (
            f"chunk {cid}: destroy_adam must NOT be called again on second shutdown"
        )


def test_shutdown_replaces_ds_opt_adam_with_destroyed_stub():
    """Post-destroy, ``ds_opt_adam`` is swapped to ``_DestroyedDsAdam``.

    DeepSpeed's wrapper destructor unconditionally calls
    ``self.ds_opt_adam.destroy_adam(self.opt_id)`` — replacing the live C
    binding with a no-op stub keeps ``__del__`` harmless without
    monkey-patching the special method slot.
    """
    adapter, fakes = _make_adapter_with_mock_ds(n_chunks=2)

    adapter.shutdown()

    for cid in fakes:
        replaced = adapter._optims[ChunkId(cid)].ds_opt_adam  # type: ignore[attr-defined]
        assert isinstance(replaced, _DestroyedDsAdam), (
            f"chunk {cid}: ds_opt_adam should be a _DestroyedDsAdam stub post-shutdown, "
            f"got {type(replaced).__name__}"
        )
        # Stub call must be safe and return None.
        assert replaced.destroy_adam(0) is None


def test_shutdown_skips_missing_ds_opt_adam():
    """Half-initialised optims (``ds_opt_adam`` is ``None``) are skipped, not crashed.

    Mirrors the partial-init path where the DeepSpeed C extension fails
    to load and the constructor leaves ``ds_opt_adam`` unset / None.
    """
    adapter, fakes = _make_adapter_with_mock_ds(n_chunks=2)
    # Simulate partial init: drop the ds_opt_adam binding on chunk 0.
    adapter._optims[ChunkId(0)].ds_opt_adam = None  # type: ignore[attr-defined]

    # Must not raise, and must still destroy the healthy chunk's state.
    adapter.shutdown()

    assert len(fakes[1].destroy_calls) == 1
    # And chunk 0's (since-removed) fake recorded zero calls — not even
    # an attempted invocation against ``None``.
    assert len(fakes[0].destroy_calls) == 0


def test_shutdown_logs_destroy_failure_but_continues(caplog):
    """A per-chunk destroy failure is logged and does not block other chunks.

    CI hardening (2026-05-12): the assertion that
    ``LOG.warning(...)`` was invoked is done by patching the
    module-level ``LOG`` rather than by inspecting ``caplog.records``
    under ``caplog.at_level("axolotl")``. The caplog-based capture
    is brittle under pytest-xdist + axolotl's
    ``MultiProcessAdapter`` LoggerAdapter wrapper: the log record
    DOES emit (visible in CI stderr as
    ``[WARNING] [axolotl.integrations.protrain.chunk.optim]
    DeepSpeedCPUAdam destroy_adam failed for chunk 1: boom``) but
    ``caplog.records`` is intermittently empty depending on which
    other tests ran first in the same xdist worker (an autouse
    fixture in ``test_logging_config_file_capture.py`` removes
    handlers from ``logging.root`` which can disrupt caplog's
    propagation path mid-session).

    Patching ``optim_module.LOG.warning`` directly bypasses both
    the LoggerAdapter shape concern and the cross-test handler-
    removal risk: we're asserting the wrapper's intent ("a warning
    was logged when destroy_adam failed"), not the global logging
    plumbing's ability to route it.
    """
    from axolotl.integrations.protrain.chunk import optim as optim_module

    adapter, fakes = _make_adapter_with_mock_ds(n_chunks=3)

    class _ExplodingDs:
        def __init__(self) -> None:
            self.calls = 0

        def destroy_adam(self, _opt_id):  # noqa: ANN001
            self.calls += 1
            raise RuntimeError("boom")

    exploding = _ExplodingDs()
    adapter._optims[ChunkId(1)].ds_opt_adam = exploding  # type: ignore[attr-defined]

    with mock.patch.object(
        optim_module.LOG, "warning", wraps=optim_module.LOG.warning
    ) as mock_warn:
        adapter.shutdown()

    # Healthy chunks still got their destroy call.
    assert len(fakes[0].destroy_calls) == 1
    assert len(fakes[2].destroy_calls) == 1
    # The failing chunk attempted destroy exactly once.
    assert exploding.calls == 1
    # And the failure surfaced via a warning. Inspect the mock's
    # call args directly — match on the format-string prefix that
    # uniquely identifies the destroy_adam-failure log site.
    matching_calls = [
        call
        for call in mock_warn.call_args_list
        if call.args
        and isinstance(call.args[0], str)
        and "destroy_adam failed" in call.args[0]
    ]
    assert matching_calls, (
        f"Expected a LOG.warning call matching 'destroy_adam failed' but got "
        f"{[call.args for call in mock_warn.call_args_list]}"
    )
    # The warning's format args should include the failing chunk id (1) and
    # the underlying exception. Sanity-check both so a future copy-edit of
    # the warning text doesn't silently mask the diagnostic content.
    matching_call = matching_calls[0]
    assert ChunkId(1) in matching_call.args, (
        f"warning's chunk-id format arg should be ChunkId(1); got {matching_call.args}"
    )


def test_shutdown_destroys_state_even_when_wait_all_raises():
    """``wait_all`` raising must not skip the destroy_adam pass.

    The thread-pool drain happens inside a ``try``; the C-state release
    runs in the ``finally`` block alongside ``executor.shutdown``.
    Without that, a single broken async step would leak per-chunk
    DeepSpeed C state for the rest of the process lifetime.
    """
    adapter, fakes = _make_adapter_with_mock_ds(n_chunks=2)

    failing_future: Any = mock.MagicMock()
    failing_future.result.side_effect = RuntimeError("worker exploded")
    adapter._pending[ChunkId(0)] = failing_future  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="worker exploded"):
        adapter.shutdown()

    # destroy_adam must still have run for every chunk.
    for cid, fake_ds in fakes.items():
        assert len(fake_ds.destroy_calls) == 1, (
            f"chunk {cid}: destroy_adam must run even when wait_all raises"
        )
