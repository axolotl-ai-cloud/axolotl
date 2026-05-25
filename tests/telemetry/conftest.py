"""Shared pytest fixtures for telemetry tests."""

import pytest


@pytest.fixture(autouse=True)
def del_track_env(monkeypatch):
    monkeypatch.delenv("AXOLOTL_DO_NOT_TRACK", raising=False)
    yield


@pytest.fixture(autouse=True)
def isolate_rank_env(monkeypatch):
    """Scrub distributed-rank env vars before each telemetry test.

    test_telemetry_disabled_for_non_main_process sets RANK=1 via
    patch.dict(os.environ, ...) which restores on test exit BUT only if
    the test reaches the patch teardown cleanly. Under pytest-xdist a
    sibling test on the same worker may observe a non-zero RANK from an
    earlier test's leaked state. This autouse fixture deletes RANK /
    LOCAL_RANK / WORLD_SIZE before every telemetry test so the worker
    starts from a clean distributed-rank state regardless of what ran
    before it.
    """
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(var, raising=False)
    yield
