"""Shared pytest fixtures for telemetry tests."""

import pytest


@pytest.fixture(autouse=True)
def del_track_env(monkeypatch):
    monkeypatch.delenv("AXOLOTL_DO_NOT_TRACK", raising=False)
    yield
