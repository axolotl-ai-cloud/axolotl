"""Shared pytest fixtures for telemetry tests."""

import pytest


@pytest.fixture(autouse=True)
def disable_telemetry(monkeypatch):
    monkeypatch.delenv("AXOLOTL_DO_NOT_TRACK", raising=False)
    yield
