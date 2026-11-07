"""Forkserver preload and the packing pool's start method."""

import importlib
import multiprocessing

import pytest

from axolotl.utils import forkserver

needs_forkserver = pytest.mark.skipif(
    "forkserver" not in multiprocessing.get_all_start_methods(),
    reason="no forkserver start method on this platform",
)


@pytest.fixture(name="preload_calls")
def fixture_preload_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(forkserver, "_configured", False)
    monkeypatch.setattr(
        forkserver.multiprocessing, "set_forkserver_preload", calls.append
    )
    return calls


@needs_forkserver
def test_preload_is_set_once(preload_calls):
    forkserver.preload_forkserver_modules()
    forkserver.preload_forkserver_modules()

    assert preload_calls == [list(forkserver.FORKSERVER_PRELOAD)]
    assert preload_calls[0][0] == "__main__"


def test_preload_skipped_without_forkserver(preload_calls, monkeypatch):
    monkeypatch.setattr(
        forkserver.multiprocessing, "get_all_start_methods", lambda: ["spawn"]
    )

    forkserver.preload_forkserver_modules()

    assert preload_calls == []


@pytest.mark.parametrize("name", forkserver.FORKSERVER_PRELOAD[1:])
def test_preload_modules_import(name):
    # the forkserver silently skips a module that fails to import
    importlib.import_module(name)
