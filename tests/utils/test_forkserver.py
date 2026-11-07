"""Forkserver preload and the packing pool's start method."""

import importlib
import multiprocessing
import os
import subprocess
import sys

import numpy as np
import pytest

from axolotl.utils import forkserver
from axolotl.utils.samplers import bin_packing

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

    assert preload_calls == [forkserver.forkserver_preload_modules()]


def test_preload_skipped_without_forkserver(preload_calls, monkeypatch):
    monkeypatch.setattr(
        forkserver.multiprocessing, "get_all_start_methods", lambda: ["spawn"]
    )

    forkserver.preload_forkserver_modules()

    assert preload_calls == []


@pytest.mark.parametrize(
    "explicit, available, heavy",
    [
        (None, ["forkserver", "spawn", "fork"], True),
        (None, ["fork", "spawn", "forkserver"], False),
        (None, ["spawn", "fork", "forkserver"], False),
        ("spawn", ["forkserver", "spawn", "fork"], False),
    ],
)
def test_dataloader_modules_only_when_workers_use_the_forkserver(
    monkeypatch, explicit, available, heavy
):
    monkeypatch.setattr(
        forkserver.multiprocessing, "get_start_method", lambda allow_none: explicit
    )
    monkeypatch.setattr(
        forkserver.multiprocessing, "get_all_start_methods", lambda: available
    )

    modules = forkserver.forkserver_preload_modules()

    assert modules[:2] == ["__main__", "axolotl.utils.samplers.bin_packing"]
    assert ("axolotl.utils.collators" in modules) is heavy


@pytest.mark.parametrize(
    "name", forkserver.PACKING_PRELOAD + forkserver.DATALOADER_PRELOAD
)
def test_preload_modules_import(name):
    # the forkserver silently skips a module that fails to import
    importlib.import_module(name)


def test_packing_children_skip_the_torch_import_path():
    code = (
        "import sys, axolotl.utils.samplers.bin_packing; "
        "print(sorted(m for m in ('torch', 'accelerate', 'transformers') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ,
    )

    assert result.stdout.strip() == "[]"


class _RecordingExecutor:
    """Runs the pool's tasks in-process and records its start method."""

    seen: dict = {}

    def __init__(self, max_workers, mp_context):
        type(self).seen["method"] = mp_context.get_start_method()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, tasks):
        return map(fn, tasks)


@pytest.fixture(name="pool")
def fixture_pool(monkeypatch):
    _RecordingExecutor.seen = {}
    monkeypatch.setattr(bin_packing, "ProcessPoolExecutor", _RecordingExecutor)
    monkeypatch.setattr(
        bin_packing,
        "preload_forkserver_modules",
        lambda: _RecordingExecutor.seen.setdefault("preloaded", True),
    )
    return _RecordingExecutor.seen


def _pack(**kwargs):
    lengths = np.array([10, 20, 30, 40], dtype=np.int32)
    bins = bin_packing.pack_parallel(
        lengths, bin_capacity=50, group_size=2, bin_size=4, num_processes=2, **kwargs
    )
    assert sorted(idx for bin_ in bins for idx in bin_) == [0, 1, 2, 3]


@needs_forkserver
def test_pack_parallel_defaults_to_preloaded_forkserver(pool):
    _pack()

    assert pool == {"method": "forkserver", "preloaded": True}


def test_pack_parallel_keeps_an_explicit_start_method(pool):
    _pack(mp_start_method="spawn")

    assert pool["method"] == "spawn"


def test_pack_parallel_packs_a_single_group_in_process(pool):
    lengths = np.array([10, 20, 30, 40], dtype=np.int32)

    bins = bin_packing.pack_parallel(
        lengths, bin_capacity=50, group_size=8, bin_size=4, num_processes=4
    )

    assert pool == {}
    assert sorted(idx for bin_ in bins for idx in bin_) == [0, 1, 2, 3]
