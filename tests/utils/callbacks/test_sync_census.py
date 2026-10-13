"""Tests for the sync census callback bucketing (no CUDA needed)."""

import json
import warnings
from types import SimpleNamespace

import pytest
import torch

from axolotl.utils.callbacks import sync_census
from axolotl.utils.callbacks.sync_census import SyncCensusCallback, _site_of


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_sync_debug_mode", lambda mode: None)


def _sync():
    warnings.warn("called a synchronizing CUDA operation", UserWarning, stacklevel=2)


def _state(step):
    return SimpleNamespace(global_step=step)


def _run(callback, tmp_path, syncs_per_microstep, ga=2, steps=2, start=1):
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    control = SimpleNamespace(should_log=False)
    for step in range(start, start + steps):
        callback.on_step_begin(args, _state(step), control)
        for sub in range(ga):
            for _ in range(syncs_per_microstep):
                _sync()
            if sub < ga - 1:
                callback.on_substep_end(args, _state(step), control)
        # HF Trainer increments global_step before on_step_end
        callback.on_step_end(args, _state(step + 1), control)
        _sync()  # logging-time sync
        callback.on_log(args, _state(step + 1), control)
    return args


def test_counts_syncs_per_microstep_and_logging_separately(fake_cuda, tmp_path):
    callback = SyncCensusCallback(start_step=1, num_steps=2)

    _run(callback, tmp_path, syncs_per_microstep=3)

    assert callback.result is not None
    assert callback.result["microsteps"] == 4
    assert callback.result["syncs_per_microstep"] == {"min": 3, "median": 3, "max": 3}
    assert callback.result["other_buckets"]["log"] == 1
    written = json.loads((tmp_path / "sync_census.json").read_text())
    assert written["microsteps"] == 4


def test_only_counts_sync_warnings(fake_cuda, tmp_path):
    callback = SyncCensusCallback(start_step=1, num_steps=1)
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    control = SimpleNamespace(should_log=False)
    callback.on_step_begin(args, _state(1), control)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        warnings.warn("something unrelated", UserWarning, stacklevel=2)
    _sync()
    callback.on_step_end(args, _state(2), control)

    assert callback.result["syncs_per_microstep"]["max"] == 1


def test_restores_warning_hooks(fake_cuda, tmp_path):
    before = warnings.showwarning
    callback = SyncCensusCallback(start_step=1, num_steps=1)

    _run(callback, tmp_path, syncs_per_microstep=1, steps=1)

    assert warnings.showwarning is before
    assert not callback._active


def test_inactive_before_start_step(fake_cuda, tmp_path):
    callback = SyncCensusCallback(start_step=5, num_steps=1)
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    control = SimpleNamespace(should_log=False)

    callback.on_step_begin(args, _state(0), control)
    _sync()
    callback.on_step_end(args, _state(1), control)

    assert callback.result is None
    assert not callback._active


def test_skips_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    callback = SyncCensusCallback(start_step=1, num_steps=1)
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    control = SimpleNamespace(should_log=False)

    callback.on_step_begin(args, _state(1), control)

    assert not callback._active


def test_site_prefers_library_frame_over_torch():
    stack = [
        SimpleNamespace(
            filename="/x/site-packages/transformers/trainer.py", lineno=10, name="train"
        ),
        SimpleNamespace(
            filename="/x/site-packages/torch/_tensor.py", lineno=5, name="__int__"
        ),
    ]
    assert _site_of(stack) == "transformers/trainer.py:10 train"


def test_site_falls_back_to_user_code():
    stack = [SimpleNamespace(filename="/home/u/script.py", lineno=3, name="main")]
    assert _site_of(stack) == "script.py:3 main"
    assert sync_census._SYNC_MARKER in "called a synchronizing CUDA operation"
