"""Tests for the MM dataloader_num_workers=0 warning."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from axolotl.core.builders import causal as _causal_builder
from axolotl.core.builders.causal import _warn_if_num_workers_zero_for_mm


def _make_cfg(processor_type=None, dataloader_num_workers=None, train_on_inputs=False):
    return SimpleNamespace(
        processor_type=processor_type,
        dataloader_num_workers=dataloader_num_workers,
        train_on_inputs=train_on_inputs,
    )


@pytest.fixture
def _reset_mm_warn_state():
    """Reset the module-level once-per-process guard set so each test is independent.

    Production keeps the warning one-shot per process so logs aren't spammed
    when ``build()`` builds collators for both train and eval.
    """
    _causal_builder._MM_NUM_WORKERS_WARNED.clear()
    yield
    _causal_builder._MM_NUM_WORKERS_WARNED.clear()


def _capture_warning(caplog, cfg) -> list[str]:
    log = logging.getLogger("axolotl.core.builders.causal")
    log.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="axolotl.core.builders.causal"):
            _warn_if_num_workers_zero_for_mm(cfg, log)
    finally:
        log.removeHandler(caplog.handler)
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_warn_num_workers_zero_for_mm_emits_warning(_reset_mm_warn_state, caplog):
    cfg = _make_cfg(processor_type="qwen2_vl", dataloader_num_workers=0)
    msgs = _capture_warning(caplog, cfg)
    assert any("dataloader_num_workers=0" in m for m in msgs)


def test_warn_num_workers_zero_for_mm_silent_when_workers_set(
    _reset_mm_warn_state, caplog
):
    cfg = _make_cfg(processor_type="qwen2_vl", dataloader_num_workers=2)
    msgs = _capture_warning(caplog, cfg)
    assert not any("dataloader_num_workers=0" in m for m in msgs)


def test_warn_num_workers_zero_for_mm_silent_when_no_processor(
    _reset_mm_warn_state, caplog
):
    cfg = _make_cfg(processor_type=None, dataloader_num_workers=0)
    msgs = _capture_warning(caplog, cfg)
    assert not any("dataloader_num_workers=0" in m for m in msgs)


def test_warn_num_workers_zero_for_mm_treats_none_as_zero(_reset_mm_warn_state, caplog):
    """``None`` (default) and ``0`` should both trigger the warning."""
    cfg = _make_cfg(processor_type="qwen2_vl", dataloader_num_workers=None)
    msgs = _capture_warning(caplog, cfg)
    assert any("dataloader_num_workers=0" in m for m in msgs)


def test_warn_num_workers_zero_for_mm_is_one_shot(_reset_mm_warn_state, caplog):
    """Second invocation in the same process should NOT re-emit the warning."""
    cfg = _make_cfg(processor_type="qwen2_vl", dataloader_num_workers=0)
    first_msgs = _capture_warning(caplog, cfg)
    assert any("dataloader_num_workers=0" in m for m in first_msgs)

    caplog.clear()
    second_msgs = _capture_warning(caplog, cfg)
    assert not any("dataloader_num_workers=0" in m for m in second_msgs)
