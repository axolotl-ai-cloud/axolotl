"""Tests for the SLOW_OFFLOAD_REGATHER per-chunk watchdog.

The watchdog tracks per-chunk wall time across OFFLOAD re-gather call paths
(forward_regather / backward_regather / resume_restore). It is the precursor
diagnostic for the bs=2 + n_offload>0 hang investigation — v71/v72-redux
verified the hang but neither SLOW_GATHER nor SLOW_ADAM fired, so the
remaining suspect is per-chunk re-gather H2D + NCCL all_gather during
backward. These tests confirm the watchdog fires on slow re-gathers and
stays silent on fast paths.
"""

from __future__ import annotations

import logging
import time
from typing import cast
from unittest.mock import patch

import pytest

from axolotl.integrations.protrain.chunk import manager as manager_mod
from axolotl.integrations.protrain.types import ChunkId


def test_slow_backward_regather_fires_warning(caplog):
    """A slow backward re-gather must WARN with chunk_id + phase."""
    cid = cast(ChunkId, 27)

    # Threshold tight enough that a 50ms sleep trips it but a 0ms gather skates by.
    threshold_s = 0.02

    def _slow_gather_impl(self, chunk_id, stream=None, phase="forward_regather"):
        time.sleep(0.05)

    with patch.object(manager_mod, "_SLOW_OFFLOAD_REGATHER_S", threshold_s):
        with patch.object(manager_mod, "_SLOW_GATHER_THRESHOLD_S", 0.0):
            with patch.object(
                manager_mod.ChunkManager, "_gather_impl", _slow_gather_impl
            ):
                # Use a bare instance — gather() only touches _gather_impl + a few
                # read-only attrs the warning message inspects.
                mgr = manager_mod.ChunkManager.__new__(manager_mod.ChunkManager)
                mgr._chunk_shards = {}
                mgr._active_chunks = set()
                mgr.buffer_pool = None

                with caplog.at_level(logging.WARNING):
                    mgr.gather(cid, phase="backward_regather")

    matching = [
        rec for rec in caplog.records if "SLOW_OFFLOAD_REGATHER" in rec.getMessage()
    ]
    assert matching, (
        "expected a SLOW_OFFLOAD_REGATHER warning when gather exceeds the threshold"
    )
    msg = matching[0].getMessage()
    assert "chunk_id=27" in msg
    assert "phase=backward_regather" in msg


def test_fast_regather_does_not_warn(caplog):
    """A fast re-gather (below threshold) must produce no SLOW_OFFLOAD_REGATHER warning."""
    cid = cast(ChunkId, 5)
    threshold_s = 5.0  # well above any reasonable fast path

    def _fast_gather_impl(self, chunk_id, stream=None, phase="forward_regather"):
        # No-op; instantaneous.
        return None

    with patch.object(manager_mod, "_SLOW_OFFLOAD_REGATHER_S", threshold_s):
        with patch.object(manager_mod, "_SLOW_GATHER_THRESHOLD_S", 0.0):
            with patch.object(
                manager_mod.ChunkManager, "_gather_impl", _fast_gather_impl
            ):
                mgr = manager_mod.ChunkManager.__new__(manager_mod.ChunkManager)
                mgr._chunk_shards = {}
                mgr._active_chunks = set()
                mgr.buffer_pool = None

                with caplog.at_level(logging.WARNING):
                    mgr.gather(cid, phase="backward_regather")
                    mgr.gather(cid, phase="forward_regather")

    matching = [
        rec for rec in caplog.records if "SLOW_OFFLOAD_REGATHER" in rec.getMessage()
    ]
    assert not matching, (
        f"expected no SLOW_OFFLOAD_REGATHER warning for fast re-gather, got: "
        f"{[rec.getMessage() for rec in matching]}"
    )


def test_forward_regather_phase_tag(caplog):
    """Default phase tag is forward_regather; backward callers override explicitly."""
    cid = cast(ChunkId, 9)
    threshold_s = 0.02

    def _slow_gather_impl(self, chunk_id, stream=None, phase="forward_regather"):
        time.sleep(0.05)

    with patch.object(manager_mod, "_SLOW_OFFLOAD_REGATHER_S", threshold_s):
        with patch.object(manager_mod, "_SLOW_GATHER_THRESHOLD_S", 0.0):
            with patch.object(
                manager_mod.ChunkManager, "_gather_impl", _slow_gather_impl
            ):
                mgr = manager_mod.ChunkManager.__new__(manager_mod.ChunkManager)
                mgr._chunk_shards = {}
                mgr._active_chunks = set()
                mgr.buffer_pool = None

                with caplog.at_level(logging.WARNING):
                    # No explicit phase => forward_regather.
                    mgr.gather(cid)

    matching = [
        rec for rec in caplog.records if "SLOW_OFFLOAD_REGATHER" in rec.getMessage()
    ]
    assert matching
    assert "phase=forward_regather" in matching[0].getMessage()


def test_watchdog_disabled_when_threshold_zero(caplog):
    """Setting threshold to 0 disables the watchdog entirely."""
    cid = cast(ChunkId, 13)

    def _slow_gather_impl(self, chunk_id, stream=None, phase="forward_regather"):
        time.sleep(0.05)

    with patch.object(manager_mod, "_SLOW_OFFLOAD_REGATHER_S", 0.0):
        with patch.object(manager_mod, "_SLOW_GATHER_THRESHOLD_S", 0.0):
            with patch.object(
                manager_mod.ChunkManager, "_gather_impl", _slow_gather_impl
            ):
                mgr = manager_mod.ChunkManager.__new__(manager_mod.ChunkManager)
                mgr._chunk_shards = {}
                mgr._active_chunks = set()
                mgr.buffer_pool = None

                with caplog.at_level(logging.WARNING):
                    mgr.gather(cid, phase="backward_regather")

    matching = [
        rec for rec in caplog.records if "SLOW_OFFLOAD_REGATHER" in rec.getMessage()
    ]
    assert not matching


def test_threshold_env_default_is_5s():
    """PROTRAIN_DEBUG_SLOW_OFFLOAD_REGATHER_S defaults to 5.0 when unset/invalid."""
    import os

    saved = os.environ.pop("PROTRAIN_DEBUG_SLOW_OFFLOAD_REGATHER_S", None)
    try:
        assert manager_mod._slow_offload_regather_threshold_s() == pytest.approx(5.0)
        os.environ["PROTRAIN_DEBUG_SLOW_OFFLOAD_REGATHER_S"] = "not-a-float"
        assert manager_mod._slow_offload_regather_threshold_s() == pytest.approx(5.0)
        os.environ["PROTRAIN_DEBUG_SLOW_OFFLOAD_REGATHER_S"] = "1.5"
        assert manager_mod._slow_offload_regather_threshold_s() == pytest.approx(1.5)
        # Negative values clamp to 0 (disabled).
        os.environ["PROTRAIN_DEBUG_SLOW_OFFLOAD_REGATHER_S"] = "-3.0"
        assert manager_mod._slow_offload_regather_threshold_s() == 0.0
    finally:
        if saved is None:
            os.environ.pop("PROTRAIN_DEBUG_SLOW_OFFLOAD_REGATHER_S", None)
        else:
            os.environ["PROTRAIN_DEBUG_SLOW_OFFLOAD_REGATHER_S"] = saved


def test_gather_for_backward_passes_backward_regather_phase(caplog):
    """ChunkManager.gather_for_backward must tag its gather() call with phase=backward_regather."""
    cid = cast(ChunkId, 42)
    threshold_s = 0.02
    captured_phases: list[str] = []

    real_gather = manager_mod.ChunkManager.gather

    def _spy_gather(self, chunk_id, phase="forward_regather", stream=None):
        captured_phases.append(phase)
        return real_gather(self, chunk_id, phase=phase, stream=stream)

    def _slow_gather_impl(self, chunk_id, stream=None, phase="forward_regather"):
        time.sleep(0.05)

    with patch.object(manager_mod, "_SLOW_OFFLOAD_REGATHER_S", threshold_s):
        with patch.object(manager_mod, "_SLOW_GATHER_THRESHOLD_S", 0.0):
            with patch.object(manager_mod.ChunkManager, "_gather_impl", _slow_gather_impl):
                with patch.object(manager_mod.ChunkManager, "gather", _spy_gather):
                    mgr = manager_mod.ChunkManager.__new__(manager_mod.ChunkManager)
                    mgr._chunk_shards = {}
                    mgr._active_chunks = set()
                    mgr.buffer_pool = None
                    mgr._backward_refcount = {}

                    with caplog.at_level(logging.WARNING):
                        mgr.gather_for_backward(cid)

    assert captured_phases == ["backward_regather"], (
        f"gather_for_backward must invoke gather() with phase=backward_regather, "
        f"got {captured_phases}"
    )
    matching = [
        rec for rec in caplog.records if "SLOW_OFFLOAD_REGATHER" in rec.getMessage()
    ]
    assert matching
    assert "phase=backward_regather" in matching[0].getMessage()
