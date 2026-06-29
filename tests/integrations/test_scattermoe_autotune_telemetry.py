"""Tests for scattermoe autotune telemetry integration.

These tests use mocking to verify the collection and reporting logic
without requiring Triton or CUDA.
"""

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Simulate the hash-suffixed module name that LocalLayerRepository creates.
_FAKE_MODULE_NAME = "scattermoe_lora_abc123.kernels.lora_ops"


def _make_mock_config(kwargs, num_warps=4, num_stages=3):
    """Create a mock triton.Config-like object."""
    return SimpleNamespace(kwargs=kwargs, num_warps=num_warps, num_stages=num_stages)


def _make_mock_kernel(cache=None, name="kernel", keys=("M_BUCKET", "N", "K")):
    """Mock a Triton ``Autotuner``: ``.cache`` dict + ``.base_fn`` (carrying ``__name__``,
    which the collector uses for the kernel name) + declared autotune ``.keys``."""
    return SimpleNamespace(
        cache=cache if cache is not None else {},
        base_fn=SimpleNamespace(__name__=name),
        keys=list(keys),
    )


def _make_mock_lora_ops(
    fwd_cache=None, dx_cache=None, bwd_cache=None, fused_cache=None
):
    """Build a mock ``lora_ops`` module with the four kernel attributes. The collector names a
    kernel by its wrapped fn's ``__name__``, so set those to the expected kernel names."""
    return SimpleNamespace(
        _scatter2scatter_lora=_make_mock_kernel(
            fwd_cache, name="scatter2scatter_lora_fwd"
        ),
        _scatter2scatter_lora_dX=_make_mock_kernel(
            dx_cache, name="scatter2scatter_lora_dX"
        ),
        _group_bwd_lora=_make_mock_kernel(bwd_cache, name="group_bwd_lora"),
        _group_bwd_lora_fused=_make_mock_kernel(
            fused_cache, name="group_bwd_lora_fused"
        ),
    )


@contextmanager
def _inject_lora_ops(mock_module, name="scattermoe_lora_abc123.kernels.lora_ops"):
    """Register ``mock_module`` under a ``lora_ops``-hint name in ``sys.modules`` and hide EVERY
    real kernel module the collector scans, so the sys.modules-scanning collector sees only the
    mock. Hiding just real ``lora_ops`` modules isn't enough: the collector also scans the
    dsv4/glm_dsa/scattermoe kernel modules (``_KERNEL_MODULE_HINTS``), and when an earlier test in
    the same process has autotuned one of those, its real cache otherwise leaks into the result."""
    hide = {n: None for n in _real_scanned_kernel_module_names() if n != name}
    with patch.dict(sys.modules, {name: mock_module, **hide}):
        yield


def _real_scanned_kernel_module_names():
    """``sys.modules`` keys the collector would scan (any name matching its kernel-module hints).

    Other tests in the same process may have imported/autotuned the *real* kernels (lora_ops,
    dsv4, glm_dsa, ...). Hide every such entry so the discovery tests see only the injected mock.
    """
    from axolotl.integrations.kernels.autotune_collector import _KERNEL_MODULE_HINTS

    return [
        name
        for name, mod in list(sys.modules.items())
        if mod is not None and any(h in name for h in _KERNEL_MODULE_HINTS)
    ]


# =========================================================================
# TestAutotuneCollector
# =========================================================================


class TestAutotuneCollector:
    """Test ``collect_autotune_configs`` with mocked kernel objects.

    The collector scans ``sys.modules`` for kernel-bearing modules, so each test injects a mock
    ``lora_ops`` module via ``_inject_lora_ops`` (which also hides any real ``lora_ops`` modules
    other tests in the same pytest-xdist worker may have loaded).
    """

    def test_empty_cache_returns_empty_list(self):
        """When no kernel has been autotuned yet, return ``[]``."""
        mock_lora_ops = _make_mock_lora_ops()

        with _inject_lora_ops(mock_lora_ops):
            from axolotl.integrations.kernels.autotune_collector import (
                collect_autotune_configs,
            )

            result = collect_autotune_configs()
            assert result == []

    def test_populated_cache_returns_configs(self):
        """When a cache entry exists, it appears in the output."""
        cfg = _make_mock_config(
            {"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4
        )
        mock_lora_ops = _make_mock_lora_ops(fwd_cache={(2048, 4096, 1024): cfg})

        with _inject_lora_ops(mock_lora_ops):
            from axolotl.integrations.kernels.autotune_collector import (
                collect_autotune_configs,
            )

            result = collect_autotune_configs()

        assert len(result) == 1
        entry = result[0]
        assert entry["kernel"] == "scatter2scatter_lora_fwd"
        assert entry["key"] == {"M_BUCKET": 2048, "N": 4096, "K": 1024}
        assert entry["config"]["BLOCK_N"] == 128
        assert entry["config"]["BLOCK_K"] == 64
        assert entry["config"]["num_warps"] == 8
        assert entry["config"]["num_stages"] == 4

    def test_multiple_kernels_and_keys(self):
        """Multiple cache entries across kernels are all returned."""
        cfg_fwd = _make_mock_config({"BLOCK_N": 128, "BLOCK_K": 32})
        cfg_dx = _make_mock_config({"BLOCK_K": 64, "BLOCK_N": 128}, num_warps=8)

        mock_lora_ops = _make_mock_lora_ops(
            fwd_cache={(16, 256, 128): cfg_fwd},
            dx_cache={(16, 256, 128): cfg_dx},
        )

        with _inject_lora_ops(mock_lora_ops):
            from axolotl.integrations.kernels.autotune_collector import (
                collect_autotune_configs,
            )

            result = collect_autotune_configs()

        assert len(result) == 2
        names = {r["kernel"] for r in result}
        assert "scatter2scatter_lora_fwd" in names
        assert "scatter2scatter_lora_dX" in names

    def test_extra_key_elements_stored(self):
        """Dtype or other extra elements in the cache key are captured."""
        cfg = _make_mock_config({"BLOCK_N": 64, "BLOCK_K": 32})
        cache_key = (512, 1024, 256, "float16", "float16")

        mock_lora_ops = _make_mock_lora_ops(fwd_cache={cache_key: cfg})

        with _inject_lora_ops(mock_lora_ops):
            from axolotl.integrations.kernels.autotune_collector import (
                collect_autotune_configs,
            )

            result = collect_autotune_configs()

        assert len(result) == 1
        key = result[0]["key"]
        assert key["M_BUCKET"] == 512
        assert key["N"] == 1024
        assert key["K"] == 256
        assert key["_extra"] == ["float16", "float16"]

    def test_no_module_in_sys_modules_returns_empty(self):
        """If no populated lora_ops module is loaded, return ``[]``."""
        from axolotl.integrations.kernels.autotune_collector import (
            collect_autotune_configs,
        )

        hide = {n: None for n in _real_scanned_kernel_module_names()}
        with patch.dict(sys.modules, hide):
            result = collect_autotune_configs()
        assert result == []

    def test_finds_module_under_hash_suffixed_name(self):
        """Collector finds lora_ops regardless of the hash suffix."""
        cfg = _make_mock_config({"BLOCK_N": 256, "BLOCK_K": 128})
        mock_lora_ops = _make_mock_lora_ops(fwd_cache={(8, 512, 64): cfg})

        # Use a different hash to prove it's not hardcoded.
        alt_name = "scattermoe_lora_deadbeef.kernels.lora_ops"

        # Temporarily hide any real lora_ops modules that other tests in
        # the same xdist worker may have loaded, so only our mock is found.
        real_names = _real_scanned_kernel_module_names()
        hide_patch = {name: None for name in real_names}

        with patch.dict(sys.modules, {alt_name: mock_lora_ops, **hide_patch}):
            from axolotl.integrations.kernels.autotune_collector import (
                collect_autotune_configs,
            )

            result = collect_autotune_configs()

        assert len(result) == 1
        assert result[0]["config"]["BLOCK_N"] == 256


# =========================================================================
# TestAutotuneReportCallback
# =========================================================================


class TestAutotuneReportCallback:
    """Test the callback fires once and sends the correct event."""

    def test_reports_once_on_first_step(self):
        """Callback should call ``send_event`` exactly once."""
        from axolotl.integrations.kernels.autotune_callback import (
            AutotuneReportCallback,
        )

        cb = AutotuneReportCallback()
        mock_state = MagicMock()
        mock_state.global_step = 1

        fake_configs = [{"kernel": "test_fwd", "key": {}, "config": {}}]

        with (
            patch(
                "axolotl.integrations.kernels.autotune_collector.collect_autotune_configs",
                return_value=fake_configs,
            ),
            patch("axolotl.telemetry.manager.TelemetryManager") as mock_tm_cls,
        ):
            mock_tm = MagicMock()
            mock_tm.enabled = True
            mock_tm_cls.get_instance.return_value = mock_tm

            cb.on_step_end(args=MagicMock(), state=mock_state, control=MagicMock())
            assert mock_tm.send_event.call_count == 1

            call_kwargs = mock_tm.send_event.call_args[1]
            assert call_kwargs["event_type"] == "triton-autotune"
            assert call_kwargs["properties"]["kernel_count"] == 1

            # Second call should NOT send again.
            cb.on_step_end(args=MagicMock(), state=mock_state, control=MagicMock())
            assert mock_tm.send_event.call_count == 1

    def test_retries_until_step_5_then_gives_up(self):
        """If no configs found by step 5, stop retrying."""
        from axolotl.integrations.kernels.autotune_callback import (
            AutotuneReportCallback,
        )

        cb = AutotuneReportCallback()

        with patch(
            "axolotl.integrations.kernels.autotune_collector.collect_autotune_configs",
            return_value=[],
        ):
            for step in range(1, 7):
                mock_state = MagicMock()
                mock_state.global_step = step
                cb.on_step_end(args=MagicMock(), state=mock_state, control=MagicMock())

            assert cb._reported is True

    def test_reports_on_retry_when_data_arrives(self):
        """If step 1 has no data but step 2 does, report at step 2."""
        from axolotl.integrations.kernels.autotune_callback import (
            AutotuneReportCallback,
        )

        cb = AutotuneReportCallback()
        fake_configs = [{"kernel": "fwd", "key": {}, "config": {}}]

        call_count = 0

        def _collector():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []
            return fake_configs

        with (
            patch(
                "axolotl.integrations.kernels.autotune_collector.collect_autotune_configs",
                side_effect=_collector,
            ),
            patch("axolotl.telemetry.manager.TelemetryManager") as mock_tm_cls,
        ):
            mock_tm = MagicMock()
            mock_tm.enabled = True
            mock_tm_cls.get_instance.return_value = mock_tm

            # Step 1 — empty, no report
            s1 = MagicMock()
            s1.global_step = 1
            cb.on_step_end(args=MagicMock(), state=s1, control=MagicMock())
            assert mock_tm.send_event.call_count == 0

            # Step 2 — data arrives, report
            s2 = MagicMock()
            s2.global_step = 2
            cb.on_step_end(args=MagicMock(), state=s2, control=MagicMock())
            assert mock_tm.send_event.call_count == 1

    def test_includes_gpu_info(self):
        """Event properties should include GPU identification."""
        from axolotl.integrations.kernels.autotune_callback import (
            AutotuneReportCallback,
        )

        cb = AutotuneReportCallback()
        mock_state = MagicMock()
        mock_state.global_step = 1

        fake_configs = [{"kernel": "fwd", "key": {}, "config": {}}]
        fake_gpu = {
            "gpu_name": "NVIDIA H100",
            "gpu_compute_capability": "9.0",
            "gpu_memory_bytes": 85899345920,
        }

        fake_smem = {"smem_capacity_bytes": 233472}

        with (
            patch(
                "axolotl.integrations.kernels.autotune_collector.collect_autotune_configs",
                return_value=fake_configs,
            ),
            patch(
                "axolotl.integrations.kernels.autotune_callback._get_gpu_info",
                return_value=fake_gpu,
            ),
            patch(
                "axolotl.integrations.kernels.autotune_callback._get_smem_capacity",
                return_value=fake_smem,
            ),
            patch("axolotl.telemetry.manager.TelemetryManager") as mock_tm_cls,
        ):
            mock_tm = MagicMock()
            mock_tm.enabled = True
            mock_tm_cls.get_instance.return_value = mock_tm

            cb.on_step_end(args=MagicMock(), state=mock_state, control=MagicMock())
            props = mock_tm.send_event.call_args[1]["properties"]
            assert props["gpu_name"] == "NVIDIA H100"
            assert props["gpu_compute_capability"] == "9.0"
            assert props["gpu_memory_bytes"] == 85899345920
            assert props["smem_capacity_bytes"] == 233472

    def test_skips_send_when_telemetry_disabled(self):
        """If telemetry is disabled, no event is sent."""
        from axolotl.integrations.kernels.autotune_callback import (
            AutotuneReportCallback,
        )

        cb = AutotuneReportCallback()
        mock_state = MagicMock()
        mock_state.global_step = 1

        with (
            patch(
                "axolotl.integrations.kernels.autotune_collector.collect_autotune_configs",
                return_value=[{"kernel": "fwd", "key": {}, "config": {}}],
            ),
            patch("axolotl.telemetry.manager.TelemetryManager") as mock_tm_cls,
        ):
            mock_tm = MagicMock()
            mock_tm.enabled = False
            mock_tm_cls.get_instance.return_value = mock_tm

            cb.on_step_end(args=MagicMock(), state=mock_state, control=MagicMock())
            assert mock_tm.send_event.call_count == 0
            # Should still mark as reported so we don't retry.
            assert cb._reported is True


# =========================================================================
# TestKernelsPluginCallbackRegistration
# =========================================================================


class TestKernelsPluginCallbackRegistration:
    """Test that ``KernelsPlugin`` registers the callback correctly."""

    def test_scattermoe_registers_callback(self):
        """When ``use_scattermoe=True``, plugin returns the callback."""
        from axolotl.integrations.kernels.autotune_callback import (
            AutotuneReportCallback,
        )
        from axolotl.integrations.kernels.plugin import KernelsPlugin

        plugin = KernelsPlugin()
        cfg = MagicMock()
        cfg.use_scattermoe = True
        model = MagicMock()

        callbacks = plugin.add_callbacks_pre_trainer(cfg, model)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], AutotuneReportCallback)

    def test_no_scattermoe_no_callback(self):
        """When ``use_scattermoe=False``, plugin returns empty list."""
        from axolotl.integrations.kernels.plugin import KernelsPlugin

        plugin = KernelsPlugin()
        cfg = MagicMock()
        cfg.use_scattermoe = False
        model = MagicMock()

        callbacks = plugin.add_callbacks_pre_trainer(cfg, model)
        assert callbacks == []
