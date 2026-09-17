"""Tests for fused RMSNorm+RoPE autotune telemetry.

Mocked end-to-end, so no Triton or CUDA is required (mirrors
``tests/integrations/test_scattermoe_autotune_telemetry.py``).
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_MODPATH = "axolotl.kernels.gemma4_fused_rope"


def _make_mock_config(kwargs, num_warps=2, num_stages=1):
    return SimpleNamespace(
        kwargs=kwargs, num_warps=num_warps, num_stages=num_stages, num_ctas=None
    )


def _make_fake_module(cache=None):
    kernel = SimpleNamespace(cache=cache if cache is not None else {})
    return SimpleNamespace(_rms_norm_rope_backward_kernel=kernel)


class TestCollector:
    def test_no_module_returns_empty(self):
        from axolotl.kernels.autotune_telemetry import (
            collect_fused_rope_autotune_configs,
        )

        with patch.dict(sys.modules, {_MODPATH: None}):
            assert collect_fused_rope_autotune_configs() == []

    def test_empty_cache_returns_empty(self):
        from axolotl.kernels.autotune_telemetry import (
            collect_fused_rope_autotune_configs,
        )

        with patch.dict(sys.modules, {_MODPATH: _make_fake_module()}):
            assert collect_fused_rope_autotune_configs() == []

    def test_populated_cache_returns_configs(self):
        from axolotl.kernels.autotune_telemetry import (
            collect_fused_rope_autotune_configs,
        )

        cfg = _make_mock_config({}, num_warps=2, num_stages=1)
        fake = _make_fake_module(cache={(128, "torch.bfloat16"): cfg})
        with patch.dict(sys.modules, {_MODPATH: fake}):
            result = collect_fused_rope_autotune_configs()

        assert len(result) == 1
        entry = result[0]
        assert entry["kernel"] == "fused_rms_norm_rope_bwd"
        assert entry["key"]["n_cols"] == 128
        assert entry["key"]["_extra"] == ["torch.bfloat16"]
        assert entry["config"]["num_warps"] == 2
        assert entry["config"]["num_stages"] == 1

    def test_multiple_head_dims(self):
        """head_dim 128 (Qwen3) and 256 (Qwen3.5) get separate cache entries."""
        from axolotl.kernels.autotune_telemetry import (
            collect_fused_rope_autotune_configs,
        )

        fake = _make_fake_module(
            cache={
                (128,): _make_mock_config({}, num_warps=2),
                (256,): _make_mock_config({}, num_warps=4),
            }
        )
        with patch.dict(sys.modules, {_MODPATH: fake}):
            result = collect_fused_rope_autotune_configs()

        assert {e["key"]["n_cols"] for e in result} == {128, 256}


class TestCallback:
    def _patch_collect(self, return_value=None, side_effect=None):
        return patch(
            "axolotl.kernels.autotune_telemetry.collect_fused_rope_autotune_configs",
            return_value=return_value,
            side_effect=side_effect,
        )

    def test_reports_once_on_first_step(self):
        from axolotl.kernels.autotune_telemetry import FusedRopeAutotuneReportCallback

        cb = FusedRopeAutotuneReportCallback()
        state = MagicMock()
        state.global_step = 1
        configs = [{"kernel": "fused_rms_norm_rope_bwd", "key": {}, "config": {}}]

        with (
            self._patch_collect(return_value=configs),
            patch("axolotl.telemetry.manager.TelemetryManager") as tm_cls,
        ):
            tm = MagicMock()
            tm.enabled = True
            tm_cls.get_instance.return_value = tm

            cb.on_step_end(args=MagicMock(), state=state, control=MagicMock())
            assert tm.send_event.call_count == 1
            kw = tm.send_event.call_args[1]
            assert kw["event_type"] == "fused-rope-autotune"
            assert kw["properties"]["kernel_count"] == 1

            cb.on_step_end(args=MagicMock(), state=state, control=MagicMock())
            assert tm.send_event.call_count == 1

    def test_retries_until_step_5_then_gives_up(self):
        from axolotl.kernels.autotune_telemetry import FusedRopeAutotuneReportCallback

        cb = FusedRopeAutotuneReportCallback()
        with self._patch_collect(return_value=[]):
            for step in range(1, 7):
                state = MagicMock()
                state.global_step = step
                cb.on_step_end(args=MagicMock(), state=state, control=MagicMock())
        assert cb._reported is True

    def test_includes_gpu_info(self):
        from axolotl.kernels.autotune_telemetry import FusedRopeAutotuneReportCallback

        cb = FusedRopeAutotuneReportCallback()
        state = MagicMock()
        state.global_step = 1
        configs = [{"kernel": "fused_rms_norm_rope_bwd", "key": {}, "config": {}}]
        gpu = {
            "gpu_name": "NVIDIA H100",
            "gpu_compute_capability": "9.0",
            "gpu_memory_bytes": 85899345920,
        }

        with (
            self._patch_collect(return_value=configs),
            patch("axolotl.kernels.autotune_telemetry._get_gpu_info", return_value=gpu),
            patch("axolotl.telemetry.manager.TelemetryManager") as tm_cls,
        ):
            tm = MagicMock()
            tm.enabled = True
            tm_cls.get_instance.return_value = tm

            cb.on_step_end(args=MagicMock(), state=state, control=MagicMock())
            props = tm.send_event.call_args[1]["properties"]
            assert props["gpu_name"] == "NVIDIA H100"
            assert props["gpu_compute_capability"] == "9.0"

    def test_skips_send_when_telemetry_disabled(self):
        from axolotl.kernels.autotune_telemetry import FusedRopeAutotuneReportCallback

        cb = FusedRopeAutotuneReportCallback()
        state = MagicMock()
        state.global_step = 1

        with (
            self._patch_collect(
                return_value=[{"kernel": "x", "key": {}, "config": {}}]
            ),
            patch("axolotl.telemetry.manager.TelemetryManager") as tm_cls,
        ):
            tm = MagicMock()
            tm.enabled = False
            tm_cls.get_instance.return_value = tm

            cb.on_step_end(args=MagicMock(), state=state, control=MagicMock())
            assert tm.send_event.call_count == 0
            assert cb._reported is True
