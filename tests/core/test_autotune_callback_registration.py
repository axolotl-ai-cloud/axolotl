# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Autotune reporting is independent of model and kernel plugin flags."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axolotl.integrations.kernels.autotune_callback import AutotuneReportCallback
from axolotl.integrations.kernels.sonicmoe_autotune import (
    SonicMoEAutotuneReportCallback,
)
from axolotl.kernels.autotune_telemetry import FusedRopeAutotuneReportCallback
from axolotl.utils.dict import DictDefault


@pytest.fixture
def base_builder_module():
    source = Path(__file__).resolve().parents[2] / "src/axolotl/core/builders/base.py"
    spec = importlib.util.spec_from_file_location("autotune_test_builder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model_type", ["deepseek_v4", "glm_dsa", "gemma4", "llama"])
@pytest.mark.parametrize("enabled", [True, False])
def test_callbacks_follow_telemetry_gate(base_builder_module, model_type, enabled):
    builder = SimpleNamespace()
    builder.cfg = DictDefault({"model_config_type": model_type})
    builder.model = SimpleNamespace()
    with (
        patch.object(base_builder_module, "PluginManager") as plugins,
        patch.object(base_builder_module, "TelemetryManager") as telemetry,
    ):
        plugins.get_instance.return_value.add_callbacks_pre_trainer.return_value = []
        telemetry.get_instance.return_value.enabled = enabled
        callbacks = base_builder_module.TrainerBuilderBase.get_callbacks(builder)
    for callback_type in (
        AutotuneReportCallback,
        FusedRopeAutotuneReportCallback,
        SonicMoEAutotuneReportCallback,
    ):
        assert sum(
            isinstance(callback, callback_type) for callback in callbacks
        ) == int(enabled)
