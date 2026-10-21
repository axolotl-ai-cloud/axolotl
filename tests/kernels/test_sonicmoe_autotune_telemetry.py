# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU fixtures matching bundled and upstream Quack cache formats."""

import json
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from axolotl.integrations.kernels.sonicmoe_autotune import (
    SonicMoEAutotuneReportCallback,
    collect_sonicmoe_autotune_configs,
)


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 192
    cluster_m: int = 2
    cluster_n: int = 1
    pingpong: bool = True
    swap_ab: bool = False


class QuackAutotuner:
    __module__ = "sonic_moe_abcd.quack.autotuner"

    def __init__(self, module_name, key):
        self.fn = SimpleNamespace(__name__="gemm_gated_tuned", __module__=module_name)
        self.keys = ["activation", "dynamic_scheduler"]
        self.configs = [SimpleNamespace(kwargs={"config": GemmConfig()})]
        self.cache = {key: self.configs[0]}


@pytest.fixture
def modules():
    from axolotl.integrations.kernels.sonicmoe_autotune import _is_sonic_module

    with patch.dict(sys.modules, {k: None for k in sys.modules if _is_sonic_module(k)}):
        yield


@pytest.mark.usefixtures("modules")
@pytest.mark.parametrize(
    "module_name", ["quack.gemm_interface", "sonic_moe_abcd.quack.gemm_interface"]
)
@pytest.mark.parametrize(
    "signature",
    [
        ("swiglu", "False", "torch.Size([1024, 4096])", "[2, 1]", "torch.bfloat16"),
        ("swiglu", False, (1024, 4096), (2, 1), torch.bfloat16),
    ],
)
def test_quack_config_and_shape_signatures(module_name, signature):
    module = ModuleType(module_name)
    module.gemm_gated_tuned = QuackAutotuner(module_name, signature)
    with patch.dict(sys.modules, {module_name: module}):
        coverage = []
        records = collect_sonicmoe_autotune_configs(coverage)
    assert len(records) == 1
    record = records[0]
    assert record["backend"] == "quack"
    assert record["config"]["config"]["tile_n"] == 192
    assert record["config"]["config"]["pingpong"] is True
    assert "num_warps" not in record["config"]
    assert record["key"]["declared_names"] == ["activation", "dynamic_scheduler"]
    assert record["key"]["signature"][-1] == "torch.bfloat16"
    assert record["candidate_config_count"] == 1
    assert len(record["candidate_configs_sha256"]) == 64
    assert coverage[0]["cache_entries"] == 1
    json.dumps(records)


@pytest.mark.usefixtures("modules")
def test_aliases_once_and_empty_cache_inventory():
    name = "sonic_moe_abcd.quack.gemm_interface"
    module = ModuleType(name)
    module.tuned = QuackAutotuner(name, ("swiglu",))
    module.empty = QuackAutotuner(name, ())
    module.empty.fn = SimpleNamespace(__name__="empty", __module__=name)
    module.empty.cache.clear()
    with patch.dict(sys.modules, {name: module, "quack.gemm_interface": module}):
        coverage = []
        records = collect_sonicmoe_autotune_configs(coverage)
    assert len(records) == 1
    assert len(coverage) == 2
    assert coverage[1]["status"] == "cache_empty"
    assert records[0]["key"]["signature"] == ["swiglu"]


@pytest.mark.usefixtures("modules")
def test_unsupported_values_never_serialize_tensor_data():
    name = "quack.gemm_interface"
    module = ModuleType(name)
    module.tuned = QuackAutotuner(name, (False,))
    module.tuned.configs[0].kwargs["unexpected"] = torch.tensor([123.0])
    with patch.dict(sys.modules, {name: module}):
        coverage = []
        assert collect_sonicmoe_autotune_configs(coverage) == []
    assert coverage[0]["unsupported_entries"] == 1


@pytest.mark.usefixtures("modules")
def test_bundled_triton_cache():
    name = "sonic_moe_abcd.functional.triton_kernels"
    module = ModuleType(name)
    module.tuned = SimpleNamespace(
        fn=SimpleNamespace(__name__="routing", __module__=name),
        keys=["N"],
        cache={
            (128,): SimpleNamespace(kwargs={"BLOCK": 64}, num_warps=4, num_stages=2)
        },
    )
    with patch.dict(sys.modules, {name: module}):
        records = collect_sonicmoe_autotune_configs()
    assert records[0]["backend"] == "triton"
    assert records[0]["key"] == {"N": 128}
    assert records[0]["config"]["num_warps"] == 4


@pytest.mark.usefixtures("modules")
def test_callback_winners_and_final_coverage():
    name = "quack.gemm_interface"
    module = ModuleType(name)
    module.tuned = QuackAutotuner(name, (False,))
    with (
        patch.dict(sys.modules, {name: module}),
        patch("axolotl.telemetry.manager.TelemetryManager") as tm,
        patch(
            "axolotl.integrations.kernels.sonicmoe_autotune._get_gpu_info",
            return_value={},
        ),
        patch(
            "axolotl.integrations.kernels.sonicmoe_autotune._get_smem_capacity",
            return_value={},
        ),
    ):
        tm.get_instance.return_value.enabled = True
        callback = SonicMoEAutotuneReportCallback()
        state = SimpleNamespace(global_step=1)
        callback.on_step_end(None, state, None)
        callback.on_step_end(None, SimpleNamespace(global_step=2), None)
        callback.on_train_end(None, state, None)
        calls = tm.get_instance.return_value.send_event.call_args_list
    assert len(calls) == 2
    assert all(call.kwargs["event_type"] == "sonicmoe-autotune" for call in calls)
    assert calls[0].kwargs["properties"]["kernel_count"] == 1
    assert calls[1].kwargs["properties"]["kernel_count"] == 0
    assert (
        calls[1].kwargs["properties"]["autotuner_coverage"][0]["status"]
        == "cache_populated"
    )


@pytest.mark.usefixtures("modules")
def test_no_sonicmoe_modules_no_events():
    with patch("axolotl.telemetry.manager.TelemetryManager") as tm:
        tm.get_instance.return_value.enabled = True
        callback = SonicMoEAutotuneReportCallback()
        callback.on_train_end(None, SimpleNamespace(global_step=1), None)
        tm.get_instance.return_value.send_event.assert_not_called()


@pytest.mark.usefixtures("modules")
def test_empty_loaded_cache_still_reports_final_coverage():
    name = "quack.gemm_interface"
    module = ModuleType(name)
    module.tuned = QuackAutotuner(name, ())
    module.tuned.cache.clear()
    with (
        patch.dict(sys.modules, {name: module}),
        patch("axolotl.telemetry.manager.TelemetryManager") as tm,
        patch.object(SonicMoEAutotuneReportCallback, "_hardware_info", return_value={}),
    ):
        tm.get_instance.return_value.enabled = True
        callback = SonicMoEAutotuneReportCallback()
        callback.on_train_end(None, SimpleNamespace(global_step=1), None)
        props = tm.get_instance.return_value.send_event.call_args.kwargs["properties"]
    assert props["kernel_count"] == 0
    assert props["final_snapshot"] is True
    assert props["autotuner_coverage"][0]["status"] == "cache_empty"


def test_disabled_never_collects_sonicmoe():
    with (
        patch("axolotl.telemetry.manager.TelemetryManager") as tm,
        patch(
            "axolotl.integrations.kernels.sonicmoe_autotune.collect_sonicmoe_autotune_configs"
        ) as collect,
    ):
        tm.get_instance.return_value.enabled = False
        callback = SonicMoEAutotuneReportCallback()
        callback.on_step_end(None, SimpleNamespace(global_step=1), None)
        callback.on_train_end(None, SimpleNamespace(global_step=1), None)
        collect.assert_not_called()
        tm.get_instance.return_value.send_event.assert_not_called()
