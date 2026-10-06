# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for bounded autotune telemetry and provenance."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from axolotl.integrations.kernels.autotune_collector import collect_autotune_configs
from axolotl.kernels.autotune_reporting import (
    AutotuneTelemetryCallback,
    autotuner_metadata,
)


def _record(size):
    return {"kernel": "kernel", "key": {"N": size}, "config": {"num_warps": 4}}


@pytest.fixture
def reporting():
    callback = AutotuneTelemetryCallback()
    callback._collect_configs = MagicMock(return_value=[])
    with patch("axolotl.telemetry.manager.TelemetryManager") as manager:
        manager.get_instance.return_value.enabled = True
        yield callback, manager.get_instance.return_value


def _step(callback, step):
    callback.on_step_end(None, SimpleNamespace(global_step=step), None)


def test_late_records_deduplicated_and_final_flush(reporting):
    callback, manager = reporting
    for step in range(1, 25):
        _step(callback, step)
    assert callback._collect_configs.call_count == 5
    callback._collect_configs.return_value = [_record(1)]
    _step(callback, 25)
    callback._collect_configs.return_value = [_record(1), _record(2)]
    _step(callback, 100)
    callback._collect_configs.return_value.append(_record(3))
    callback.on_train_end(None, SimpleNamespace(global_step=500), None)
    callback.on_train_end(None, SimpleNamespace(global_step=500), None)
    events = [call.kwargs["properties"] for call in manager.send_event.call_args_list]
    assert [event["kernels"] for event in events] == [[_record(i)] for i in (1, 2, 3)]
    assert events[-1]["final_snapshot"]
    assert events[-1]["global_step"] == 500
    assert "torch" in events[-1]["runtime_versions"]


def test_resume_skips_past_checkpoints(reporting):
    callback, _ = reporting
    for step in (1000, 1000, 1001):
        _step(callback, step)
    assert callback._collect_configs.call_count == 1
    callback.on_train_end(None, SimpleNamespace(global_step=1001), None)
    assert callback._collect_configs.call_count == 2


def test_disabled_does_not_inspect_caches(reporting):
    callback, manager = reporting
    manager.enabled = False
    _step(callback, 1)
    callback.on_train_end(None, SimpleNamespace(global_step=1), None)
    callback._collect_configs.assert_not_called()
    manager.send_event.assert_not_called()


def test_event_and_run_limits_preserve_pending_records(reporting):
    callback, manager = reporting
    callback.max_records_per_event = 2
    callback.max_records_per_run = 3
    callback._collect_configs.return_value = [_record(i) for i in range(5)]
    for step in (1, 2, 3):
        _step(callback, step)
    events = [call.kwargs["properties"] for call in manager.send_event.call_args_list]
    assert [event["kernels"] for event in events] == [
        [_record(0), _record(1)],
        [_record(2)],
    ]
    assert [event["unreported_record_count"] for event in events] == [3, 2]
    assert callback._collect_configs.call_count == 2


def test_same_snapshot_duplicates_and_config_changes(reporting):
    callback, manager = reporting
    callback._collect_configs.return_value = [_record(1), _record(1)]
    _step(callback, 1)
    _step(callback, 2)
    callback._collect_configs.return_value = [_record(1)]
    callback._collect_configs.return_value[0]["config"]["num_warps"] = 8
    _step(callback, 3)
    assert manager.send_event.call_count == 2
    assert len(callback._seen) == 2


@pytest.mark.parametrize(
    "module_name,kernel_name,keys,values",
    [
        (
            "axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped",
            "_grouped_dx_fp8_kernel",
            ["N", "K", "BM"],
            (512, 1024, 32),
        ),
        (
            "scattermoe_lora_abc123.multi_lora",
            "_grouped_gram_kernel",
            ["M_BUCKET", "WIDE", "RANK_IS_I"],
            (2048, 4096, True),
        ),
        (
            "scattermoe_lora_abc123.kernels.ops",
            "_scatter2scatter",
            ["M_BUCKET", "N", "K"],
            (2048, 4096, 512),
        ),
        (
            "axolotl.monkeypatch.attention.flash_attn_d512",
            "_fwd",
            ["N_CTX", "HEAD_DIM", "CAUSAL", "VARLEN"],
            (1024, 512, True, False),
        ),
    ],
)
def test_new_kernel_discovery(module_name, kernel_name, keys, values):
    config = SimpleNamespace(kwargs={"BLOCK_M": 32}, num_warps=4, num_stages=2)
    module = ModuleType(module_name)
    setattr(
        module,
        kernel_name,
        SimpleNamespace(
            cache={values: config},
            base_fn=SimpleNamespace(__name__=kernel_name),
            keys=keys,
            configs=[config],
        ),
    )
    with patch.dict(sys.modules, {module_name: module}):
        records = [
            r for r in collect_autotune_configs() if r["module_fqn"] == module_name
        ]
    assert len(records) == 1
    assert records[0]["key"] == dict(zip(keys, values, strict=True))
    assert records[0]["candidate_config_count"] == 1


def test_provenance_fingerprints_without_source_or_paths(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("kernel source")
    module = SimpleNamespace(__file__=str(source))
    config = SimpleNamespace(kwargs={"BLOCK_M": 32}, num_warps=4, num_stages=2)
    autotuner = SimpleNamespace(configs=[config])
    metadata = autotuner_metadata(autotuner, module)
    assert set(metadata) == {
        "module_source_sha256",
        "candidate_configs_sha256",
        "candidate_config_count",
    }
    assert len(metadata["module_source_sha256"]) == 64
    config.num_warps = 8
    changed = autotuner_metadata(autotuner, module)
    assert changed["candidate_configs_sha256"] != metadata["candidate_configs_sha256"]
    assert changed["module_source_sha256"] == metadata["module_source_sha256"]
    assert autotuner_metadata(SimpleNamespace(), SimpleNamespace()) == {}


def test_empty_triton_cache_has_coverage_without_a_winner():
    module_name = "scattermoe_lora_test.kernels.ops"
    module = ModuleType(module_name)
    module.kernel = SimpleNamespace(
        cache={}, base_fn=SimpleNamespace(__name__="kernel"), keys=["N"]
    )
    with patch.dict(sys.modules, {module_name: module}):
        coverage = []
        records = collect_autotune_configs(coverage)
    assert not any(record["module_fqn"] == module_name for record in records)
    entry = next(item for item in coverage if item["module_fqn"] == module_name)
    assert entry["status"] == "cache_empty"
    assert entry["cache_entries"] == 0
