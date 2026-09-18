"""Tests for launcher resolution (CLI > runtime file > legacy YAML > default)."""

import click
import pytest

from axolotl.cli.launchers.base import merge_launcher_args, pop_legacy_launcher_kwargs
from axolotl.cli.launchers.resolve import peek_legacy_use_ray, resolve_launch


@pytest.fixture
def train_cfg(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("base_model: HuggingFaceTB/SmolLM2-135M\n")
    return str(path)


@pytest.fixture
def ray_train_cfg(tmp_path):
    path = tmp_path / "config-ray.yaml"
    path.write_text("base_model: HuggingFaceTB/SmolLM2-135M\nuse_ray: true\n")
    return str(path)


def write_runtime(tmp_path, content: str) -> str:
    path = tmp_path / "runtime.yaml"
    path.write_text(content)
    return str(path)


class TestLauncherPrecedence:
    """Precedence: CLI > runtime file > YAML use_ray peek > default."""

    def test_default_is_accelerate(self, train_cfg):
        resolved = resolve_launch(train_cfg, None, None, {}, [])
        assert resolved.launcher == "accelerate"
        assert resolved.launcher_args == []
        assert resolved.env == {}

    def test_explicit_cli_launcher(self, train_cfg):
        resolved = resolve_launch(train_cfg, None, "torchrun", {}, [])
        assert resolved.launcher == "torchrun"

    def test_cli_beats_runtime_file(self, tmp_path, train_cfg):
        runtime = write_runtime(tmp_path, "launcher: torchrun\n")
        resolved = resolve_launch(train_cfg, runtime, "accelerate", {}, [])
        assert resolved.launcher == "accelerate"

    def test_runtime_file_launcher(self, tmp_path, train_cfg):
        runtime = write_runtime(tmp_path, "launcher: ray\n")
        kwargs = {}
        resolved = resolve_launch(train_cfg, runtime, None, kwargs, [])
        assert resolved.launcher == "ray"
        assert kwargs["use_ray"] is True

    def test_runtime_single_block_derives(self, tmp_path, train_cfg):
        runtime = write_runtime(tmp_path, "torchrun:\n  nnodes: 2\n")
        resolved = resolve_launch(train_cfg, runtime, None, {}, [])
        assert resolved.launcher == "torchrun"

    def test_runtime_ambiguous_blocks_usage_error(self, tmp_path, train_cfg):
        runtime = write_runtime(tmp_path, "ray: {}\ntorchrun: {}\n")
        with pytest.raises(click.UsageError, match="multiple launcher blocks"):
            resolve_launch(train_cfg, runtime, None, {}, [])

    def test_yaml_use_ray_selects_ray(self, ray_train_cfg):
        kwargs = {}
        resolved = resolve_launch(ray_train_cfg, None, None, kwargs, [])
        assert resolved.launcher == "ray"
        assert kwargs["use_ray"] is True

    def test_no_use_ray_flag_vetoes_yaml(self, ray_train_cfg):
        kwargs = {"use_ray": False}
        resolved = resolve_launch(ray_train_cfg, None, None, kwargs, [])
        assert resolved.launcher == "accelerate"

    def test_explicit_launcher_neutralizes_yaml_use_ray(self, ray_train_cfg):
        kwargs = {}
        resolved = resolve_launch(ray_train_cfg, None, "accelerate", kwargs, [])
        assert resolved.launcher == "accelerate"
        assert kwargs["use_ray"] is False

    def test_runtime_launcher_beats_yaml_use_ray(self, tmp_path, ray_train_cfg):
        runtime = write_runtime(tmp_path, "launcher: torchrun\n")
        kwargs = {}
        resolved = resolve_launch(ray_train_cfg, runtime, None, kwargs, [])
        assert resolved.launcher == "torchrun"
        assert kwargs["use_ray"] is False


class TestUseRayFlag:
    """Legacy --use-ray handling."""

    def test_use_ray_flag_maps_to_ray(self, train_cfg, caplog):
        kwargs = {"use_ray": True}
        with caplog.at_level("WARNING"):
            resolved = resolve_launch(train_cfg, None, None, kwargs, [])
        assert resolved.launcher == "ray"
        assert any("deprecated" in rec.message for rec in caplog.records)

    def test_use_ray_conflicts_with_other_launcher(self, train_cfg):
        with pytest.raises(click.UsageError, match="conflicts"):
            resolve_launch(train_cfg, None, "torchrun", {"use_ray": True}, [])

    def test_use_ray_with_launcher_ray_ok(self, train_cfg):
        resolved = resolve_launch(train_cfg, None, "ray", {"use_ray": True}, [])
        assert resolved.launcher == "ray"

    def test_ray_rejected_when_unsupported(self, tmp_path, train_cfg):
        runtime = write_runtime(tmp_path, "launcher: ray\n")
        with pytest.raises(click.UsageError, match="only supported"):
            resolve_launch(train_cfg, runtime, None, {}, [], supports_ray=False)

    def test_yaml_use_ray_ignored_when_unsupported(self, ray_train_cfg):
        resolved = resolve_launch(ray_train_cfg, None, None, {}, [], supports_ray=False)
        assert resolved.launcher == "accelerate"


class TestRuntimeLauncherArgs:
    """Runtime block → launcher argv."""

    def test_torchrun_block_args(self, tmp_path, train_cfg):
        runtime = write_runtime(
            tmp_path,
            "torchrun:\n  nnodes: 2\n  nproc_per_node: 8\n  rdzv_endpoint: 10.0.0.4:29400\n",
        )
        resolved = resolve_launch(train_cfg, runtime, None, {}, [])
        assert "--nnodes=2" in resolved.launcher_args
        assert "--nproc_per_node=8" in resolved.launcher_args
        assert "--rdzv_endpoint=10.0.0.4:29400" in resolved.launcher_args

    def test_accelerate_block_args(self, tmp_path, train_cfg):
        runtime = write_runtime(
            tmp_path,
            "accelerate:\n  num_processes: 16\n  num_machines: 2\n  main_process_ip: 10.0.0.4\n",
        )
        resolved = resolve_launch(train_cfg, runtime, None, {}, [])
        assert "--num_processes=16" in resolved.launcher_args
        assert "--num_machines=2" in resolved.launcher_args
        assert "--main_process_ip=10.0.0.4" in resolved.launcher_args

    def test_passthrough_overrides_runtime_block(self, tmp_path, train_cfg):
        runtime = write_runtime(tmp_path, "torchrun:\n  nproc_per_node: 8\n")
        resolved = resolve_launch(
            train_cfg, runtime, None, {}, ["--nproc-per-node", "4"]
        )
        assert "--nproc_per_node=8" not in resolved.launcher_args
        assert resolved.launcher_args[-2:] == ["--nproc-per-node", "4"]

    def test_foreign_block_produces_no_args(self, tmp_path, train_cfg):
        runtime = write_runtime(
            tmp_path, "launcher: accelerate\ntorchrun:\n  nnodes: 2\n"
        )
        resolved = resolve_launch(train_cfg, runtime, None, {}, [])
        assert resolved.launcher == "accelerate"
        assert resolved.launcher_args == []

    def test_env_block_returned(self, tmp_path, train_cfg):
        runtime = write_runtime(
            tmp_path, "launcher: torchrun\nenv:\n  NCCL_DEBUG: WARN\n"
        )
        resolved = resolve_launch(train_cfg, runtime, None, {}, [])
        assert resolved.env == {"NCCL_DEBUG": "WARN"}


class TestLegacyLauncherKwargs:
    """num_processes / main_process_port popping and translation."""

    def test_accelerate_translation(self, train_cfg):
        kwargs = {"num_processes": 4, "main_process_port": 29501}
        resolved = resolve_launch(train_cfg, None, "accelerate", kwargs, [])
        assert "--num_processes=4" in resolved.launcher_args
        assert "--main_process_port=29501" in resolved.launcher_args
        assert "num_processes" not in kwargs

    def test_torchrun_translation(self, train_cfg):
        kwargs = {"num_processes": 4, "main_process_port": 29501}
        resolved = resolve_launch(train_cfg, None, "torchrun", kwargs, [])
        assert "--nproc_per_node=4" in resolved.launcher_args
        assert "--master_port=29501" in resolved.launcher_args
        assert "num_processes" not in kwargs

    def test_ray_drops_with_warning(self, train_cfg, caplog):
        kwargs = {"use_ray": True, "num_processes": 4}
        with caplog.at_level("WARNING"):
            resolved = resolve_launch(train_cfg, None, None, kwargs, [])
        assert resolved.launcher_args == []
        assert "num_processes" not in kwargs
        assert any("ignored for launcher=ray" in rec.message for rec in caplog.records)


class TestRayKwargsInjection:
    """Runtime ray block → legacy flat cfg kwargs."""

    def test_runtime_values_injected(self, tmp_path, train_cfg):
        runtime = write_runtime(
            tmp_path,
            "launcher: ray\nray:\n  num_workers: 8\n  run_name: myrun\n"
            "  resources_per_worker:\n    GPU: 1\n",
        )
        kwargs = {}
        resolve_launch(train_cfg, runtime, None, kwargs, [])
        assert kwargs["use_ray"] is True
        assert kwargs["ray_num_workers"] == 8
        assert kwargs["ray_run_name"] == "myrun"
        assert kwargs["resources_per_worker"] == {"GPU": 1}

    def test_cli_flags_beat_runtime_file(self, tmp_path, train_cfg):
        runtime = write_runtime(tmp_path, "launcher: ray\nray:\n  num_workers: 8\n")
        kwargs = {"ray_num_workers": 2}
        resolve_launch(train_cfg, runtime, None, kwargs, [])
        assert kwargs["ray_num_workers"] == 2

    def test_auto_num_workers_not_injected(self, tmp_path, train_cfg):
        runtime = write_runtime(tmp_path, "launcher: ray\nray:\n  num_workers: auto\n")
        kwargs = {}
        resolve_launch(train_cfg, runtime, None, kwargs, [])
        assert "ray_num_workers" not in kwargs

    def test_resources_per_worker_string_coerced(self, train_cfg):
        kwargs = {"use_ray": True, "resources_per_worker": "{GPU: 1}"}
        resolve_launch(train_cfg, None, None, kwargs, [])
        assert kwargs["resources_per_worker"] == {"GPU": 1}

    def test_resources_per_worker_garbage_rejected(self, train_cfg):
        kwargs = {"use_ray": True, "resources_per_worker": "not a mapping"}
        with pytest.raises(click.BadParameter):
            resolve_launch(train_cfg, None, None, kwargs, [])

    def test_ray_rejects_passthrough_args(self, train_cfg):
        with pytest.raises(click.UsageError, match="passthrough"):
            resolve_launch(train_cfg, None, "ray", {}, ["--nnodes=2"])


class TestHelpers:
    """merge_launcher_args / pop_legacy_launcher_kwargs / peek_legacy_use_ray."""

    def test_merge_normalizes_flag_names(self):
        merged = merge_launcher_args(
            ["--nproc_per_node=2", "--nnodes=1"], ["--nproc-per-node", "4"]
        )
        assert merged == ["--nnodes=1", "--nproc-per-node", "4"]

    def test_merge_equals_form_passthrough(self):
        merged = merge_launcher_args(["--nnodes=1"], ["--nnodes=3"])
        assert merged == ["--nnodes=3"]

    def test_pop_legacy_kwargs(self):
        kwargs = {"num_processes": 2, "main_process_port": 1234, "learning_rate": 1e-4}
        legacy = pop_legacy_launcher_kwargs(kwargs)
        assert legacy == {"num_processes": 2, "main_process_port": 1234}
        assert kwargs == {"learning_rate": 1e-4}

    def test_peek_missing_file(self):
        assert peek_legacy_use_ray("/nonexistent/config.yaml") is False

    def test_peek_non_mapping(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        assert peek_legacy_use_ray(str(path)) is False

    def test_peek_use_ray_true(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text("use_ray: true\n")
        assert peek_legacy_use_ray(str(path)) is True


def test_invalid_runtime_file_is_usage_error(tmp_path, train_cfg):
    runtime = write_runtime(tmp_path, "launcher: bogus\n")
    with pytest.raises(click.UsageError, match="invalid runtime config"):
        resolve_launch(train_cfg, runtime, None, {}, [])
