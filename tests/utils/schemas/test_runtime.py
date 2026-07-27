"""Tests for the runtime (--runtime file) config schema."""

import pytest
from pydantic import ValidationError

from axolotl.utils.schemas.runtime import (
    RayLauncherConfig,
    RuntimeConfig,
)


class TestRuntimeConfigSchema:
    """Validation and defaults for RuntimeConfig."""

    def test_empty_config_defaults(self):
        cfg = RuntimeConfig.model_validate({})
        assert cfg.launcher is None
        assert cfg.env == {}
        assert cfg.ray is None
        assert cfg.resolve_launcher_choice() is None

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValidationError):
            RuntimeConfig.model_validate({"luancher": "ray"})

    def test_unknown_block_key_rejected(self):
        with pytest.raises(ValidationError):
            RuntimeConfig.model_validate({"ray": {"num_wrokers": 2}})

    def test_env_values_stringified(self):
        cfg = RuntimeConfig.model_validate({"env": {"NCCL_DEBUG": 1, "FOO": True}})
        assert cfg.env == {"NCCL_DEBUG": "1", "FOO": "True"}

    def test_ray_defaults(self):
        cfg = RuntimeConfig.model_validate({"ray": {}})
        assert cfg.ray.num_workers == "auto"
        assert cfg.ray.resources_per_worker == {"GPU": 1.0}
        assert cfg.ray.detach is False


class TestLauncherDerivation:
    """resolve_launcher_choice behavior."""

    def test_explicit_launcher_wins(self):
        cfg = RuntimeConfig.model_validate({"launcher": "torchrun", "ray": {}})
        assert cfg.resolve_launcher_choice() == "torchrun"

    def test_single_block_derives_launcher(self):
        cfg = RuntimeConfig.model_validate({"torchrun": {"nnodes": 2}})
        assert cfg.resolve_launcher_choice() == "torchrun"

    def test_multiple_blocks_without_launcher_raises(self):
        cfg = RuntimeConfig.model_validate({"ray": {}, "torchrun": {}})
        with pytest.raises(ValueError, match="multiple launcher blocks"):
            cfg.resolve_launcher_choice()

    def test_foreign_block_warns_but_validates(self, caplog):
        with caplog.at_level("WARNING"):
            cfg = RuntimeConfig.model_validate({"launcher": "ray", "torchrun": {}})
        assert cfg.launcher == "ray"
        assert any("ignoring blocks" in rec.message for rec in caplog.records)


class TestRayLauncherConfig:
    """Ray block validation."""

    def test_num_workers_zero_rejected(self):
        with pytest.raises(ValidationError, match="num_workers"):
            RayLauncherConfig.model_validate({"num_workers": 0})

    def test_num_workers_auto(self):
        assert RayLauncherConfig.model_validate({"num_workers": "auto"}).num_workers == "auto"

    @pytest.mark.parametrize(
        "address,expected",
        [
            (None, False),
            ("auto", False),
            ("ray://head:10001", False),
            ("http://head:8265", True),
            ("https://head:8265", True),
        ],
    )
    def test_is_job_submission(self, address, expected):
        cfg = RayLauncherConfig.model_validate({"address": address})
        assert cfg.is_job_submission is expected

    def test_runtime_env_extra_keys_pass_through(self):
        cfg = RayLauncherConfig.model_validate(
            {"runtime_env": {"env_vars": {"A": 1}, "conda": "myenv"}}
        )
        dumped = cfg.runtime_env.model_dump(exclude_none=True)
        assert dumped["conda"] == "myenv"
        assert dumped["env_vars"] == {"A": "1"}

    def test_accelerator_type_resources(self):
        cfg = RayLauncherConfig.model_validate(
            {"resources_per_worker": {"GPU": 1, "accelerator_type:H100": 0.001}}
        )
        assert cfg.resources_per_worker["accelerator_type:H100"] == 0.001


class TestFromFile:
    """RuntimeConfig.from_file behavior."""

    def test_round_trip(self, tmp_path):
        path = tmp_path / "runtime.yaml"
        path.write_text(
            "launcher: ray\nray:\n  num_workers: 4\n  resources_per_worker:\n    GPU: 1\n"
        )
        cfg = RuntimeConfig.from_file(str(path))
        assert cfg.launcher == "ray"
        assert cfg.ray.num_workers == 4

    def test_non_mapping_rejected(self, tmp_path):
        path = tmp_path / "runtime.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            RuntimeConfig.from_file(str(path))

    def test_empty_file_ok(self, tmp_path):
        path = tmp_path / "runtime.yaml"
        path.write_text("")
        cfg = RuntimeConfig.from_file(str(path))
        assert cfg.resolve_launcher_choice() is None
