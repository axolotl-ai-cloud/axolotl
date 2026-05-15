"""Tests for GPU capability detection in `load_cfg` and `ray_train_func`."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from axolotl.cli.config import load_cfg
from axolotl.cli.train import ray_train_func

_BASE_CONFIG = """
base_model: HuggingFaceTB/SmolLM2-135M
datasets:
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
sequence_len: 2048
max_steps: 1
micro_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1e-3
special_tokens:
  pad_token: <|endoftext|>
"""


def _write_cfg(tmp_path: Path, extra: str = "") -> Path:
    """Write the base test config (plus any extra YAML lines) to a temp file."""
    path = tmp_path / "config.yml"
    path.write_text(_BASE_CONFIG + extra)
    return path


def _patch_load_cfg_dependencies(monkeypatch, validate_mock=None):
    """Stub everything `load_cfg` does after validation so the test can focus
    on whether GPU capabilities were probed on the driver.

    If ``validate_mock`` is given, it is installed as ``validate_config`` so the
    test can inspect the arguments it was called with; otherwise a simple
    identity stub is used.
    """
    monkeypatch.setattr(
        "axolotl.cli.config.validate_config",
        validate_mock if validate_mock is not None else (lambda cfg, **_: cfg),
    )
    monkeypatch.setattr("axolotl.cli.config.normalize_config", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.normalize_cfg_datasets", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.prepare_debug_log", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.prepare_optim_env", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.setup_wandb_env_vars", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.setup_mlflow_env_vars", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.setup_comet_env_vars", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.setup_trackio_env_vars", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.plugin_set_cfg", lambda *_: None)
    monkeypatch.setattr(
        "axolotl.cli.config.TELEMETRY_MANAGER.send_event", lambda *_, **__: None
    )


def test_load_cfg_probes_capabilities_by_default(tmp_path, monkeypatch):
    """Without `use_ray`, `load_cfg` probes GPU capabilities on the local host
    and passes the results into `validate_config`."""
    validate_mock = MagicMock(side_effect=lambda cfg, **_: cfg)
    _patch_load_cfg_dependencies(monkeypatch, validate_mock=validate_mock)
    config_path = _write_cfg(tmp_path)

    with patch("axolotl.cli.config.gpu_capabilities") as mock_caps:
        mock_caps.return_value = ({"bf16": False}, {"torch_version": "2.6.0"})
        load_cfg(str(config_path))

    mock_caps.assert_called_once()
    _, kwargs = validate_mock.call_args
    assert kwargs["capabilities"] == {"bf16": False}
    assert kwargs["env_capabilities"] == {"torch_version": "2.6.0"}


def test_load_cfg_skips_capabilities_under_ray(tmp_path, monkeypatch):
    """With `use_ray: true`, capability detection is deferred to the worker
    and `validate_config` receives `None` for both capability dicts."""
    validate_mock = MagicMock(side_effect=lambda cfg, **_: cfg)
    _patch_load_cfg_dependencies(monkeypatch, validate_mock=validate_mock)
    config_path = _write_cfg(tmp_path, "use_ray: true\nray_num_workers: 1\n")

    with patch("axolotl.cli.config.gpu_capabilities") as mock_caps:
        load_cfg(str(config_path))

    mock_caps.assert_not_called()
    _, kwargs = validate_mock.call_args
    assert kwargs["capabilities"] is None
    assert kwargs["env_capabilities"] is None


def test_ray_train_func_validates_with_worker_capabilities(monkeypatch):
    """`ray_train_func` must probe `gpu_capabilities()` on the worker and feed
    the result into `validate_config` before training runs."""
    cfg_dict = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "gradient_accumulation_steps": 1,
    }

    validate_mock = MagicMock(side_effect=lambda cfg, **_: cfg)
    do_train_mock = MagicMock()
    accelerator_mock = MagicMock()

    monkeypatch.setattr("axolotl.cli.train.validate_config", validate_mock)
    monkeypatch.setattr("axolotl.cli.train.do_train", do_train_mock)
    monkeypatch.setattr("axolotl.cli.train.prepare_optim_env", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.train.normalize_config", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.train.resolve_dtype", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.train.Accelerator", accelerator_mock)

    with patch("axolotl.cli.train.gpu_capabilities") as mock_caps:
        mock_caps.return_value = (
            {"bf16": True, "fp8": False, "tf32": True, "compute_capability": "sm_90"},
            {"torch_version": "2.6.0"},
        )
        ray_train_func({"cfg": cfg_dict, "cli_args": MagicMock()})

    mock_caps.assert_called_once()
    validate_mock.assert_called_once()
    _, kwargs = validate_mock.call_args
    assert kwargs["capabilities"] == {
        "bf16": True,
        "fp8": False,
        "tf32": True,
        "compute_capability": "sm_90",
    }
    assert kwargs["env_capabilities"] == {"torch_version": "2.6.0"}
    do_train_mock.assert_called_once()


def test_ray_train_func_registers_plugins_before_validate_config(monkeypatch):
    """Regression: plugins must be registered before `validate_config` so the
    plugin-extended pydantic schema is in scope. Otherwise `merge_input_args`
    sees an empty PluginManager on the worker and `model_dump(exclude_none=True)`
    silently drops plugin-specific cfg fields.
    """
    cfg_dict = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "gradient_accumulation_steps": 1,
        "plugins": ["axolotl.integrations.liger.LigerPlugin"],
    }

    parent = MagicMock()
    parent.validate_config.side_effect = lambda cfg, **_: cfg

    # Patch at the source module so a local `from axolotl.cli.config import ...`
    # inside the function also resolves to the mock; also patch the train module
    # for top-level imports (raising=False keeps it tolerant of either style).
    monkeypatch.setattr("axolotl.cli.config.prepare_plugins", parent.prepare_plugins)
    monkeypatch.setattr("axolotl.cli.config.plugin_set_cfg", parent.plugin_set_cfg)
    monkeypatch.setattr(
        "axolotl.cli.train.prepare_plugins", parent.prepare_plugins, raising=False
    )
    monkeypatch.setattr(
        "axolotl.cli.train.plugin_set_cfg", parent.plugin_set_cfg, raising=False
    )
    monkeypatch.setattr("axolotl.cli.train.validate_config", parent.validate_config)
    monkeypatch.setattr("axolotl.cli.train.gpu_capabilities", lambda: ({}, {}))
    monkeypatch.setattr("axolotl.cli.train.do_train", MagicMock())
    monkeypatch.setattr("axolotl.cli.train.prepare_optim_env", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.train.normalize_config", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.train.resolve_dtype", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.train.Accelerator", MagicMock())

    ray_train_func({"cfg": cfg_dict, "cli_args": MagicMock()})

    # Filter to the calls we care about, in the order they happened.
    ordered = [
        call[0]
        for call in parent.mock_calls
        if call[0] in ("prepare_plugins", "validate_config", "plugin_set_cfg")
    ]
    assert ordered == ["prepare_plugins", "validate_config", "plugin_set_cfg"], (
        f"Expected prepare_plugins -> validate_config -> plugin_set_cfg; got {ordered}"
    )


def test_ray_train_func_skips_plugin_registration_when_no_plugins(monkeypatch):
    """When no plugins are configured, neither `prepare_plugins` nor
    `plugin_set_cfg` should be invoked on the worker."""
    cfg_dict = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "gradient_accumulation_steps": 1,
    }

    prepare_plugins_mock = MagicMock()
    plugin_set_cfg_mock = MagicMock()

    monkeypatch.setattr("axolotl.cli.train.prepare_plugins", prepare_plugins_mock)
    monkeypatch.setattr("axolotl.cli.train.plugin_set_cfg", plugin_set_cfg_mock)
    monkeypatch.setattr(
        "axolotl.cli.train.validate_config", MagicMock(side_effect=lambda cfg, **_: cfg)
    )
    monkeypatch.setattr("axolotl.cli.train.gpu_capabilities", lambda: ({}, {}))
    monkeypatch.setattr("axolotl.cli.train.do_train", MagicMock())
    monkeypatch.setattr("axolotl.cli.train.prepare_optim_env", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.train.normalize_config", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.train.resolve_dtype", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.train.Accelerator", MagicMock())

    ray_train_func({"cfg": cfg_dict, "cli_args": MagicMock()})

    prepare_plugins_mock.assert_not_called()
    plugin_set_cfg_mock.assert_not_called()
