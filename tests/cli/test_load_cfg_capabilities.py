"""Tests for GPU capability detection in `load_cfg`."""

from pathlib import Path
from unittest.mock import patch

from axolotl.cli.config import load_cfg

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


def _patch_load_cfg_dependencies(monkeypatch):
    """Stub everything `load_cfg` does after validation so the test can focus
    on whether GPU capabilities were probed on the driver."""
    monkeypatch.setattr("axolotl.cli.config.validate_config", lambda cfg, **_: cfg)
    monkeypatch.setattr("axolotl.cli.config.normalize_config", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.normalize_cfg_datasets", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.prepare_debug_log", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.prepare_optim_env", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.setup_wandb_env_vars", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.setup_mlflow_env_vars", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.setup_comet_env_vars", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.setup_trackio_env_vars", lambda *_: None)
    monkeypatch.setattr("axolotl.cli.config.plugin_set_cfg", lambda *_: None)


def test_load_cfg_probes_capabilities_by_default(tmp_path, monkeypatch):
    """Without `use_ray`, `load_cfg` probes GPU capabilities on the local host."""
    _patch_load_cfg_dependencies(monkeypatch)
    config_path = _write_cfg(tmp_path)

    with patch("axolotl.cli.config.gpu_capabilities") as mock_caps:
        mock_caps.return_value = ({"bf16": False}, {"torch_version": "2.6.0"})
        load_cfg(str(config_path))

    mock_caps.assert_called_once()


def test_load_cfg_skips_capabilities_under_ray(tmp_path, monkeypatch):
    """With `use_ray: true`, capability detection is deferred to the worker."""
    _patch_load_cfg_dependencies(monkeypatch)
    config_path = _write_cfg(tmp_path, "use_ray: true\nray_num_workers: 1\n")

    with patch("axolotl.cli.config.gpu_capabilities") as mock_caps:
        load_cfg(str(config_path))

    mock_caps.assert_not_called()
