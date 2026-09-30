"""Tests for ConfigPath: config arguments accept local files and HTTPS URLs.

Covers https://github.com/axolotl-ai-cloud/axolotl/issues/4041: `axolotl train`
rejected HTTPS config URLs at Click parsing even though `docs/cli.qmd` says
"The config file can be local or a URL to a raw YAML file".
"""

from unittest.mock import MagicMock, patch

import pytest
import yaml
from click import UsageError

from axolotl.cli.main import ConfigPath, cli

CONFIG_URL = (
    "https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main"
    "/examples/llama-3/lora-1b.yml"
)


class TestConfigPath:
    """Unit tests for the ConfigPath click param type."""

    param_type = ConfigPath(exists=True, path_type=str)

    def test_https_url_passes_through_unchanged(self):
        assert self.param_type.convert(CONFIG_URL, None, None) == CONFIG_URL

    def test_existing_local_file_passes(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("base_model: foo\n")
        result = self.param_type.convert(str(config_file), None, None)
        assert result == str(config_file)

    def test_missing_local_file_fails(self):
        with pytest.raises(UsageError, match="does not exist"):
            self.param_type.convert("nonexistent-config.yml", None, None)

    def test_http_url_fails_without_download_support(self):
        with pytest.raises(UsageError, match="does not exist"):
            self.param_type.convert("http://nonexistent.invalid/config.yml", None, None)

    @pytest.mark.parametrize(
        "command",
        [
            "preprocess",
            "train",
            "evaluate",
            "inference",
            "merge-sharded-fsdp-weights",
            "merge-lora",
            "vllm-serve",
            "quantize",
            "export",
        ],
    )
    def test_command_config_uses_config_path(self, command):
        config_param = next(
            p for p in cli.commands[command].params if p.name == "config"
        )
        assert isinstance(config_param.type, ConfigPath)


class TestTrainConfigUrl:
    """End-to-end: `axolotl train <https-url>` passes Click parsing."""

    def test_train_accepts_config_url(self, cli_runner):
        with patch("axolotl.cli.main.launch_training") as mock_launch:
            result = cli_runner.invoke(
                cli,
                ["train", CONFIG_URL, "--launcher", "python"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        mock_launch.assert_called_once()
        assert mock_launch.call_args[0][0] == CONFIG_URL

    def test_train_sweep_downloads_config_url(self, cli_runner, tmp_path):
        sweep_path = tmp_path / "sweep.yml"
        sweep_path.write_text(yaml.dump({"learning_rate": [0.1, 0.2]}))
        response = MagicMock(content=b"base_model: foo\n")
        launched = []

        def record_launch(cfg_file, *_args):
            with open(cfg_file, encoding="utf-8") as fin:
                launched.append(yaml.safe_load(fin))

        with (
            patch("requests.get", return_value=response) as mock_get,
            patch("axolotl.cli.main.launch_training", side_effect=record_launch),
        ):
            result = cli_runner.invoke(
                cli,
                ["train", CONFIG_URL, "--sweep", str(sweep_path)],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        mock_get.assert_called_once()
        assert sorted(cfg["learning_rate"] for cfg in launched) == [0.1, 0.2]
        assert all(cfg["base_model"] == "foo" for cfg in launched)
