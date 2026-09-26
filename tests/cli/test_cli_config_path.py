"""Tests for ConfigPath: config arguments accept local files and HTTPS URLs.

Covers https://github.com/axolotl-ai-cloud/axolotl/issues/4041: `axolotl train`
rejected HTTPS config URLs at Click parsing even though `docs/cli.qmd` says
"The config file can be local or a URL to a raw YAML file".
"""

from unittest.mock import patch

import pytest
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
        # check_remote_config only downloads https:// URLs, so http:// URLs
        # must keep failing at parse time instead of later in load_cfg.
        # requests.get is mocked to prove no network traffic leaves CI;
        # nonexistent.invalid (RFC 2606) can never resolve anyway.
        with (
            patch("requests.get") as mock_get,
            pytest.raises(UsageError, match="does not exist"),
        ):
            self.param_type.convert(
                "http://nonexistent.invalid/config.yml", None, None
            )
        mock_get.assert_not_called()


class TestTrainConfigUrl:
    """End-to-end: `axolotl train <https-url>` passes Click parsing."""

    def test_train_accepts_config_url(self, cli_runner):
        # launch_training is mocked; requests.get is patched and asserted
        # uncalled so CI never hits the real URL.
        with (
            patch("axolotl.cli.main.launch_training") as mock_launch,
            patch("requests.get") as mock_get,
        ):
            result = cli_runner.invoke(
                cli, ["train", CONFIG_URL, "--launcher", "python"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        mock_launch.assert_called_once()
        assert mock_launch.call_args[0][0] == CONFIG_URL
        mock_get.assert_not_called()
