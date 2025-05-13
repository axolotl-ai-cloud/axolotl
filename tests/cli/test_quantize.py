"""Tests for quantize CLI command."""

from unittest.mock import MagicMock, patch

from axolotl.cli.main import cli

from .test_cli_base import BaseCliTest

class TestQuantizeCommand(BaseCliTest):
    """Test cases for quantize command."""

    cli = cli

    def test_quantize_cli_validation(self, cli_runner):
        """Test CLI validation"""
        self._test_cli_validation(cli_runner, "quantize")

    def test_quantize_basic_execution(self, cli_runner, tmp_path, valid_test_config):
        """Test basic successful execution"""
        self._test_basic_execution(cli_runner, tmp_path, valid_test_config, "quantize")

    def test_quantize_cli_overrides(self, cli_runner, tmp_path, valid_test_config):
        """Test CLI arguments properly override config values"""
        self._test_cli_overrides(cli_runner, tmp_path, valid_test_config, "quantize")
