"""pytest tests for axolotl CLI inference command."""

from unittest.mock import patch

from axolotl.cli.main import cli


def test_inference_basic(cli_runner, config_path):
    """Test basic inference"""
    with patch("axolotl.cli.inference.do_inference") as mock:
        result = cli_runner.invoke(
            cli,
            ["inference", str(config_path), "--no-accelerate"],
            catch_exceptions=False,
        )

        assert mock.called
        assert result.exit_code == 0


def test_inference_gradio(cli_runner, config_path):
    """Test basic inference (gradio path)"""
    with patch("axolotl.cli.inference.do_inference_gradio") as mock:
        result = cli_runner.invoke(
            cli,
            ["inference", str(config_path), "--no-accelerate", "--gradio"],
            catch_exceptions=False,
        )

        assert mock.called
        assert result.exit_code == 0
