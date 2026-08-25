"""pytest tests for axolotl CLI export command."""

from unittest.mock import patch

import pytest

from axolotl.cli.export import do_export, resolve_model_dir
from axolotl.cli.main import cli
from axolotl.utils.dict import DictDefault


@pytest.fixture
def export_config_path(tmp_path, valid_test_config):
    """A config whose `output_dir` holds a full (already merged) model."""
    path = tmp_path / "config.yml"
    path.write_text(f"{valid_test_config}\noutput_dir: {tmp_path / 'run'}\n")

    return path


def test_export_basic(cli_runner, export_config_path):
    """Test basic export command"""
    with patch("axolotl.cli.export.do_export") as mock_do_export:
        result = cli_runner.invoke(cli, ["export", str(export_config_path)])
        assert result.exit_code == 0

        mock_do_export.assert_called_once()
        assert mock_do_export.call_args.args[0] == str(export_config_path)


def test_export_cli_args(cli_runner, export_config_path, tmp_path):
    """Test export with CLI overrides"""
    with patch("axolotl.cli.export.do_export") as mock_do_export:
        result = cli_runner.invoke(
            cli,
            [
                "export",
                str(export_config_path),
                "--model-dir",
                str(tmp_path),
                "--quantize",
                "Q4_K_M,Q8_0",
                "--outtype",
                "bf16",
            ],
        )
        assert result.exit_code == 0

        cli_args = mock_do_export.call_args.args[1]
        assert cli_args == {
            "model_dir": str(tmp_path),
            "quantize": "Q4_K_M,Q8_0",
            "outtype": "bf16",
        }


def test_export_nonexistent_config(cli_runner, tmp_path):
    """Test export with nonexistent config"""
    result = cli_runner.invoke(cli, ["export", str(tmp_path / "nonexistent.yml")])
    assert result.exit_code != 0


class TestResolveModelDir:
    """Tests for picking which checkpoint gets exported."""

    def test_explicit_model_dir_wins(self, tmp_path):
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "lora"})
        assert resolve_model_dir(cfg, str(tmp_path / "elsewhere")) == (
            tmp_path / "elsewhere"
        )

    def test_prefers_merged_dir(self, tmp_path):
        (tmp_path / "merged").mkdir()
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "lora"})
        assert resolve_model_dir(cfg) == tmp_path / "merged"

    def test_full_finetune_output_dir(self, tmp_path):
        cfg = DictDefault({"output_dir": str(tmp_path)})
        assert resolve_model_dir(cfg) == tmp_path

    def test_unmerged_adapter(self, tmp_path):
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "qlora"})
        with pytest.raises(ValueError, match="axolotl merge-lora"):
            resolve_model_dir(cfg)


class TestDoExport:
    """Tests for wiring config and CLI args through to the exporter."""

    @pytest.fixture
    def run_export(self, tmp_path):
        """Runs `do_export` against a stubbed config, returning the exporter's call."""

        def _run(export: dict | None = None, **cli_args):
            cfg = DictDefault({"output_dir": str(tmp_path / "run"), "export": export})
            with patch("axolotl.cli.export.load_cfg", return_value=cfg):
                with patch(
                    "axolotl.cli.export.export_gguf", return_value=[]
                ) as mock_export:
                    do_export("config.yml", cli_args)

            return mock_export.call_args

        return _run

    def test_defaults(self, run_export, tmp_path):
        args, kwargs = run_export()

        assert args == (tmp_path / "run", tmp_path / "run" / "gguf")
        assert kwargs == {
            "name": "run",
            "outtype": "f16",
            "quantize": [],
            "llama_cpp_dir": None,
        }

    def test_config_block(self, run_export, tmp_path):
        args, kwargs = run_export(
            {
                "outtype": "bf16",
                "quantize": ["q4_k_m"],
                "output_dir": str(tmp_path / "ggufs"),
                "llama_cpp_dir": "/opt/llama.cpp",
            }
        )

        assert args[1] == tmp_path / "ggufs"
        assert kwargs["outtype"] == "bf16"
        assert kwargs["quantize"] == ["Q4_K_M"]
        assert kwargs["llama_cpp_dir"] == "/opt/llama.cpp"

    def test_cli_args_override_config_block(self, run_export, tmp_path):
        args, kwargs = run_export(
            {"outtype": "bf16", "quantize": ["q8_0"]},
            quantize="Q4_K_M",
            model_dir=str(tmp_path),
        )

        assert args[0] == tmp_path
        assert kwargs["quantize"] == ["Q4_K_M"]
        assert kwargs["outtype"] == "bf16"

    def test_invalid_quant_type(self, run_export):
        with pytest.raises(ValueError, match="Unknown GGUF quant type"):
            run_export(quantize="Q9_K")
