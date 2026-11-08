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
        path, is_lora = resolve_model_dir(cfg, str(tmp_path / "elsewhere"))
        assert path == tmp_path / "elsewhere"
        assert is_lora is False

    def test_explicit_model_dir_lora_flag(self, tmp_path):
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "lora"})
        path, is_lora = resolve_model_dir(cfg, str(tmp_path / "elsewhere"), lora=True)
        assert path == tmp_path / "elsewhere"
        assert is_lora is True

    def test_explicit_model_dir_auto_detect_adapter(self, tmp_path):
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        cfg = DictDefault({"output_dir": str(tmp_path)})
        path, is_lora = resolve_model_dir(cfg, str(adapter_dir))
        assert path == adapter_dir
        assert is_lora is True

    def test_prefers_merged_dir(self, tmp_path):
        (tmp_path / "merged").mkdir()
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "lora"})
        path, is_lora = resolve_model_dir(cfg)
        assert path == tmp_path / "merged"
        assert is_lora is False

    def test_merged_dir_with_lora_flag(self, tmp_path):
        (tmp_path / "merged").mkdir()
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "lora"})
        path, is_lora = resolve_model_dir(cfg, lora=True)
        assert path == tmp_path
        assert is_lora is True

    def test_full_finetune_output_dir(self, tmp_path):
        cfg = DictDefault({"output_dir": str(tmp_path)})
        path, is_lora = resolve_model_dir(cfg)
        assert path == tmp_path
        assert is_lora is False

    def test_unmerged_adapter_auto_detected(self, tmp_path):
        """Adapter-only output_dir is auto-detected as a LoRA export."""
        (tmp_path / "adapter_config.json").write_text("{}")
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "lora"})
        path, is_lora = resolve_model_dir(cfg)
        assert path == tmp_path
        assert is_lora is True

    def test_unmerged_adapter_explicit_no_lora(self, tmp_path):
        """User can explicitly refuse LoRA export even when adapter is present."""
        (tmp_path / "adapter_config.json").write_text("{}")
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "qlora"})
        with pytest.raises(ValueError, match="axolotl merge-lora"):
            resolve_model_dir(cfg, lora=False)

    def test_qlora_adapter_auto_detected(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "qlora"})
        path, is_lora = resolve_model_dir(cfg)
        assert path == tmp_path
        assert is_lora is True


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
                    with patch(
                        "axolotl.cli.export.export_lora_gguf", return_value=tmp_path / "lora.gguf"
                    ) as mock_lora_export:
                        do_export("config.yml", cli_args)

            return mock_export.call_args, mock_lora_export.call_args

        return _run

    def test_defaults(self, run_export, tmp_path):
        args, _ = run_export()

        assert args.args == (
            tmp_path / "run",
            str(tmp_path / "run" / "gguf" / "run-{ftype}.gguf"),
        )
        assert args.kwargs == {
            "outtype": "f16",
            "quantize": [],
            "llama_cpp_dir": None,
        }

    def test_config_block(self, run_export, tmp_path):
        args, _ = run_export(
            {
                "outtype": "bf16",
                "quantize": ["q4_k_m"],
                "outfile": str(tmp_path / "ggufs" / "{ftype}.gguf"),
                "llama_cpp_dir": "/opt/llama.cpp",
            }
        )

        assert args.args[1] == str(tmp_path / "ggufs" / "{ftype}.gguf")
        assert args.kwargs["outtype"] == "bf16"
        assert args.kwargs["quantize"] == ["Q4_K_M"]
        assert args.kwargs["llama_cpp_dir"] == "/opt/llama.cpp"

    def test_cli_args_override_config_block(self, run_export, tmp_path):
        args, _ = run_export(
            {"outtype": "bf16", "quantize": ["q8_0"]},
            quantize="Q4_K_M",
            model_dir=str(tmp_path),
        )

        assert args.args[0] == tmp_path
        assert args.kwargs["quantize"] == ["Q4_K_M"]
        assert args.kwargs["outtype"] == "bf16"

    def test_invalid_quant_type(self, run_export):
        with pytest.raises(ValueError, match="Unknown GGUF quant type"):
            run_export(quantize="Q9_K")

    def test_lora_export_dispatches_to_lora_exporter(self, run_export, tmp_path):
        """When lora=True and adapter is detected, export_lora_gguf is called."""
        # Create an adapter_config.json so the output dir is detected as an adapter
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "adapter_config.json").write_text("{}")

        _, lora_args = run_export(export={"lora": True})

        assert lora_args is not None
        assert lora_args.args[0] == run_dir
        assert lora_args.args[1] == str(run_dir / "gguf" / "run-f16.gguf")
        assert lora_args.kwargs["outtype"] == "f16"

    def test_lora_export_with_custom_outtype(self, run_export, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "adapter_config.json").write_text("{}")

        _, lora_args = run_export(export={"lora": True, "outtype": "f32"})

        assert lora_args is not None
        assert lora_args.args[1] == str(run_dir / "gguf" / "run-f32.gguf")
        assert lora_args.kwargs["outtype"] == "f32"

    def test_lora_export_rejects_quantize(self, run_export):
        with pytest.raises(ValueError, match="quantize.*not supported for LoRA"):
            run_export(export={"lora": True, "quantize": ["Q4_K_M"]})

    def test_lora_export_rejects_invalid_outtype(self, run_export):
        with pytest.raises(ValueError, match="outtype.*must be one of"):
            run_export(export={"lora": True, "outtype": "tq1_0"})