"""pytest tests for axolotl CLI export command."""

from pathlib import Path
from unittest.mock import patch

import pytest

from axolotl.cli.export import do_export, resolve_model_dir
from axolotl.cli.main import cli
from axolotl.utils.dict import DictDefault


@pytest.fixture
def adapter_output_dir(tmp_path) -> Path:
    """An `output_dir` holding an unmerged PEFT adapter."""
    (tmp_path / "adapter_config.json").write_text("{}")

    return tmp_path


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


@pytest.mark.parametrize("flag, expected", [("--lora", True), ("--no-lora", False)])
def test_export_lora_flag(cli_runner, export_config_path, flag, expected):
    """Test the --lora/--no-lora override"""
    with patch("axolotl.cli.export.do_export") as mock_do_export:
        result = cli_runner.invoke(cli, ["export", str(export_config_path), flag])
        assert result.exit_code == 0

        assert mock_do_export.call_args.args[1] == {"lora": expected}


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

    def test_adapter_dir_is_not_exported_without_the_flag(self, adapter_output_dir):
        cfg = DictDefault({"output_dir": str(adapter_output_dir), "adapter": "lora"})
        with pytest.raises(ValueError, match="pass --lora"):
            resolve_model_dir(cfg)

    def test_merged_wins_over_adapter(self, adapter_output_dir):
        (adapter_output_dir / "merged").mkdir()
        cfg = DictDefault({"output_dir": str(adapter_output_dir), "adapter": "lora"})
        assert resolve_model_dir(cfg) == adapter_output_dir / "merged"

    def test_explicit_lora_wins_over_merged(self, adapter_output_dir):
        (adapter_output_dir / "merged").mkdir()
        cfg = DictDefault({"output_dir": str(adapter_output_dir), "adapter": "lora"})
        assert resolve_model_dir(cfg, lora=True) == adapter_output_dir

    def test_explicit_lora_without_an_adapter(self, tmp_path):
        cfg = DictDefault({"output_dir": str(tmp_path), "adapter": "lora"})
        with pytest.raises(ValueError, match="holds no adapter_config.json"):
            resolve_model_dir(cfg, lora=True)

    def test_explicit_no_lora_still_refuses_the_adapter(self, adapter_output_dir):
        cfg = DictDefault({"output_dir": str(adapter_output_dir), "adapter": "qlora"})
        with pytest.raises(ValueError, match="axolotl merge-lora"):
            resolve_model_dir(cfg, lora=False)


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

        assert args == (
            tmp_path / "run",
            str(tmp_path / "run" / "gguf" / "run-{ftype}.gguf"),
        )
        assert kwargs == {
            "outtype": "f16",
            "quantize": [],
            "llama_cpp_dir": None,
        }

    def test_config_block(self, run_export, tmp_path):
        args, kwargs = run_export(
            {
                "outtype": "bf16",
                "quantize": ["q4_k_m"],
                "outfile": str(tmp_path / "ggufs" / "{ftype}.gguf"),
                "llama_cpp_dir": "/opt/llama.cpp",
            }
        )

        assert args[1] == str(tmp_path / "ggufs" / "{ftype}.gguf")
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


class TestDoExportLora:
    """Tests for the adapter branch of `do_export`."""

    @pytest.fixture
    def run_lora_export(self, tmp_path):
        """Runs `do_export` against an adapter dir, returning the LoRA exporter's call."""

        def _run(export: dict | None = None, **cli_args):
            run_dir = tmp_path / "run"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "adapter_config.json").write_text("{}")
            cfg = DictDefault(
                {
                    "output_dir": str(run_dir),
                    "adapter": "lora",
                    "base_model": "org/base",
                    "export": {**(export or {}), "lora": True},
                }
            )
            with patch("axolotl.cli.export.load_cfg", return_value=cfg):
                with patch(
                    "axolotl.cli.export.export_lora_gguf", return_value=[]
                ) as mock_export:
                    do_export("config.yml", cli_args)

            return mock_export.call_args

        return _run

    def test_lora_export(self, run_lora_export, tmp_path):
        args, kwargs = run_lora_export()

        # The `-lora` marker keeps an adapter export off a full model's filename.
        assert args == (
            tmp_path / "run",
            str(tmp_path / "run" / "gguf" / "run-lora-{ftype}.gguf"),
        )
        assert kwargs == {
            "outtype": "f32",
            "llama_cpp_dir": None,
        }

    def test_config_outtype_wins_over_the_lora_default(self, run_lora_export):
        assert run_lora_export({"outtype": "bf16"}).kwargs["outtype"] == "bf16"

    def test_cli_outtype_wins_over_the_lora_default(self, run_lora_export):
        assert run_lora_export(outtype="f16").kwargs["outtype"] == "f16"

    def test_quantize_rejected_for_a_lora_export(self, run_lora_export):
        with pytest.raises(ValueError, match="only takes full models"):
            run_lora_export({"quantize": ["Q4_K_M"]})

    def test_explicit_no_lora_takes_the_full_model_path(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "adapter_config.json").write_text("{}")
        cfg = DictDefault(
            {"output_dir": str(run_dir), "adapter": "lora", "export": {"lora": False}}
        )
        with patch("axolotl.cli.export.load_cfg", return_value=cfg):
            with pytest.raises(ValueError, match="axolotl merge-lora"):
                do_export("config.yml", {})
