"""pytest tests for GGUF export helpers."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from axolotl.utils.gguf import (
    export_gguf,
    preflight,
    resolve_llama_cpp,
    resolve_quantize_bin,
)


@pytest.fixture
def model_dir(tmp_path) -> Path:
    """A minimal HuggingFace checkpoint that passes every preflight check."""
    path = tmp_path / "merged"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "vocab_size": 32})
    )
    (path / "tokenizer.json").write_text(
        json.dumps(
            {"model": {"vocab": {str(i): i for i in range(30)}}, "added_tokens": []}
        )
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"})
    )
    (path / "model.safetensors").write_bytes(b"\0" * 1024)

    return path


@pytest.fixture
def llama_cpp_dir(tmp_path) -> Path:
    """A llama.cpp checkout with a built `llama-quantize`."""
    path = tmp_path / "llama.cpp"
    (path / "build" / "bin").mkdir(parents=True)
    (path / "convert_hf_to_gguf.py").touch()
    (path / "build" / "bin" / "llama-quantize").touch()

    return path


class TestResolveLlamaCpp:
    """Tests for locating a llama.cpp checkout and its binaries."""

    def test_explicit_path(self, llama_cpp_dir):
        assert resolve_llama_cpp(str(llama_cpp_dir)) == llama_cpp_dir

    def test_env_var_fallback(self, llama_cpp_dir, monkeypatch):
        monkeypatch.setenv("LLAMA_CPP_DIR", str(llama_cpp_dir))
        assert resolve_llama_cpp() == llama_cpp_dir

    def test_explicit_path_wins_over_env(self, llama_cpp_dir, monkeypatch):
        monkeypatch.setenv("LLAMA_CPP_DIR", "/nonexistent")
        assert resolve_llama_cpp(llama_cpp_dir) == llama_cpp_dir

    def test_unset(self, monkeypatch):
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        with pytest.raises(ValueError, match="No llama.cpp checkout found"):
            resolve_llama_cpp()

    def test_not_a_checkout(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        with pytest.raises(ValueError, match="is not a llama.cpp checkout"):
            resolve_llama_cpp(tmp_path)

    def test_quantize_bin(self, llama_cpp_dir):
        assert resolve_quantize_bin(llama_cpp_dir).name == "llama-quantize"

    def test_quantize_bin_unbuilt(self, llama_cpp_dir):
        (llama_cpp_dir / "build" / "bin" / "llama-quantize").unlink()
        with patch("axolotl.utils.gguf.shutil.which", return_value=None):
            with pytest.raises(ValueError, match="Build llama.cpp first"):
                resolve_quantize_bin(llama_cpp_dir)


class TestPreflight:
    """Tests for the checks that run before a (slow) conversion."""

    def test_valid_checkpoint(self, model_dir, tmp_path):
        preflight(model_dir, tmp_path)

    def test_missing_config(self, tmp_path):
        with pytest.raises(ValueError, match="not a HuggingFace checkpoint"):
            preflight(tmp_path, tmp_path)

    def test_pre_quantized(self, model_dir, tmp_path):
        _patch_config(model_dir, quantization_config={"quant_method": "torchao"})
        with pytest.raises(ValueError, match="pre-quantized checkpoint"):
            preflight(model_dir, tmp_path)

    def test_mtp_layers(self, model_dir, tmp_path):
        _patch_config(model_dir, num_nextn_predict_layers=1)
        with pytest.raises(ValueError, match="num_nextn_predict_layers"):
            preflight(model_dir, tmp_path)

    def test_mtp_layers_zero_is_fine(self, model_dir, tmp_path):
        _patch_config(model_dir, num_nextn_predict_layers=0)
        preflight(model_dir, tmp_path)

    def test_vocab_overflow(self, model_dir, tmp_path):
        _patch_config(model_dir, vocab_size=16)
        with pytest.raises(ValueError, match="vocab_size=16"):
            preflight(model_dir, tmp_path)

    def test_added_tokens_counted(self, model_dir, tmp_path):
        (model_dir / "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": {}}, "added_tokens": [{"id": 99}]})
        )
        with pytest.raises(ValueError, match="Tokenizer has 100 tokens"):
            preflight(model_dir, tmp_path)

    def test_unparseable_tokenizer_skips_vocab_check(self, model_dir, tmp_path):
        (model_dir / "tokenizer.json").write_text("not json")
        preflight(model_dir, tmp_path)

    def test_missing_chat_template_warns(self, model_dir, tmp_path, caplog):
        (model_dir / "tokenizer_config.json").write_text(json.dumps({}))
        preflight(model_dir, tmp_path)
        assert "No chat template" in caplog.text

    def test_chat_template_file(self, model_dir, tmp_path, caplog):
        (model_dir / "tokenizer_config.json").unlink()
        (model_dir / "chat_template.jinja").write_text("{{ messages }}")
        preflight(model_dir, tmp_path)
        assert "No chat template" not in caplog.text

    def test_insufficient_disk(self, model_dir, tmp_path):
        with patch("axolotl.utils.gguf.shutil.disk_usage") as mock_usage:
            mock_usage.return_value.free = 512
            with pytest.raises(ValueError, match="free in"):
                preflight(model_dir, tmp_path)


class TestExportGGUF:
    """Tests for the conversion / quantization orchestration."""

    def test_convert_only(self, model_dir, llama_cpp_dir, tmp_path):
        with patch("axolotl.utils.gguf._run") as mock_run:
            outputs = export_gguf(
                model_dir, tmp_path / "out", llama_cpp_dir=llama_cpp_dir
            )

        assert outputs == [tmp_path / "out" / "merged-f16.gguf"]
        assert mock_run.call_count == 1
        cmd = mock_run.call_args.args[0]
        assert cmd[1] == str(llama_cpp_dir / "convert_hf_to_gguf.py")
        assert cmd[2:] == [
            str(model_dir),
            "--outfile",
            str(outputs[0]),
            "--outtype",
            "f16",
        ]

    def test_quantize(self, model_dir, llama_cpp_dir, tmp_path):
        with patch("axolotl.utils.gguf._run") as mock_run:
            outputs = export_gguf(
                model_dir,
                tmp_path / "out",
                name="my-run",
                outtype="bf16",
                quantize=["Q4_K_M", "Q8_0"],
                llama_cpp_dir=llama_cpp_dir,
            )

        assert [path.name for path in outputs] == [
            "my-run-bf16.gguf",
            "my-run-Q4_K_M.gguf",
            "my-run-Q8_0.gguf",
        ]
        assert mock_run.call_count == 3
        # Every quant is derived from the unquantized conversion, not from each other.
        for call, quant_type in zip(
            mock_run.call_args_list[1:], ["Q4_K_M", "Q8_0"], strict=True
        ):
            source, target, arg = call.args[0][1:]
            assert (source, arg) == (str(outputs[0]), quant_type)
            assert target.endswith(f"{quant_type}.gguf")

    def test_missing_model_dir(self, llama_cpp_dir, tmp_path):
        with pytest.raises(ValueError, match="Model directory does not exist"):
            export_gguf(
                tmp_path / "nope", tmp_path / "out", llama_cpp_dir=llama_cpp_dir
            )

    def test_unbuilt_quantize_bin_fails_before_converting(
        self, model_dir, llama_cpp_dir, tmp_path
    ):
        (llama_cpp_dir / "build" / "bin" / "llama-quantize").unlink()
        with patch("axolotl.utils.gguf.shutil.which", return_value=None):
            with patch("axolotl.utils.gguf._run") as mock_run:
                with pytest.raises(ValueError, match="llama-quantize"):
                    export_gguf(
                        model_dir,
                        tmp_path / "out",
                        quantize=["Q4_K_M"],
                        llama_cpp_dir=llama_cpp_dir,
                    )

        mock_run.assert_not_called()

    def test_preflight_runs_before_converting(self, model_dir, llama_cpp_dir, tmp_path):
        _patch_config(model_dir, quantization_config={"quant_method": "torchao"})
        with patch("axolotl.utils.gguf._run") as mock_run:
            with pytest.raises(ValueError, match="pre-quantized"):
                export_gguf(model_dir, tmp_path / "out", llama_cpp_dir=llama_cpp_dir)

        mock_run.assert_not_called()


def _patch_config(model_dir: Path, **updates) -> None:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**config, **updates}))
