"""pytest tests for GGUF export helpers."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from axolotl.utils.gguf import (
    export_gguf,
    export_lora_gguf,
    lora_preflight,
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
def adapter_dir(tmp_path) -> Path:
    """A minimal PEFT LoRA adapter that passes every preflight check."""
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "org/base"})
    )
    (path / "adapter_model.safetensors").touch()

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

    def test_checks_for_the_script_it_will_run(self, llama_cpp_dir):
        with pytest.raises(ValueError, match="convert_lora_to_gguf.py missing"):
            resolve_llama_cpp(llama_cpp_dir, "convert_lora_to_gguf.py")


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
                model_dir,
                str(tmp_path / "out" / "model.gguf"),
                llama_cpp_dir=llama_cpp_dir,
            )

        assert outputs == [tmp_path / "out" / "model.gguf"]
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
                str(tmp_path / "out" / "my-run-{ftype}.gguf"),
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
                tmp_path / "nope",
                str(tmp_path / "out.gguf"),
                llama_cpp_dir=llama_cpp_dir,
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
                        str(tmp_path / "{ftype}.gguf"),
                        quantize=["Q4_K_M"],
                        llama_cpp_dir=llama_cpp_dir,
                    )

        mock_run.assert_not_called()

    def test_preflight_runs_before_converting(self, model_dir, llama_cpp_dir, tmp_path):
        _patch_config(model_dir, quantization_config={"quant_method": "torchao"})
        with patch("axolotl.utils.gguf._run") as mock_run:
            with pytest.raises(ValueError, match="pre-quantized"):
                export_gguf(
                    model_dir, str(tmp_path / "out.gguf"), llama_cpp_dir=llama_cpp_dir
                )

        mock_run.assert_not_called()


class TestLoraPreflight:
    """Tests for the checks that run before converting an adapter."""

    def test_valid_adapter(self, adapter_dir):
        lora_preflight(adapter_dir)

    def test_missing_adapter_config(self, tmp_path):
        with pytest.raises(ValueError, match="not a PEFT adapter"):
            lora_preflight(tmp_path)

    def test_dora(self, adapter_dir):
        _patch_adapter_config(adapter_dir, use_dora=True)
        with pytest.raises(ValueError, match="DoRA adapter"):
            lora_preflight(adapter_dir)

    def test_dora_disabled_is_fine(self, adapter_dir):
        _patch_adapter_config(adapter_dir, use_dora=False)
        lora_preflight(adapter_dir)

    def test_modules_to_save(self, adapter_dir):
        _patch_adapter_config(adapter_dir, modules_to_save=["lm_head"])
        with pytest.raises(ValueError, match=r"modules_to_save=\['lm_head'\]"):
            lora_preflight(adapter_dir)

    def test_modules_to_save_null_is_fine(self, adapter_dir):
        _patch_adapter_config(adapter_dir, modules_to_save=None)
        lora_preflight(adapter_dir)


class TestExportLoraGGUF:
    """Tests for converting an adapter to a standalone GGUF LoRA."""

    @pytest.fixture
    def llama_cpp_dir(self, llama_cpp_dir) -> Path:
        """The same checkout, with the LoRA converter alongside the model one."""
        (llama_cpp_dir / "convert_lora_to_gguf.py").touch()

        return llama_cpp_dir

    def test_convert(self, adapter_dir, llama_cpp_dir, tmp_path):
        with patch("axolotl.utils.gguf._run") as mock_run:
            outputs = export_lora_gguf(
                adapter_dir,
                str(tmp_path / "out" / "run-lora-{ftype}.gguf"),
                llama_cpp_dir=llama_cpp_dir,
            )

        assert outputs == [tmp_path / "out" / "run-lora-f32.gguf"]
        assert mock_run.call_count == 1
        cmd = mock_run.call_args.args[0]
        assert cmd[1] == str(llama_cpp_dir / "convert_lora_to_gguf.py")
        assert cmd[2:] == [
            str(adapter_dir),
            "--outfile",
            str(outputs[0]),
            "--outtype",
            "f32",
        ]

    def test_base_config_beside_the_adapter_is_passed(
        self, adapter_dir, llama_cpp_dir, tmp_path
    ):
        # Training pre-saves the base model's config into the adapter dir.
        (adapter_dir / "config.json").write_text("{}")
        with patch("axolotl.utils.gguf._run") as mock_run:
            export_lora_gguf(
                adapter_dir,
                str(tmp_path / "out.gguf"),
                llama_cpp_dir=llama_cpp_dir,
            )

        assert mock_run.call_args.args[0][-2:] == ["--base", str(adapter_dir)]

    def test_adapter_without_a_base_config_is_left_to_the_converter(
        self, adapter_dir, llama_cpp_dir, tmp_path
    ):
        with patch("axolotl.utils.gguf._run") as mock_run:
            export_lora_gguf(
                adapter_dir,
                str(tmp_path / "out.gguf"),
                llama_cpp_dir=llama_cpp_dir,
            )

        assert "--base" not in mock_run.call_args.args[0]

    def test_missing_adapter_dir(self, llama_cpp_dir, tmp_path):
        with pytest.raises(ValueError, match="Adapter directory does not exist"):
            export_lora_gguf(
                tmp_path / "nope",
                str(tmp_path / "out.gguf"),
                llama_cpp_dir=llama_cpp_dir,
            )

    def test_checkout_without_the_lora_converter(self, adapter_dir, tmp_path):
        bare = tmp_path / "bare-llama.cpp"
        bare.mkdir()
        (bare / "convert_hf_to_gguf.py").touch()
        with pytest.raises(ValueError, match="convert_lora_to_gguf.py missing"):
            export_lora_gguf(
                adapter_dir, str(tmp_path / "out.gguf"), llama_cpp_dir=bare
            )

    def test_preflight_runs_before_converting(
        self, adapter_dir, llama_cpp_dir, tmp_path
    ):
        _patch_adapter_config(adapter_dir, use_dora=True)
        with patch("axolotl.utils.gguf._run") as mock_run:
            with pytest.raises(ValueError, match="DoRA"):
                export_lora_gguf(
                    adapter_dir,
                    str(tmp_path / "out.gguf"),
                    llama_cpp_dir=llama_cpp_dir,
                )

        mock_run.assert_not_called()


def _patch_config(model_dir: Path, **updates) -> None:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**config, **updates}))


def _patch_adapter_config(adapter_dir: Path, **updates) -> None:
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**config, **updates}))
