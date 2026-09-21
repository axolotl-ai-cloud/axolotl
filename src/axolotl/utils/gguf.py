"""Export a HuggingFace checkpoint to GGUF via a local llama.cpp checkout."""

import json
import os
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import Sequence

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

CONVERT_SCRIPT = "convert_hf_to_gguf.py"
LORA_CONVERT_SCRIPT = "convert_lora_to_gguf.py"
QUANTIZE_BIN_PATHS = ("build/bin/llama-quantize", "llama-quantize")
WEIGHT_SUFFIXES = (".safetensors", ".bin")
ADAPTER_CONFIG = "adapter_config.json"


def resolve_llama_cpp(
    llama_cpp_dir: str | Path | None = None, script: str = CONVERT_SCRIPT
) -> Path:
    """Locate a llama.cpp checkout from config, then `$LLAMA_CPP_DIR`."""
    root = llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    if not root:
        raise ValueError(
            "No llama.cpp checkout found. Clone https://github.com/ggml-org/llama.cpp, "
            "build it, then set `export.llama_cpp_dir` or $LLAMA_CPP_DIR."
        )

    root = Path(root).expanduser()
    if not (root / script).is_file():
        raise ValueError(f"{root} is not a llama.cpp checkout ({script} missing).")

    return root


def resolve_quantize_bin(llama_cpp_dir: Path) -> Path:
    """Locate the compiled `llama-quantize` binary."""
    for rel_path in QUANTIZE_BIN_PATHS:
        if (bin_path := llama_cpp_dir / rel_path).is_file():
            return bin_path
    if which := shutil.which("llama-quantize"):
        return Path(which)

    raise ValueError(
        f"`llama-quantize` not found in {llama_cpp_dir}. Build llama.cpp first: "
        f"`cmake -S {llama_cpp_dir} -B {llama_cpp_dir}/build && "
        f"cmake --build {llama_cpp_dir}/build --config Release -j`."
    )


def _tokenizer_vocab_size(model_dir: Path) -> int | None:
    """
    Highest token id in `tokenizer.json`, including added tokens, as a count. Returns
    None for any tokenizer we cannot parse, which skips the vocab check.
    """
    try:
        tokenizer = json.loads(
            (model_dir / "tokenizer.json").read_text(encoding="utf-8")
        )
        added_ids = [token["id"] for token in tokenizer.get("added_tokens", [])]
        return max(len(tokenizer["model"]["vocab"]), max(added_ids, default=-1) + 1)
    except Exception:
        return None


def _has_chat_template(model_dir: Path) -> bool:
    if (model_dir / "chat_template.jinja").is_file():
        return True
    try:
        tokenizer_config = json.loads(
            (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
        )
    except OSError:
        return False

    return bool(tokenizer_config.get("chat_template"))


def _weights_size(model_dir: Path) -> int:
    return sum(
        path.stat().st_size
        for path in model_dir.glob("*")
        if path.suffix in WEIGHT_SUFFIXES
    )


def preflight(model_dir: Path, output_dir: Path) -> None:
    """Fail fast on checkpoints llama.cpp cannot convert, or would convert incorrectly."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(
            f"No config.json in {model_dir} — not a HuggingFace checkpoint."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("quantization_config"):
        raise ValueError(
            "Cannot convert a pre-quantized checkpoint to GGUF. Re-export the model in "
            "bf16 first (`axolotl merge-lora --dequant`)."
        )

    # llama.cpp has no tensors for GLM4-MoE-style multi-token-prediction layers and
    # fails at load time with `missing tensor 'blk.N.attn_norm.weight'`.
    if config.get("num_nextn_predict_layers"):
        raise ValueError(
            f"{config_path} sets num_nextn_predict_layers="
            f"{config['num_nextn_predict_layers']}, which llama.cpp cannot load. "
            "Set it to 0 before converting."
        )

    vocab_size, tokenizer_size = (
        config.get("vocab_size"),
        _tokenizer_vocab_size(model_dir),
    )
    if vocab_size and tokenizer_size and tokenizer_size > vocab_size:
        raise ValueError(
            f"Tokenizer has {tokenizer_size} tokens but config.json declares "
            f"vocab_size={vocab_size}. Resize the embeddings before exporting."
        )

    if not _has_chat_template(model_dir):
        LOG.warning(
            "No chat template found in %s. llama.cpp runtimes will fall back to a "
            "default template, which usually degrades a fine-tuned model.",
            model_dir,
        )

    # Conversion writes a full-size GGUF, and quantization reads it while writing another.
    needed = 2 * _weights_size(model_dir)
    if (free := shutil.disk_usage(output_dir).free) < needed:
        raise ValueError(
            f"Need ~{needed / 1e9:.1f}GB free in {output_dir} to export, but only "
            f"{free / 1e9:.1f}GB available."
        )


def lora_preflight(adapter_dir: Path) -> None:
    """Fail fast on adapters llama.cpp's LoRA converter rejects."""
    config_path = adapter_dir / ADAPTER_CONFIG
    if not config_path.is_file():
        raise ValueError(f"No {ADAPTER_CONFIG} in {adapter_dir}, not a PEFT adapter.")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    merge_instead = "Run `axolotl merge-lora` and export the merged model instead."

    # The converter only pairs up lora_A/lora_B tensors; anything saved whole aborts it.
    if config.get("use_dora"):
        raise ValueError(f"llama.cpp cannot convert a DoRA adapter. {merge_instead}")

    if modules := config.get("modules_to_save"):
        raise ValueError(
            f"llama.cpp cannot convert an adapter with "
            f"modules_to_save={sorted(modules)}. {merge_instead}"
        )


def _run(cmd: list[str], output: Path) -> None:
    LOG.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)  # nosec B603
    except BaseException:
        # A half-written .gguf is indistinguishable from a good one on disk.
        output.unlink(missing_ok=True)
        raise


def export_gguf(
    model_dir: str | Path,
    outfile: str,
    *,
    outtype: str = "f16",
    quantize: Sequence[str] = (),
    llama_cpp_dir: str | Path | None = None,
) -> list[Path]:
    """
    Convert a HuggingFace checkpoint to GGUF, plus one file per requested quant type.

    `{ftype}` in `outfile` is replaced by each weight type, as in llama.cpp.

    Returns:
        Paths of the written GGUF files, unquantized conversion first.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ValueError(f"Model directory does not exist: {model_dir}")

    llama_cpp = resolve_llama_cpp(llama_cpp_dir)
    quantize_bin = resolve_quantize_bin(llama_cpp) if quantize else None

    converted = Path(outfile.replace("{ftype}", outtype))
    converted.parent.mkdir(parents=True, exist_ok=True)
    preflight(model_dir, converted.parent)

    LOG.info("Converting %s to GGUF (%s)...", model_dir, outtype)
    _run(
        [
            sys.executable,
            str(llama_cpp / CONVERT_SCRIPT),
            str(model_dir),
            "--outfile",
            str(converted),
            "--outtype",
            outtype,
        ],
        converted,
    )

    outputs = [converted]
    for quant_type in quantize:
        quantized = Path(outfile.replace("{ftype}", quant_type))
        LOG.info("Quantizing to %s...", quant_type)
        _run([str(quantize_bin), str(converted), str(quantized), quant_type], quantized)
        outputs.append(quantized)

    return outputs


def export_lora_gguf(
    adapter_dir: str | Path,
    outfile: str,
    *,
    outtype: str = "f32",
    llama_cpp_dir: str | Path | None = None,
) -> list[Path]:
    """
    Convert a PEFT LoRA adapter to a standalone GGUF, loaded at runtime on top of a
    GGUF of the same base model (`llama-cli -m base.gguf --lora adapter.gguf`).

    Returns:
        Path of the written GGUF, as a single-element list.
    """
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise ValueError(f"Adapter directory does not exist: {adapter_dir}")

    llama_cpp = resolve_llama_cpp(llama_cpp_dir, LORA_CONVERT_SCRIPT)

    converted = Path(outfile.replace("{ftype}", outtype))
    converted.parent.mkdir(parents=True, exist_ok=True)
    lora_preflight(adapter_dir)

    cmd = [
        sys.executable,
        str(llama_cpp / LORA_CONVERT_SCRIPT),
        str(adapter_dir),
        "--outfile",
        str(converted),
        "--outtype",
        outtype,
    ]
    # Without `--base` the converter fetches the base config from the Hub.
    if (adapter_dir / "config.json").is_file():
        cmd += ["--base", str(adapter_dir)]

    LOG.info("Converting %s to a GGUF LoRA (%s)...", adapter_dir, outtype)
    _run(cmd, converted)

    return [converted]
