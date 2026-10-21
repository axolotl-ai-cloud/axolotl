"""Carry a base checkpoint's MTP (multi-token prediction) weights into a saved model.

transformers' causal-LM classes don't load the ``mtp.*`` / extra ``num_nextn_predict_layers``
decoder layers, so ``save_pretrained`` drops them and vLLM/SGLang lose speculative decoding.
"""

import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MTP_SHARD = "model-mtp.safetensors"


def _text_config(config: dict) -> dict:
    return config.get("text_config") or config


def _is_mtp_key(key: str, num_layers: int | None) -> bool:
    if re.search(r"(^|\.)mtp[._]", key):
        return True
    match = re.match(r"model\.(language_model\.)?layers\.(\d+)\.", key)
    return bool(match and num_layers is not None and int(match.group(2)) >= num_layers)


def _file_keys(path: str | Path) -> dict[str, str]:
    with safe_open(path, framework="pt") as f:
        return dict.fromkeys(f.keys(), Path(path).name)


def _weight_map(model_dir: Path) -> dict[str, str]:
    if (model_dir / SAFE_WEIGHTS_INDEX_NAME).exists():
        return json.loads((model_dir / SAFE_WEIGHTS_INDEX_NAME).read_text())[
            "weight_map"
        ]
    weight_map: dict[str, str] = {}
    for name in (SAFE_WEIGHTS_NAME, MTP_SHARD):
        if (model_dir / name).exists():
            weight_map.update(_file_keys(model_dir / name))
    return weight_map


def _base_files(base_model: str, revision: str | None):
    def fetch(name: str) -> str:
        return cached_file(base_model, name, revision=revision)

    config = json.loads(Path(fetch("config.json")).read_text())
    index = cached_file(
        base_model,
        SAFE_WEIGHTS_INDEX_NAME,
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )
    if index:
        weight_map = json.loads(Path(index).read_text())["weight_map"]
    else:
        weight_map = _file_keys(fetch(SAFE_WEIGHTS_NAME))
    return weight_map, config, fetch


def maybe_copy_mtp_weights(cfg: DictDefault, output_dir: str | Path) -> int:
    """Carry MTP weights into a full fine-tune's output; adapters and compressed outputs are skipped."""
    if cfg.adapter or cfg.llmcompressor:
        return 0
    return copy_mtp_weights(cfg.base_model, output_dir, revision=cfg.revision_of_model)


def copy_mtp_weights(
    base_model: str, output_dir: str | Path, revision: str | None = None
) -> int:
    """Copy MTP tensors missing from ``output_dir`` out of ``base_model``. Returns the count."""
    try:
        return _copy_mtp_weights(base_model, Path(output_dir), revision)
    except Exception as exc:
        LOG.warning(
            "Skipping MTP carry-over from %s into %s: %s", base_model, output_dir, exc
        )
        return 0


def _copy_mtp_weights(base_model: str, output_dir: Path, revision: str | None) -> int:
    out_map = _weight_map(output_dir)
    if not out_map:
        return 0
    base_map, config, fetch = _base_files(base_model, revision)

    text_config = _text_config(config)
    num_layers = (
        text_config.get("num_hidden_layers")
        if text_config.get("num_nextn_predict_layers")
        else None
    )
    missing = {
        k: f
        for k, f in base_map.items()
        if k not in out_map and _is_mtp_key(k, num_layers)
    }
    if not missing:
        return 0

    out_config = json.loads((output_dir / "config.json").read_text())
    if config.get("quantization_config") and not out_config.get("quantization_config"):
        LOG.warning(
            "Base MTP weights are quantized but %s is not; not copying %d MTP tensors.",
            output_dir,
            len(missing),
        )
        return 0

    tensors = {}
    for name in sorted(set(missing.values())):
        with safe_open(fetch(name), framework="pt") as handle:
            tensors.update(
                {k: handle.get_tensor(k) for k, f in missing.items() if f == name}
            )
    # write-then-rename: without an index, loaders pick up every *.safetensors, even a partial one
    tmp = output_dir / f"{MTP_SHARD}.tmp"
    save_file(tensors, tmp, metadata={"format": "pt"})
    tmp.replace(output_dir / MTP_SHARD)

    # Single-file outputs get no index, so a later save_pretrained can't leave one dangling
    index_path = output_dir / SAFE_WEIGHTS_INDEX_NAME
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index["weight_map"].update(dict.fromkeys(tensors, MTP_SHARD))
        index["metadata"]["total_size"] = int(
            index["metadata"].get("total_size", 0)
        ) + sum(t.numel() * t.element_size() for t in tensors.values())
        index_path.write_text(json.dumps(index, indent=2))

    LOG.info(
        "Copied %d MTP tensors from %s into %s", len(tensors), base_model, output_dir
    )
    return len(tensors)
