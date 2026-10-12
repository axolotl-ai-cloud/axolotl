"""Carry a base checkpoint's MTP (multi-token prediction) weights into a saved model.

transformers' causal-LM classes don't load the ``mtp.*`` / extra ``num_nextn_predict_layers``
decoder layers, so ``save_pretrained`` drops them and vLLM/SGLang lose speculative decoding.
"""

import json
import re
from pathlib import Path

from huggingface_hub import get_safetensors_metadata, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"
MTP_SHARD = "model-mtp.safetensors"


def _text_config(config: dict) -> dict:
    return config.get("text_config") or config


def _is_mtp_key(key: str, num_layers: int | None) -> bool:
    if re.search(r"(^|\.)mtp\.", key):
        return True
    match = re.match(r"model\.layers\.(\d+)\.", key)
    return bool(match and num_layers is not None and int(match.group(1)) >= num_layers)


def _weight_map(model_dir: Path) -> dict[str, str]:
    if (model_dir / INDEX_NAME).exists():
        return json.loads((model_dir / INDEX_NAME).read_text())["weight_map"]
    if (model_dir / SINGLE_NAME).exists():
        with safe_open(model_dir / SINGLE_NAME, framework="pt") as f:
            return dict.fromkeys(f.keys(), SINGLE_NAME)
    return {}


def _base_files(base_model: str):
    local = Path(base_model)
    if local.is_dir():
        config = json.loads((local / "config.json").read_text())
        return _weight_map(local), config, lambda name: local / name
    config = json.loads(Path(hf_hub_download(base_model, "config.json")).read_text())
    weight_map = get_safetensors_metadata(base_model).weight_map
    return weight_map, config, lambda name: hf_hub_download(base_model, name)


def copy_mtp_weights(base_model: str, output_dir: str | Path) -> int:
    """Copy MTP tensors missing from ``output_dir`` out of ``base_model``. Returns the count."""
    output_dir = Path(output_dir)
    out_map = _weight_map(output_dir)
    if not out_map:
        return 0
    try:
        base_map, config, fetch = _base_files(base_model)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("Skipping MTP carry-over, can't read %s: %s", base_model, exc)
        return 0

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
    save_file(tensors, output_dir / MTP_SHARD, metadata={"format": "pt"})

    # transformers prefers a lone model.safetensors over an index, so a single file must be sharded
    if (output_dir / INDEX_NAME).exists():
        index = json.loads((output_dir / INDEX_NAME).read_text())
    else:
        trunk = "model-00001-of-00001.safetensors"
        (output_dir / SINGLE_NAME).rename(output_dir / trunk)
        index = {
            "metadata": {"total_size": (output_dir / trunk).stat().st_size},
            "weight_map": dict.fromkeys(out_map, trunk),
        }
    index["weight_map"].update(dict.fromkeys(tensors, MTP_SHARD))
    index["metadata"]["total_size"] = int(index["metadata"].get("total_size", 0)) + sum(
        t.numel() * t.element_size() for t in tensors.values()
    )
    (output_dir / INDEX_NAME).write_text(json.dumps(index, indent=2))

    LOG.info(
        "Copied %d MTP tensors from %s into %s", len(tensors), base_model, output_dir
    )
    return len(tensors)
