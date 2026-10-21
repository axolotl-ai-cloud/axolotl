import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.utils.mtp import MTP_SHARD, _is_mtp_key, copy_mtp_weights


def _write_model(path, tensors, config):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    save_file(tensors, path / "model.safetensors")


def _shard_keys(path):
    with safe_open(path / MTP_SHARD, framework="pt") as f:
        return set(f.keys())


@pytest.mark.parametrize("nextn", [1, None])
def test_selects_only_mtp_keys(tmp_path, nextn):
    config = {
        "text_config": {"num_hidden_layers": 1, "num_nextn_predict_layers": nextn}
    }
    trunk = {"model.layers.0.w": torch.ones(2)}
    base = {
        **trunk,
        "lm_head.weight": torch.ones(2),
        "model.layers.1.eh_proj.weight": torch.full((2,), 3.0),
        "mtp.fc.weight": torch.zeros(2),
    }
    _write_model(tmp_path / "base", base, config)
    _write_model(tmp_path / "out", trunk, config)

    expected = {"mtp.fc.weight"} | (
        {"model.layers.1.eh_proj.weight"} if nextn else set()
    )
    assert copy_mtp_weights(str(tmp_path / "base"), tmp_path / "out") == len(expected)
    assert _shard_keys(tmp_path / "out") == expected


@pytest.mark.parametrize(
    "key,expected",
    [
        ("model.mtp_block.0.mlp.gate.weight", True),
        ("model.mtp_emb_norm.0.weight", True),
        ("model.language_model.layers.1.eh_proj.weight", True),
        ("model.language_model.layers.0.mlp.gate.weight", False),
        ("model.visual.blocks.1.attn.qkv.weight", False),
    ],
)
def test_mtp_key_layouts(key, expected):
    assert _is_mtp_key(key, num_layers=1) == expected


@pytest.mark.parametrize("max_shard_size", ["1GB", "1KB"])
def test_mtp_survives_resave(tmp_path, max_shard_size):
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model.config.num_nextn_predict_layers = 1
    model.save_pretrained(tmp_path / "base")
    base_tensors = {
        "model.layers.1.eh_proj.weight": torch.ones(4),
        **{k: v.contiguous() for k, v in model.state_dict().items()},
    }
    save_file(base_tensors, tmp_path / "base" / "model.safetensors")

    out = tmp_path / "out"
    for _ in range(2):
        # second pass mimics the re-save inside push_to_hub
        model.save_pretrained(out, max_shard_size=max_shard_size)
        copy_mtp_weights(str(tmp_path / "base"), out)

        assert "model.layers.1.eh_proj.weight" in _shard_keys(out)
        index_path = out / "model.safetensors.index.json"
        assert index_path.exists() == (max_shard_size == "1KB")
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            assert weight_map["model.layers.1.eh_proj.weight"] == MTP_SHARD
            assert all((out / f).exists() for f in set(weight_map.values()))

    assert copy_mtp_weights(str(tmp_path / "base"), out) == 0


def test_quantized_base_into_unquantized_output_is_skipped(tmp_path):
    config = {"num_hidden_layers": 1, "num_nextn_predict_layers": 1}
    trunk = {"model.layers.0.w": torch.ones(2)}
    _write_model(
        tmp_path / "base",
        {**trunk, "model.layers.1.w": torch.ones(2)},
        {**config, "quantization_config": {"quant_method": "fp8"}},
    )
    _write_model(tmp_path / "out", trunk, config)

    assert copy_mtp_weights(str(tmp_path / "base"), tmp_path / "out") == 0
    assert not (tmp_path / "out" / MTP_SHARD).exists()
