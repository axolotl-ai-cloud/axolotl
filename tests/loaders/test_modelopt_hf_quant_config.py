"""A modelopt export that records its quantization only in hf_quant_config.json."""

import json

from transformers import Qwen3MoeConfig

from axolotl.loaders.utils import load_model_config
from axolotl.utils.dict import DictDefault

HF_QUANT = {
    "producer": {"name": "modelopt", "version": "0.33.0"},
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": ["model.layers.0.mlp.gate", "lm_head"],
    },
}


def _export(tmp_path, *, hf_quant=True, in_config=None):
    config = Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
    )
    if in_config is not None:
        config.quantization_config = in_config
    config.save_pretrained(tmp_path)
    if hf_quant:
        (tmp_path / "hf_quant_config.json").write_text(json.dumps(HF_QUANT))
    return DictDefault(base_model=str(tmp_path))


def test_quantization_config_from_hf_quant_config(tmp_path):
    qcfg = load_model_config(_export(tmp_path)).quantization_config
    assert qcfg["quant_method"] == "modelopt"
    assert qcfg["quant_algo"] == "NVFP4"
    assert qcfg["ignore"] == ["model.layers.0.mlp.gate", "lm_head"]
    group = qcfg["config_groups"]["group_0"]
    assert group["weights"]["num_bits"] == 4 and group["weights"]["group_size"] == 16
    assert qcfg["kv_cache_scheme"]["num_bits"] == 8


def test_config_json_quantization_config_wins(tmp_path):
    own = {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}
    qcfg = load_model_config(_export(tmp_path, in_config=own)).quantization_config
    assert qcfg["ignore"] == ["lm_head"] and "config_groups" not in qcfg


def test_unquantized_export_untouched(tmp_path):
    config = load_model_config(_export(tmp_path, hf_quant=False))
    assert getattr(config, "quantization_config", None) is None
