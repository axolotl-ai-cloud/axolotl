"""The mistral_large4 MoE renamings must be fixed points on HF keys (PEFT re-applies them)."""

from transformers.core_model_loading import WeightRenaming, rename_source_key

from axolotl.monkeypatch.models.mistral_large4.conversion_mapping import (
    _idempotent_renamings,
)

HF_KEYS = [
    "model.language_model.layers.1.mlp.experts.gate_up_proj",
    "model.language_model.layers.1.mlp.shared_experts.up_proj.lora_A.weight",
    "model.language_model.layers.1.mlp.gate.weight",
]
NATIVE = {
    "layers.1.experts.0.w1.weight": "layers.1.mlp.experts.0.w1.weight",
    "layers.1.shared_experts.w1.weight": "layers.1.mlp.shared_experts.w1.weight",
    "layers.1.gate.weight": "layers.1.mlp.gate.weight",
}


def _upstream_renamings():
    return [
        WeightRenaming(r"\.gate\.", r"\.mlp.gate\."),
        WeightRenaming(r"\.experts\.", r"\.mlp.experts\."),
        WeightRenaming(r"\.shared_experts\.", r"\.mlp.shared_experts\."),
    ]


def test_upstream_renamings_are_not_idempotent():
    ren = _upstream_renamings()
    assert any(rename_source_key(k, ren, [])[0] != k for k in HF_KEYS)


def test_patched_renamings_keep_hf_keys_and_convert_native_keys():
    ren, changed = _idempotent_renamings(_upstream_renamings())
    assert changed == 3
    for key in HF_KEYS:
        assert rename_source_key(key, ren, [])[0] == key
    for native, hf in NATIVE.items():
        assert rename_source_key(native, ren, [])[0] == hf
    assert _idempotent_renamings(ren)[1] == 0
