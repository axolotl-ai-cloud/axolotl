"""The mistral_large4 MoE renamings must be fixed points on HF keys (PEFT re-applies them)."""

import pytest
import transformers.conversion_mapping as cm
from transformers.core_model_loading import WeightRenaming, rename_source_key

from axolotl.monkeypatch.models.mistral_large4 import conversion_mapping as patch

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


def test_patched_renamings_keep_hf_keys_and_convert_native_keys():
    ren = _upstream_renamings()
    assert any(rename_source_key(k, ren, [])[0] != k for k in HF_KEYS)
    ren, changed = patch._idempotent_renamings(ren)
    assert changed == 3
    for key in HF_KEYS:
        assert rename_source_key(key, ren, [])[0] == key
    for native, hf in NATIVE.items():
        assert rename_source_key(native, ren, [])[0] == hf


@pytest.mark.parametrize(
    "known, expected", [("unpatched", 3), ("patched", 0), (None, ValueError)]
)
def test_patch_only_applies_to_the_fingerprinted_mapping(monkeypatch, known, expected):
    mapping = _upstream_renamings()
    registered = {}
    fingerprint = patch._renamings_fingerprint(mapping)
    if known == "unpatched":
        monkeypatch.setattr(patch, "_UNPATCHED_SHA256", fingerprint)
    elif known == "patched":
        monkeypatch.setattr(patch, "_PATCHED_SHA256", fingerprint)
    monkeypatch.setattr(cm, "get_checkpoint_conversion_mapping", lambda _: mapping)
    monkeypatch.setattr(
        cm,
        "register_checkpoint_conversion_mapping",
        lambda model_type, m, overwrite: registered.update({model_type: m}),
    )
    if expected is ValueError:
        with pytest.raises(ValueError, match="inform the axolotl maintainers"):
            patch.patch_mistral_large4_moe_renamings()
        return
    assert patch.patch_mistral_large4_moe_renamings() == expected
    assert bool(registered) == bool(expected)
