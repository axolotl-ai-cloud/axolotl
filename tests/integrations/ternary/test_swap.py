"""CPU-only tests for the strict ternary module swap and its manifest."""

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import subln as subln_module, swap as swap_module
from axolotl.integrations.ternary.modules import TernaryLinear, iter_ternary_modules
from axolotl.integrations.ternary.swap import (
    MANIFEST_FILENAME,
    SwapManifest,
    convert_model,
    get_manifest,
    module_family,
    resolve_preset,
)
from axolotl.utils.dict import DictDefault

LAYERS = 2
LINEARS_PER_LAYER = 7


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _cfg(**ternary) -> DictDefault:
    return DictDefault({"ternary": ternary})


class _FusedExperts(nn.Module):
    """Stand-in for an arch that stores experts as fused 3D parameter stacks."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 64, 128))
        self.down_proj = nn.Parameter(torch.zeros(4, 128, 64))


def test_convert_swaps_every_block_linear():
    model = _tiny_llama()

    manifest = convert_model(model, _cfg())

    assert len(manifest.entries) == LAYERS * LINEARS_PER_LAYER
    assert len(list(iter_ternary_modules(model))) == LAYERS * LINEARS_PER_LAYER
    assert {entry.family for entry in manifest.entries} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert manifest.model_type == "llama"
    assert manifest.weight_scale == "absmean"
    assert manifest.activation_bits == 8
    assert manifest.quantizer["scheme"] == "ternary"
    assert manifest.quantizer["weight_dequant_scale"] == "f16_round"


def test_convert_keeps_head_and_embeddings_fp():
    model = _tiny_llama()

    manifest = convert_model(model, _cfg())

    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, TernaryLinear)
    assert isinstance(model.model.embed_tokens, nn.Embedding)
    assert manifest.kept_fp == ["lm_head"]
    assert all("lm_head" not in entry.name for entry in manifest.entries)


def test_convert_preserves_shapes_and_weights():
    model = _tiny_llama()
    original = model.model.layers[0].self_attn.k_proj.weight

    manifest = convert_model(model, _cfg())

    swapped = model.model.layers[0].self_attn.k_proj
    assert isinstance(swapped, TernaryLinear)
    assert swapped.weight is original
    entry = next(
        e for e in manifest.entries if e.name == "model.layers.0.self_attn.k_proj"
    )
    assert (entry.in_features, entry.out_features) == (64, 32)


def test_manifest_attached_to_model():
    model = _tiny_llama()

    manifest = convert_model(model, _cfg())

    assert get_manifest(model) is manifest
    assert get_manifest(_tiny_llama()) is None


def test_manifest_roundtrip(tmp_path):
    model = _tiny_llama()
    manifest = convert_model(model, _cfg())

    path = manifest.save(tmp_path)

    assert path.name == MANIFEST_FILENAME
    assert SwapManifest.load(tmp_path) == manifest


def test_manifest_load_without_file(tmp_path):
    with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
        SwapManifest.load(tmp_path)


def test_convert_writes_manifest_to_output_dir(tmp_path):
    model = _tiny_llama()
    cfg = DictDefault({"ternary": {}, "output_dir": str(tmp_path)})

    manifest = convert_model(model, cfg)

    assert SwapManifest.load(tmp_path) == manifest


def test_strict_enumeration_rejects_unmatched_linear():
    model = _tiny_llama()
    model.model.layers[0].mlp.extra_proj = nn.Linear(64, 64, bias=False)

    with pytest.raises(ValueError, match="model.layers.0.mlp.extra_proj"):
        convert_model(model, _cfg())

    assert not list(iter_ternary_modules(model))


def test_strict_enumeration_off_keeps_unmatched_linear():
    model = _tiny_llama()
    model.model.layers[0].mlp.extra_proj = nn.Linear(64, 64, bias=False)

    manifest = convert_model(model, _cfg(strict_enumeration=False))

    assert isinstance(model.model.layers[0].mlp.extra_proj, nn.Linear)
    assert "model.layers.0.mlp.extra_proj" in manifest.kept_fp
    assert len(manifest.entries) == LAYERS * LINEARS_PER_LAYER


def test_overlapping_target_and_keep_regexes_error():
    model = _tiny_llama()
    cfg = _cfg(
        target_modules=[
            r".*\.self_attn\.(q|k|v|o)_proj",
            r".*\.mlp\.(gate|up|down)_proj",
        ],
        keep_fp_modules=[r".*\.mlp\.down_proj"],
    )

    with pytest.raises(ValueError, match="both ternary.target_modules"):
        convert_model(model, cfg)


def test_overlap_resolves_to_keep_without_strict_enumeration():
    model = _tiny_llama()
    cfg = _cfg(
        target_modules=[
            r".*\.self_attn\.(q|k|v|o)_proj",
            r".*\.mlp\.(gate|up|down)_proj",
        ],
        keep_fp_modules=[r".*\.mlp\.down_proj"],
        strict_enumeration=False,
    )

    manifest = convert_model(model, cfg)

    assert isinstance(model.model.layers[0].mlp.down_proj, nn.Linear)
    assert len(manifest.entries) == LAYERS * (LINEARS_PER_LAYER - 1)


def test_router_style_gate_is_not_matched_by_the_preset():
    model = _tiny_llama()
    model.model.layers[0].mlp.gate = nn.Linear(64, 4, bias=False)

    with pytest.raises(ValueError, match="model.layers.0.mlp.gate"):
        convert_model(model, _cfg())


def test_router_style_gate_stays_fp_when_enumerated():
    model = _tiny_llama()
    model.model.layers[0].mlp.gate = nn.Linear(64, 4, bias=False)

    manifest = convert_model(model, _cfg(keep_fp_modules=[r".*\.mlp\.gate"]))

    assert isinstance(model.model.layers[0].mlp.gate, nn.Linear)
    assert "model.layers.0.mlp.gate" in manifest.kept_fp
    assert isinstance(model.model.layers[0].mlp.gate_proj, TernaryLinear)


def test_targeting_lm_head_is_rejected():
    model = _tiny_llama()
    cfg = _cfg(
        target_modules=[
            r".*\.self_attn\.(q|k|v|o)_proj",
            r".*\.mlp\.(gate|up|down)_proj",
            r"lm_head",
        ]
    )

    with pytest.raises(ValueError, match="always stay full precision"):
        convert_model(model, cfg)


def test_biased_target_is_rejected():
    model = _tiny_llama()
    model.model.layers[1].self_attn.q_proj = nn.Linear(64, 64, bias=True)

    with pytest.raises(ValueError, match="biased Linears"):
        convert_model(model, _cfg())

    assert not list(iter_ternary_modules(model))


def test_fused_expert_stacks_are_rejected():
    model = _tiny_llama()
    model.model.layers[0].mlp.experts = _FusedExperts()

    with pytest.raises(NotImplementedError, match="fused MoE expert stacks"):
        convert_model(model, _cfg())


def test_double_conversion_is_rejected():
    model = _tiny_llama()
    convert_model(model, _cfg())

    with pytest.raises(ValueError, match="already a TernaryLinear"):
        convert_model(model, _cfg())


def test_unknown_model_type_needs_explicit_targets():
    model = _tiny_llama()
    model.config.model_type = "not_a_real_arch"

    with pytest.raises(ValueError, match="set ternary.target_modules"):
        convert_model(model, _cfg())

    manifest = convert_model(
        model,
        _cfg(
            target_modules=[r".*\.self_attn\.(q|k|v|o)_proj"],
            keep_fp_modules=[r".*\.mlp\..*"],
        ),
    )
    assert len(manifest.entries) == LAYERS * 4
    assert manifest.model_type == "not_a_real_arch"


def test_resolve_preset():
    assert (
        resolve_preset("qwen3").target_modules == resolve_preset("llama").target_modules
    )

    with pytest.raises(ValueError, match="no ternary arch preset"):
        resolve_preset("mixtral")


def test_module_family():
    assert module_family("model.layers.0.self_attn.q_proj") == "q_proj"
    assert module_family("lm_head") == "lm_head"


def test_swap_config_flows_into_modules():
    model = _tiny_llama()

    convert_model(
        model, _cfg(activation_bits=None, fused_fake_quant=False, int8_forward=False)
    )

    _, module = next(iter_ternary_modules(model))
    assert module.activation_bits is None
    assert module.fused is False
    assert module.int8_forward is False
    assert module.weight_scale == "absmean"
    assert module.group_size is None


def test_manifest_records_the_init_and_subln_settings():
    model = _tiny_llama()

    manifest = convert_model(model, _cfg())

    assert manifest.init == "absmean"
    assert manifest.subln is False
    assert manifest.subln_modules == []
    assert {entry.weight_scale for entry in manifest.entries} == {"absmean"}


def test_convert_hands_the_swapped_model_to_the_init_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(
        swap_module,
        "initialize_model_latents",
        lambda model, manifest, cfg: calls.append((model, manifest)),
    )
    model = _tiny_llama()

    manifest = convert_model(model, _cfg())

    assert calls == [(model, manifest)]


def test_subln_is_inserted_on_the_linears_before_they_are_swapped(monkeypatch):
    seen: dict[str, object] = {}

    def _insert(model, cfg, target_names):
        names = list(target_names)
        seen["kinds"] = {type(model.get_submodule(name)).__name__ for name in names}
        return [f"{name}.sub_norm" for name in names]

    monkeypatch.setattr(subln_module, "insert_subln", _insert)
    model = _tiny_llama()

    manifest = convert_model(
        model, _cfg(subln=True, export={"formats": ["master_bf16"]})
    )

    assert seen["kinds"] == {"Linear"}
    assert manifest.subln is True
    assert len(manifest.subln_modules) == LAYERS * LINEARS_PER_LAYER


def test_swap_defaults_int8_forward_to_auto():
    model = _tiny_llama()

    convert_model(model, _cfg())

    _, module = next(iter_ternary_modules(model))
    assert module.int8_forward == "auto"
