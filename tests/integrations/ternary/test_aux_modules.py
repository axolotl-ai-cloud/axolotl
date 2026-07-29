"""CPU-only tests for the auxiliary-module registry and the seams that read it."""

import re

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import aux_modules
from axolotl.integrations.ternary.aux_modules import (
    AUX_MODULE_FAMILIES,
    ROUTER_CANARY_NAMES,
)
from axolotl.integrations.ternary.modules import TernaryLinear, iter_ternary_modules
from axolotl.integrations.ternary.swap import convert_model, resolve_preset
from axolotl.utils.dict import DictDefault

from .test_export_hf import QWEN35_LINEARS

ROUTER_NAME = "model.layers.0.mlp.gate"


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _cfg(**ternary) -> DictDefault:
    return DictDefault({"ternary": ternary})


# ------------------------------------------------------------------- the registry


@pytest.mark.parametrize("key", sorted(AUX_MODULE_FAMILIES))
def test_every_family_is_well_formed(key):
    family = AUX_MODULE_FAMILIES[key]

    assert family.key == key
    assert family.label and family.rationale
    assert family.patterns
    for pattern in family.patterns:
        re.compile(pattern)


@pytest.mark.parametrize("key", sorted(AUX_MODULE_FAMILIES))
def test_every_canary_classifies_into_its_own_family(key):
    family = AUX_MODULE_FAMILIES[key]

    assert family.canaries, f"{key} has no canary to warn a target regex against"
    for canary in family.canaries:
        assert aux_modules.classify(canary) is family, canary


def test_router_canaries_come_from_the_registry():
    assert ROUTER_CANARY_NAMES == AUX_MODULE_FAMILIES["routers"].canaries
    assert "model.layers.0.mlp.gate" in ROUTER_CANARY_NAMES


@pytest.mark.parametrize(
    "name,key",
    [
        ("model.layers.0.mlp.gate", "routers"),
        ("model.layers.0.block_sparse_moe.gate", "routers"),
        ("model.layers.3.mlp.router", "routers"),
        ("model.visual.blocks.0.attn.qkv", "vision_towers"),
        ("visual.blocks.0.attn.qkv", "vision_towers"),
        ("model.vision_tower.encoder.layers.0.self_attn.q_proj", "vision_towers"),
        ("vision_model.encoder.layers.0.mlp.fc1", "vision_towers"),
        ("model.multi_modal_projector.linear_1", "projectors"),
        ("model.mm_projector.0", "projectors"),
        ("model.language_model.merger.linear_fc1", "projectors"),
        ("model.layers.0.mtp.up_proj", "mtp"),
        ("model.nextn.embed_proj", "mtp"),
        ("medusa_head.0.linear", "draft_heads"),
        ("eagle_head.layers.0.q_proj", "draft_heads"),
        ("draft_model.layers.0.mlp.up_proj", "draft_heads"),
        ("value_head", "value_heads"),
        ("v_head.summary", "value_heads"),
        ("score", "value_heads"),
    ],
)
def test_known_aux_names_classify(name, key):
    family = aux_modules.classify(name)

    assert family is not None, name
    assert family.key == key


@pytest.mark.parametrize("name", QWEN35_LINEARS)
def test_the_registry_never_claims_a_block_projection(name):
    """The real qwen3_5 Linear names: only the tower and its merger are auxiliary."""
    family = aux_modules.classify(name)

    if name.startswith("model.visual."):
        assert family is not None and family.key == "vision_towers"
    else:
        assert family is None, name


def test_linear_attention_projections_are_not_an_aux_family():
    """They are kept FP by the qwen3_5 preset for a different reason, not by name."""
    assert (
        aux_modules.classify("model.language_model.layers.1.linear_attn.in_proj_a")
        is None
    )


def test_suggest_keep_fp_is_paste_ready():
    names = [ROUTER_NAME, "model.visual.blocks.0.attn.qkv"]

    patterns = aux_modules.suggest_keep_fp(names)

    assert patterns
    compiled = [re.compile(pattern) for pattern in patterns]
    for name in names:
        assert any(pattern.fullmatch(name) for pattern in compiled), name


def test_suggest_keep_fp_only_returns_patterns_that_matched():
    patterns = aux_modules.suggest_keep_fp([ROUTER_NAME])

    assert patterns == [r"(.*\.)?mlp\.gate"]


def test_group_by_family_keeps_registry_order():
    grouped = aux_modules.group_by_family(
        ["score", "model.visual.blocks.0.attn.qkv", ROUTER_NAME]
    )

    assert list(grouped) == ["routers", "vision_towers", "value_heads"]


def test_explain_names_the_family_the_reason_and_the_fix():
    text = aux_modules.explain([ROUTER_NAME])

    assert "MoE router" in text
    assert "top-k" in text
    assert ROUTER_NAME in text
    assert "keep_fp_modules" in text
    assert r"    - (.*\.)?mlp\.gate" in text


def test_explain_flags_a_best_effort_family():
    text = aux_modules.explain(["model.layers.0.mtp.up_proj"])

    assert "multi-token-prediction head" in text
    assert "name-based detection only" in text


def test_explain_is_empty_for_ordinary_modules():
    assert aux_modules.explain(["model.layers.0.mlp.extra_proj"]) == ""


def test_family_patterns_rejects_an_unknown_family():
    assert (
        aux_modules.family_patterns("routers")
        == AUX_MODULE_FAMILIES["routers"].patterns
    )

    with pytest.raises(KeyError):
        aux_modules.family_patterns("not_a_family")


# -------------------------------------------------------------- the swap seam


def test_the_qwen3_5_preset_takes_its_vision_patterns_from_the_registry():
    keeps = resolve_preset("qwen3_5").keep_fp_modules

    for pattern in AUX_MODULE_FAMILIES["vision_towers"].patterns:
        assert pattern in keeps
    assert any("linear_attn" in pattern for pattern in keeps)


def test_a_preset_and_a_user_list_do_not_repeat_a_pattern():
    model = _tiny_llama()
    model.model.layers[0].mlp.gate = nn.Linear(64, 4, bias=False)
    shared = r"(.*\.)?mlp\.gate"

    manifest = convert_model(model, _cfg(keep_fp_modules=[shared, shared]))

    assert "model.layers.0.mlp.gate" in manifest.kept_fp


def test_strict_enumeration_error_says_the_module_is_a_router():
    model = _tiny_llama()
    model.model.layers[0].mlp.gate = nn.Linear(64, 4, bias=False)

    with pytest.raises(ValueError) as excinfo:
        convert_model(model, _cfg())

    message = str(excinfo.value)
    assert "MoE router" in message
    assert "top-k" in message
    assert r"    - (.*\.)?mlp\.gate" in message


def test_the_suggested_pattern_actually_fixes_the_error():
    """The paste-ready line has to work when pasted, or it is worse than nothing."""
    model = _tiny_llama()
    model.model.layers[0].mlp.gate = nn.Linear(64, 4, bias=False)
    with pytest.raises(ValueError) as excinfo:
        convert_model(model, _cfg())
    suggested = [
        line.strip()[2:]
        for line in str(excinfo.value).splitlines()
        if line.startswith("    - ")
    ]

    manifest = convert_model(_fresh_with_router(), _cfg(keep_fp_modules=suggested))

    assert "model.layers.0.mlp.gate" in manifest.kept_fp


def _fresh_with_router() -> LlamaForCausalLM:
    model = _tiny_llama()
    model.model.layers[0].mlp.gate = nn.Linear(64, 4, bias=False)
    return model


def test_an_ordinary_unmatched_linear_gets_no_family_section():
    model = _tiny_llama()
    model.model.layers[0].mlp.extra_proj = nn.Linear(64, 64, bias=False)

    with pytest.raises(ValueError) as excinfo:
        convert_model(model, _cfg())

    message = str(excinfo.value)
    assert "model.layers.0.mlp.extra_proj" in message
    assert "keep_fp_modules:" not in message


def test_non_strict_enumeration_still_names_the_family(caplog):
    model = _fresh_with_router()

    with caplog.at_level("WARNING"):
        manifest = convert_model(model, _cfg(strict_enumeration=False))

    assert "model.layers.0.mlp.gate" in manifest.kept_fp
    assert "MoE router" in caplog.text


def test_deliberately_targeting_a_router_warns(caplog):
    model = _fresh_with_router()

    with caplog.at_level("WARNING"):
        manifest = convert_model(
            model,
            _cfg(
                target_modules=[
                    r".*\.self_attn\.(q|k|v|o)_proj",
                    r".*\.mlp\.(gate|up|down)_proj",
                    r".*\.mlp\.gate",
                ]
            ),
        )

    assert isinstance(model.model.layers[0].mlp.gate, TernaryLinear)
    assert any(entry.name.endswith("mlp.gate") for entry in manifest.entries)
    assert "keeps full precision by default" in caplog.text


def test_the_swap_applies_the_configured_codebook():
    """The codebook reaches a module only through the swap, so the swap must carry it."""
    model = _tiny_llama()

    manifest = convert_model(model, _cfg(codebook="binary"))

    assert manifest.codebook == "binary"
    assert {module.codebook for _, module in iter_ternary_modules(model)} == {"binary"}
    assert {entry.codebook for entry in manifest.entries} == {"binary"}
