"""Nemotron VLM profile: exclusive matching, text-scoped adapters, frozen vision."""

from types import SimpleNamespace

import pytest
from torch import nn

from axolotl.model_support.nemotron_diffusion import compat
from axolotl.model_support.nemotron_diffusion.compat import LM_VARIANT, VLM_VARIANT
from axolotl.model_support.nemotron_diffusion_vlm import (
    _freeze_vision,
    _model_class,
    _reject_vision_adapters,
    _validate,
    scope_lora_targets_to_text_layers,
)
from axolotl.model_support.profile import ModelHookContext
from axolotl.model_support.registry import get_model_support_for_cfg
from axolotl.utils.dict import DictDefault


@pytest.mark.parametrize(
    "base_model,expected",
    [
        ("nvidia/Nemotron-Labs-Diffusion-8B", "NemotronDiffusionSupport"),
        ("nvidia/Nemotron-Labs-Diffusion-VLM-8B", "NemotronDiffusionVLMSupport"),
        ("NVIDIA/nemotron-labs-diffusion-vlm-8b", "NemotronDiffusionVLMSupport"),
    ],
)
def test_lm_and_vlm_matchers_are_exclusive(base_model, expected):
    support = get_model_support_for_cfg(DictDefault(base_model=base_model))
    assert type(support).__name__ == expected


def test_local_source_matches_by_modeling_file(tmp_path):
    (tmp_path / VLM_VARIANT.modeling_file).write_text("")
    (tmp_path / LM_VARIANT.modeling_file).write_text("")
    support = get_model_support_for_cfg(DictDefault(base_model=str(tmp_path)))
    assert type(support).__name__ == "NemotronDiffusionVLMSupport"


def test_resolver_requires_the_variant_files(tmp_path):
    (tmp_path / VLM_VARIANT.modeling_file).write_text("")
    with pytest.raises(ValueError, match="Nemotron VLM source lacks") as info:
        compat.resolve_nemotron_model_class(tmp_path, variant=VLM_VARIANT)
    assert VLM_VARIANT.configuration_file in str(info.value)
    assert "modeling_ministral.py" in str(info.value)


def test_auto_model_class_passes_the_vlm_variant(monkeypatch):
    seen = []

    class Mock:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return cls()

    def resolve(model_source, *, revision=None, variant=LM_VARIANT):
        seen.append((model_source, revision, variant))
        return Mock

    monkeypatch.setattr(compat, "resolve_nemotron_model_class", resolve)
    assert isinstance(_model_class().from_pretrained("repo", revision="r1"), Mock)
    assert seen == [("repo", "r1", VLM_VARIANT)]


@pytest.mark.parametrize(
    "targets,expected",
    [
        (
            ["q_proj", "v_proj", "q_proj"],
            r"^encoder\.layers\.\d+\.(?:self_attn|mlp)\.(?:q_proj|v_proj)$",
        ),
        ("^encoder\\.layers\\..*", "^encoder\\.layers\\..*"),
        (["encoder.layers.0.self_attn.q_proj"], ["encoder.layers.0.self_attn.q_proj"]),
        ([], []),
    ],
)
def test_plain_targets_are_scoped_to_text_layers(targets, expected):
    assert scope_lora_targets_to_text_layers(targets) == expected


def test_validate_rewrites_plain_targets_and_defaults_attention():
    cfg = DictDefault(
        trust_remote_code=True,
        diffusion_lm={},
        adapter="lora",
        lora_target_modules=["q_proj", "k_proj"],
    )
    _validate(ModelHookContext(cfg=cfg, model_config=None))
    assert cfg.attn_implementation == "flex_attention"
    assert cfg.lora_target_modules.startswith(r"^encoder\.layers\.")


def test_validate_requires_remote_code():
    cfg = DictDefault(trust_remote_code=False, diffusion_lm={}, adapter="lora")
    with pytest.raises(ValueError, match="trust_remote_code"):
        _validate(ModelHookContext(cfg=cfg))


class _Lora(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(2, 2)
        self.lora_A = nn.Linear(2, 1, bias=False)
        self.lora_B = nn.Linear(1, 2, bias=False)


def _vlm_like(*, vision_lora: bool):
    encoder = nn.Module()
    encoder.layers = nn.ModuleList([nn.Module()])
    encoder.layers[0].self_attn = nn.Module()
    encoder.layers[0].self_attn.q_proj = _Lora()
    encoder.vision_tower = nn.Module()
    encoder.vision_tower.attention = nn.Module()
    encoder.vision_tower.attention.q_proj = _Lora() if vision_lora else nn.Linear(2, 2)
    encoder.multi_modal_projector = nn.Linear(2, 2)
    model = nn.Module()
    model.encoder = encoder
    model.diffusion_head = nn.Linear(2, 2)
    return model


def test_freeze_vision_leaves_text_trainable():
    model = _vlm_like(vision_lora=False)
    _freeze_vision(ModelHookContext(cfg=DictDefault(), model=model))
    grads = {name: p.requires_grad for name, p in model.named_parameters()}
    assert not any(v for n, v in grads.items() if "vision_tower" in n)
    assert not any(v for n, v in grads.items() if "multi_modal_projector" in n)
    assert all(v for n, v in grads.items() if n.startswith("encoder.layers"))
    assert grads["diffusion_head.weight"]


def test_adapter_guard_rejects_vision_lora_and_accepts_text_lora():
    _reject_vision_adapters(
        ModelHookContext(cfg=DictDefault(), model=_vlm_like(vision_lora=False))
    )
    with pytest.raises(ValueError, match="vision modules"):
        _reject_vision_adapters(
            ModelHookContext(cfg=DictDefault(), model=_vlm_like(vision_lora=True))
        )


def test_flex_attention_class_name_follows_the_variant(monkeypatch):
    import sys
    import types

    module = types.ModuleType("fake_vlm_modeling")
    module.NemotronLabsDiffusionVLMFlexAttention = type("Flex", (), {})
    module.Ministral3Attention = type("Dense", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model_class = type("Model", (), {"__module__": module.__name__})
    monkeypatch.setattr(
        compat, "resolve_nemotron_model_class", lambda *a, **k: model_class
    )
    from axolotl.model_support.nemotron_diffusion_vlm import _lora_attention_cls

    cfg = SimpleNamespace(
        base_model="x", revision_of_model=None, attn_implementation="flex_attention"
    )
    assert _lora_attention_cls(cfg) is module.NemotronLabsDiffusionVLMFlexAttention
    cfg.attn_implementation = "sdpa"
    assert _lora_attention_cls(cfg) is module.Ministral3Attention


def test_masking_utils_alias_is_restored_for_the_remote_import(monkeypatch):
    from transformers import masking_utils

    monkeypatch.delattr(masking_utils, "sdpa_mask_older_torch", raising=False)
    compat.ensure_masking_utils_aliases()
    assert masking_utils.sdpa_mask_older_torch is masking_utils.sdpa_mask


def test_mask_builder_kwargs_are_adapted_inside_the_remote_module():
    import types

    seen = []

    def create_causal_mask(*, config, inputs_embeds, attention_mask, position_ids=None):
        seen.append((config, inputs_embeds, attention_mask, position_ids))
        return "mask"

    module = types.ModuleType("fake_ministral")
    module.create_causal_mask = create_causal_mask
    compat.adapt_mask_function_kwargs(module)
    compat.adapt_mask_function_kwargs(module)

    result = module.create_causal_mask(
        config="cfg",
        input_embeds="emb",
        attention_mask="am",
        cache_position="cp",
        position_ids="pos",
    )
    assert result == "mask"
    assert seen == [("cfg", "emb", "am", "pos")]
    assert create_causal_mask(config=1, inputs_embeds=2, attention_mask=3) == "mask"
