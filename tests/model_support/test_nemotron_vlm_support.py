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


def test_local_vlm_source_named_like_the_lm_matches_the_vlm(tmp_path):
    source = tmp_path / "nemotron-labs-diffusion-out"
    source.mkdir()
    (source / VLM_VARIANT.modeling_file).write_text("")
    support = get_model_support_for_cfg(DictDefault(base_model=str(source)))
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
        (["diffusion_head"], ["diffusion_head"]),
        ([], []),
    ],
)
def test_plain_targets_are_scoped_to_text_layers(targets, expected):
    assert scope_lora_targets_to_text_layers(targets) == expected


_MODULE_KEYS = (
    "encoder.layers.0.self_attn.q_proj",
    "encoder.layers.1.self_attn.q_proj",
    "encoder.layers.1.mlp.down_proj",
    "encoder.vision_tower.transformer.layers.0.attention.q_proj",
    "encoder.vision_tower.transformer.layers.1.feed_forward.down_proj",
    "diffusion_head",
    "encoder.layers.1.self_attn.k_proj",
)


def _matched(pattern):
    import re

    return [key for key in _MODULE_KEYS if re.fullmatch(pattern, key)]


def test_scoping_keeps_non_shared_targets_with_suffix_semantics():
    pattern = scope_lora_targets_to_text_layers(
        ["q_proj", "down_proj", "diffusion_head", "self_attn.k_proj"]
    )
    assert _matched(pattern) == [
        "encoder.layers.0.self_attn.q_proj",
        "encoder.layers.1.self_attn.q_proj",
        "encoder.layers.1.mlp.down_proj",
        "diffusion_head",
        "encoder.layers.1.self_attn.k_proj",
    ]


def test_scoping_folds_layers_to_transform_into_the_pattern():
    pattern = scope_lora_targets_to_text_layers(["q_proj", "k_proj"], [1])
    assert _matched(pattern) == [
        "encoder.layers.1.self_attn.q_proj",
        "encoder.layers.1.self_attn.k_proj",
    ]
    cfg = DictDefault(
        trust_remote_code=True,
        diffusion_lm={},
        adapter="lora",
        lora_target_modules=["q_proj", "v_proj"],
        peft_layers_to_transform=[0, 1],
    )
    _validate(ModelHookContext(cfg=cfg, model_config=None))
    assert cfg.peft_layers_to_transform is None
    assert r"encoder\.layers\.(?:0|1)\." in cfg.lora_target_modules
    from peft import LoraConfig

    LoraConfig(
        target_modules=cfg.lora_target_modules,
        layers_to_transform=cfg.peft_layers_to_transform,
    )


def test_scoping_rejects_projector_only_targets():
    with pytest.raises(ValueError, match="exist only in the frozen vision"):
        scope_lora_targets_to_text_layers(["q_proj", "linear_1"])


def _fsdp_cfg(**fsdp_config):
    return DictDefault(
        trust_remote_code=True,
        diffusion_lm={},
        adapter="lora",
        fsdp_version=2,
        fsdp_config=fsdp_config,
    )


def test_fsdp_transformer_wrap_defaults_to_decoder_layers(monkeypatch):
    monkeypatch.delenv("FSDP_TRANSFORMER_CLS_TO_WRAP", raising=False)
    cfg = _fsdp_cfg(auto_wrap_policy="TRANSFORMER_BASED_WRAP")
    _validate(ModelHookContext(cfg=cfg, model_config=None))
    assert cfg.fsdp_config["transformer_layer_cls_to_wrap"] == "Ministral3DecoderLayer"
    import os

    assert os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] == "Ministral3DecoderLayer"


@pytest.mark.parametrize(
    "names",
    ["Ministral3DecoderLayer,Ministral3RMSNorm", "PixtralAttentionLayer"],
)
def test_fsdp_transformer_wrap_rejects_non_decoder_units(names):
    cfg = _fsdp_cfg(
        auto_wrap_policy="TRANSFORMER_BASED_WRAP", transformer_layer_cls_to_wrap=names
    )
    with pytest.raises(ValueError, match="must name only Ministral3DecoderLayer"):
        _validate(ModelHookContext(cfg=cfg, model_config=None))


def test_fsdp_without_transformer_wrap_is_left_alone():
    cfg = _fsdp_cfg(
        auto_wrap_policy="TRANSFORMER_BASED_WRAP",
        transformer_layer_cls_to_wrap="Ministral3DecoderLayer",
    )
    _validate(ModelHookContext(cfg=cfg, model_config=None))
    cfg = _fsdp_cfg(reshard_after_forward=True)
    _validate(ModelHookContext(cfg=cfg, model_config=None))
    assert "transformer_layer_cls_to_wrap" not in cfg.fsdp_config


def test_required_files_cover_the_vlm_modeling_relative_imports():
    import re
    from pathlib import Path

    snapshot = Path(
        "/mnt/data/hf_cache/hub/models--nvidia--Nemotron-Labs-Diffusion-VLM-8B/"
        "snapshots/adca93d16471c1e07d594ae444d23e1876f6b365"
    )
    pending = [VLM_VARIANT.modeling_file]
    if not (snapshot / pending[0]).is_file():
        pytest.skip("Nemotron VLM snapshot not cached")
    needed = set()
    while pending:
        name = pending.pop()
        needed.add(name)
        text = (snapshot / name).read_text(encoding="utf-8")
        for module in re.findall(r"^\s*from \.(\w+) import", text, re.MULTILINE):
            if f"{module}.py" not in needed:
                pending.append(f"{module}.py")
    assert needed <= set(VLM_VARIANT.required_files)
    assert "chat_utils.py" in VLM_VARIANT.required_files


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


def _peft_wrapped(model):
    wrapper = nn.Module()
    wrapper.base_model = nn.Module()
    wrapper.base_model.model = model
    return wrapper


@pytest.mark.parametrize("wrap", [lambda model: model, _peft_wrapped])
def test_adapter_guard_rejects_vision_lora_and_accepts_text_lora(wrap):
    _reject_vision_adapters(
        ModelHookContext(cfg=DictDefault(), model=wrap(_vlm_like(vision_lora=False)))
    )
    with pytest.raises(ValueError, match="vision modules"):
        _reject_vision_adapters(
            ModelHookContext(cfg=DictDefault(), model=wrap(_vlm_like(vision_lora=True)))
        )


def test_freeze_vision_handles_the_peft_prefix():
    model = _peft_wrapped(_vlm_like(vision_lora=False))
    _freeze_vision(ModelHookContext(cfg=DictDefault(), model=model))
    grads = {name: p.requires_grad for name, p in model.named_parameters()}
    assert not any(v for n, v in grads.items() if "vision_tower" in n)
    assert all(v for n, v in grads.items() if ".encoder.layers" in n)


def test_lora_kernels_are_supported_on_the_vlm():
    from axolotl.model_support.base import Supported
    from axolotl.model_support.nemotron_diffusion_vlm import (
        NemotronDiffusionVLMSupport,
    )

    capabilities = NemotronDiffusionVLMSupport.profile.capabilities
    assert isinstance(capabilities["lora_kernels"], Supported)
    for name in ("fsdp", "quantized_lora", "diffusion_varlen"):
        assert "image batches" in capabilities[name].note


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


def test_freeze_vision_disables_complementary_mask():
    model = _vlm_like(vision_lora=False)
    model.config = SimpleNamespace(complementary_mask=True)
    _freeze_vision(ModelHookContext(cfg=DictDefault(), model=model))
    assert model.config.complementary_mask is False


def test_forward_drops_the_none_inputs_embeds_peft_passes(tmp_path, monkeypatch):
    for name in VLM_VARIANT.required_files:
        (tmp_path / name).write_text("")
    seen = []

    class Remote(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.encoder = SimpleNamespace(rotary_emb=None)

        def forward(self, input_ids, **kwargs):
            seen.append(kwargs)
            return "out"

    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        lambda *args, **kwargs: Remote,
    )
    monkeypatch.setattr(
        compat, "enable_nemotron_explicit_attention_mask", lambda m: None
    )
    monkeypatch.setattr(compat, "keep_rotary_fp32", lambda r: None)
    model_class = compat.resolve_nemotron_model_class(tmp_path, variant=VLM_VARIANT)
    model = model_class(SimpleNamespace(dlm_paradigm="bidirectional"))
    embeds = object()

    assert model(input_ids=1, inputs_embeds=None, pixel_values=2) == "out"
    assert model(input_ids=1, inputs_embeds=embeds) == "out"
    assert seen == [{"pixel_values": 2}, {"inputs_embeds": embeds}]


@pytest.mark.parametrize(
    "module_name,converted",
    [
        ("encoder.vision_tower.transformer.layers.0.attention.q_proj", False),
        ("encoder.multi_modal_projector.linear_1", False),
        ("diffusion_head", False),
        ("encoder.layers.0.self_attn.q_proj", True),
    ],
)
def test_4bit_skip_list_matches_the_real_module_names(module_name, converted):
    from transformers.quantizers.quantizers_utils import should_convert_module

    from axolotl.utils.nf4 import (
        architecture_skip_modules,
        nf4_should_quantize,
        nf4_skip_modules,
    )

    patterns = list(architecture_skip_modules("nemotron_labs_diffusion_vlm"))
    assert should_convert_module(module_name, patterns) is converted
    merge_skips = nf4_skip_modules("nemotron_labs_diffusion_vlm", {})
    assert (
        nf4_should_quantize(
            f"{module_name}.weight", linear=True, expert=False, skips=merge_skips
        )
        is converted
    )


def test_4bit_loader_passes_the_vlm_skip_list_to_bitsandbytes():
    from axolotl.loaders.model import ModelLoader

    loader = SimpleNamespace(
        cfg=DictDefault(
            adapter="qlora",
            load_in_4bit=True,
            model_config_type="nemotron_labs_diffusion_vlm",
            torch_dtype=None,
        ),
        model_config=SimpleNamespace(),
        model_kwargs={},
        is_fsdp_enabled=False,
    )
    ModelLoader._set_quantization_config(loader)
    assert loader.model_kwargs["quantization_config"].llm_int8_skip_modules == [
        "encoder.vision_tower",
        "encoder.multi_modal_projector",
        "diffusion_head",
    ]


def test_bnb_merge_selects_with_the_vlm_skip_list():
    from unittest.mock import patch

    from axolotl.cli.merge_lora import _do_merge_lora_efficient

    cfg = DictDefault(
        base_model="base",
        lora_model_dir="adapter",
        output_dir="out",
        model_config_type="nemotron_labs_diffusion_vlm",
        nf4_backend="bitsandbytes",
        _original_load_in_4bit=True,
        _original_adapter="qlora",
    )
    with patch("axolotl.cli.merge_lora.merge_lora_sharded_efficient") as merge:
        _do_merge_lora_efficient(cfg=cfg)
    assert merge.call_args.kwargs["bnb_skip_modules"] == [
        "encoder.vision_tower",
        "encoder.multi_modal_projector",
        "diffusion_head",
    ]
