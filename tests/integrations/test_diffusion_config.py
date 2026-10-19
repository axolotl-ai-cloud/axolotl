"""Compatibility tests for the legacy diffusion configuration shim."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from axolotl.core.trainers.diffusion_lm.config import get_diffusion_config
from axolotl.core.trainers.diffusion_lm.tokens import (
    resolve_mask_token_id as resolve_core_mask_token_id,
)
from axolotl.integrations.diffusion import plugin as plugin_module
from axolotl.integrations.diffusion.plugin import DiffusionPlugin
from axolotl.integrations.diffusion.utils import resolve_mask_token_id
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.diffusion import DiffusionLMConfig


@pytest.fixture(autouse=True)
def reset_legacy_warning(monkeypatch):
    monkeypatch.setattr(plugin_module, "_LEGACY_WARNING_EMITTED", False)


def test_legacy_default_translates_once_and_is_idempotent():
    cfg = DictDefault()
    plugin = DiffusionPlugin()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plugin.register(cfg)
        plugin.register(cfg)

    assert cfg.diffusion_lm["from_causal_lm"] is True
    assert len(caught) == 1
    assert "deprecated" in str(caught[0].message)


def test_legacy_mapping_and_pydantic_values_translate_to_canonical_config():
    plugin = DiffusionPlugin()
    mapping_cfg = DictDefault(diffusion={"mask_token_id": 17})
    model_cfg = DictDefault(diffusion=DiffusionLMConfig(mask_token_id=19))

    plugin.register(mapping_cfg)
    plugin.register(model_cfg)

    assert get_diffusion_config(mapping_cfg).mask_token_id == 17
    assert get_diffusion_config(model_cfg).mask_token_id == 19
    assert mapping_cfg.diffusion_lm["from_causal_lm"] is True
    assert model_cfg.diffusion_lm["from_causal_lm"] is True


def test_plain_dict_legacy_config_translates_to_attribute_accessible_settings():
    cfg = {"diffusion": {"mask_token_id": 29}}

    DiffusionPlugin().register(cfg)

    assert get_diffusion_config(cfg).mask_token_id == 29
    assert cfg["diffusion_lm"].from_causal_lm is True


def test_repeat_registration_rejects_a_mutated_translated_block():
    cfg = DictDefault(diffusion={"mask_token_id": 17})
    plugin = DiffusionPlugin()
    plugin.register(cfg)
    cfg.diffusion_lm["mask_token_id"] = 19

    with pytest.raises(ValueError, match="only one of `diffusion` and `diffusion_lm`"):
        plugin.register(cfg)


def test_conflicting_legacy_and_canonical_blocks_are_rejected():
    cfg = DictDefault(
        diffusion={"mask_token_id": 17},
        diffusion_lm={"mask_token_id": 19, "from_causal_lm": True},
    )

    with pytest.raises(ValueError, match="only one of `diffusion` and `diffusion_lm`"):
        DiffusionPlugin().register(cfg)


def test_causal_compatible_canonical_config_is_accepted_with_legacy_plugin():
    cfg = DictDefault(
        diffusion_lm=DiffusionLMConfig(mask_token_id=23, from_causal_lm=True)
    )

    DiffusionPlugin().register(cfg)

    assert cfg.diffusion_lm.mask_token_id == 23
    assert cfg.diffusion_lm.from_causal_lm is True


def test_native_canonical_config_is_rejected_by_legacy_plugin():
    cfg = DictDefault(diffusion_lm=DiffusionLMConfig(from_causal_lm=False))

    with pytest.raises(ValueError, match="remove the plugin"):
        DiffusionPlugin().register(cfg)


def test_mask_resolution_after_translation_updates_both_config_blocks():
    tokenizer = SimpleNamespace(
        vocab_size=32,
        unk_token_id=0,
        all_special_tokens=["<mask>"],
        additional_special_tokens=[],
        convert_tokens_to_ids=lambda value: 7 if value == "<mask>" else 0,
    )
    cfg = DictDefault(
        diffusion=DictDefault(mask_token_str="<mask>", mask_token_id=None)
    )
    DiffusionPlugin().register(cfg)

    assert resolve_core_mask_token_id(tokenizer, cfg, allow_add=False) == 7
    assert cfg.diffusion_lm["mask_token_id"] == 7
    assert cfg.diffusion.mask_token_id == 7


def test_legacy_mask_resolution_mutates_the_legacy_configuration():
    tokenizer = SimpleNamespace(
        vocab_size=32,
        unk_token_id=0,
        all_special_tokens=["<mask>"],
        additional_special_tokens=[],
        convert_tokens_to_ids=lambda value: 7 if value == "<mask>" else 0,
    )
    cfg = DictDefault(
        diffusion=DictDefault(mask_token_str="<mask>", mask_token_id=None)
    )

    assert resolve_mask_token_id(tokenizer, cfg, allow_add=False) == 7
    assert cfg.diffusion.mask_token_id == 7


def test_equal_legacy_warmup_ratio_normalizes_fractional_steps():
    cfg = DictDefault(
        diffusion={},
        warmup_steps=0.1,
        warmup_ratio=0.1,
    )

    DiffusionPlugin().register(cfg)

    assert cfg.warmup_steps is None
    assert cfg.warmup_ratio == 0.1


def test_conflicting_legacy_warmup_ratio_is_rejected():
    cfg = DictDefault(
        diffusion={},
        warmup_steps=0.1,
        warmup_ratio=0.2,
    )

    with pytest.raises(ValueError, match="fractional warmup_steps conflicts"):
        DiffusionPlugin().register(cfg)


def test_integer_legacy_warmup_steps_remain_unchanged():
    cfg = DictDefault(diffusion={}, warmup_steps=10)

    DiffusionPlugin().register(cfg)

    assert cfg.warmup_steps == 10
    assert cfg.warmup_ratio is None


def test_canonical_only_settings_are_not_normalized_by_legacy_plugin():
    cfg = DictDefault(
        diffusion_lm=DiffusionLMConfig(from_causal_lm=True),
        warmup_steps=0.1,
        save_strategy="best",
        metric_for_best_model=None,
    )

    DiffusionPlugin().register(cfg)

    assert cfg.warmup_steps == 0.1
    assert cfg.warmup_ratio is None
    assert cfg.metric_for_best_model is None
