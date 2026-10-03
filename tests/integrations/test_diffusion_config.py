"""Compatibility tests for canonical diffusion settings and the old alias."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from axolotl.integrations.diffusion import plugin as plugin_module
from axolotl.integrations.diffusion.args import DiffusionArgs
from axolotl.integrations.diffusion.lm.config import get_diffusion_config
from axolotl.integrations.diffusion.lm.tokens import resolve_mask_token_id
from axolotl.integrations.diffusion.plugin import DiffusionPlugin
from axolotl.integrations.diffusion.schema import DiffusionLMConfig
from axolotl.utils.dict import DictDefault


@pytest.fixture(autouse=True)
def reset_alias_warning(monkeypatch):
    monkeypatch.setattr(plugin_module, "_ALIAS_WARNING_EMITTED", False)


def test_existing_diffusion_block_defaults_to_causal_mode_and_is_idempotent():
    cfg = DictDefault(diffusion={"mask_token_id": 17})
    plugin = DiffusionPlugin()

    plugin.register(cfg)
    plugin.register(cfg)

    assert cfg.diffusion.from_causal_lm is True
    assert cfg.diffusion.mask_token_id == 17
    assert "diffusion_lm" not in cfg


def test_plugin_without_settings_keeps_legacy_causal_default():
    cfg = DictDefault()

    DiffusionPlugin().register(cfg)

    assert cfg.diffusion.from_causal_lm is True


def test_deprecated_alias_normalizes_once_to_native_mode():
    cfg = DictDefault(diffusion_lm={"mask_token_id": 19})
    plugin = DiffusionPlugin()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plugin.register(cfg)
        plugin.register(cfg)

    assert cfg.diffusion.from_causal_lm is False
    assert cfg.diffusion.mask_token_id == 19
    assert "diffusion_lm" not in cfg
    assert len(caught) == 1
    assert "deprecated" in str(caught[0].message)


def test_pydantic_alias_and_mapping_validate_under_one_field():
    cfg = DiffusionArgs.model_validate({"diffusion_lm": {"mask_token_id": 23}})
    plugin_cfg = DictDefault(diffusion_lm=DiffusionLMConfig(mask_token_id=29))

    DiffusionPlugin().register(plugin_cfg)

    assert cfg.diffusion.mask_token_id == 23
    assert cfg.diffusion.from_causal_lm is False
    assert "diffusion_lm" not in cfg.model_dump()
    assert plugin_cfg.diffusion.mask_token_id == 29
    assert plugin_cfg.diffusion.from_causal_lm is False


def test_both_spellings_rejected_even_if_blocks_match():
    values = {"diffusion": {}, "diffusion_lm": {}}

    with pytest.raises(ValueError, match="only one of"):
        DiffusionPlugin().register(DictDefault(values))
    with pytest.raises(ValidationError, match="only one of"):
        DiffusionArgs.model_validate(values)


def test_explicit_native_mode_on_canonical_field():
    cfg = DictDefault(diffusion={"from_causal_lm": False})

    DiffusionPlugin().register(cfg)

    assert cfg.diffusion.from_causal_lm is False
    assert get_diffusion_config(cfg) is cfg.diffusion


def test_mask_resolution_updates_only_canonical_block():
    tokenizer = SimpleNamespace(
        vocab_size=32,
        unk_token_id=0,
        all_special_tokens=["<mask>"],
        additional_special_tokens=[],
        convert_tokens_to_ids=lambda value: 7 if value == "<mask>" else 0,
    )
    cfg = DictDefault(diffusion={"mask_token_str": "<mask>"})
    DiffusionPlugin().register(cfg)

    assert resolve_mask_token_id(tokenizer, cfg, allow_add=False) == 7
    assert cfg.diffusion.mask_token_id == 7
    assert "diffusion_lm" not in cfg


def test_equal_legacy_warmup_ratio_normalizes_fractional_steps():
    cfg = DictDefault(diffusion={}, warmup_steps=0.1, warmup_ratio=0.1)

    DiffusionPlugin().register(cfg)

    assert cfg.warmup_steps is None
    assert cfg.warmup_ratio == 0.1


def test_conflicting_legacy_warmup_ratio_is_rejected():
    cfg = DictDefault(diffusion={}, warmup_steps=0.1, warmup_ratio=0.2)

    with pytest.raises(ValueError, match="fractional warmup_steps conflicts"):
        DiffusionPlugin().register(cfg)


def test_integer_legacy_warmup_steps_remain_unchanged():
    cfg = DictDefault(diffusion={}, warmup_steps=10)

    DiffusionPlugin().register(cfg)

    assert cfg.warmup_steps == 10
    assert cfg.warmup_ratio is None


def test_native_settings_do_not_use_legacy_normalization():
    cfg = DictDefault(
        diffusion={"from_causal_lm": False},
        warmup_steps=0.1,
        save_strategy="best",
        metric_for_best_model=None,
    )

    DiffusionPlugin().register(cfg)

    assert cfg.warmup_steps == 0.1
    assert cfg.warmup_ratio is None
    assert cfg.metric_for_best_model is None
