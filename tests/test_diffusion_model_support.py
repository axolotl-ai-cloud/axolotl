"""Focused contracts for lightweight diffusion model-support declarations."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ModelFamilyTemplate,
    ModelProfile,
    ModelStrategies,
    ModelStrategyOverrides,
    ModelSupport,
    ObjectiveReduction,
    TimeWeighting,
    resolve_model_support,
)
from axolotl.model_support.native_adapters import validate_native_diffusion_lora
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.diffusion import DiffusionConfig, DiffusionLMConfig


def _dream_spec() -> DiffusionSpec:
    return DiffusionSpec(
        noise=DiffusionNoise.ABSORBING,
        layout=DiffusionLayout.FULL_SEQUENCE,
        logit_alignment=LogitAlignment.SHIFTED,
        first_position_alignment=FirstPositionAlignment.DUPLICATE_FIRST,
        self_conditioning=False,
        max_canvas=None,
        max_context=2048,
        eos_handling=EosHandling.TREAT_EOS_AS_ONE,
        mask_token_policy=MaskTokenPolicy.MODEL,
        default_time_weighting=TimeWeighting.INV_T,
        objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
        generation_adapter=GenerationAdapter.DREAM,
    )


def test_diffusion_spec_is_manifest_serializable_and_validates_noise_policy():
    spec = _dream_spec()

    assert spec.to_dict() == {
        "noise": "absorbing",
        "layout": "full_sequence",
        "logit_alignment": "shifted",
        "first_position_alignment": "duplicate_first",
        "self_conditioning": False,
        "max_canvas": None,
        "max_context": 2048,
        "eos_handling": "treat_eos_as_one",
        "mask_token_policy": "model",
        "default_time_weighting": "inv_t",
        "objective_reduction": "masked_token_mean",
        "generation_adapter": "dream",
        "time_floor": 0.0,
        "reduction_scope": "microbatch",
    }
    with pytest.raises(ValueError, match="uniform diffusion"):
        DiffusionSpec(
            noise=DiffusionNoise.UNIFORM,
            layout=DiffusionLayout.ENCODER_CANVAS,
            logit_alignment=LogitAlignment.ALIGNED,
            first_position_alignment=FirstPositionAlignment.REQUIRES_PREDECESSOR,
            self_conditioning=True,
            max_canvas=256,
            max_context=None,
            eos_handling=EosHandling.INDEPENDENT,
            mask_token_policy=MaskTokenPolicy.MODEL,
            default_time_weighting=TimeWeighting.NONE,
            objective_reduction=ObjectiveReduction.SUPERVISED_TOKEN_MEAN,
            generation_adapter=GenerationAdapter.ENCODER_CANVAS,
        )


def test_profile_propagates_immutable_diffusion_facts_and_strategy_none_override():
    class FamilyTrainer:
        pass

    class FamilyCollator:
        pass

    family = ModelFamilyTemplate(
        name="diffusion_profile_test",
        strategies=ModelStrategies(
            trainer_cls=lambda: FamilyTrainer,
            collator_cls=lambda: FamilyCollator,
        ),
    )

    class Inheriting(ModelSupport):
        model_types = ("diffusion_profile_inheriting",)
        profile = ModelProfile(family=family, diffusion=_dream_spec())

    class Clearing(ModelSupport):
        model_types = ("diffusion_profile_clearing",)
        profile = ModelProfile(
            family=family,
            diffusion=_dream_spec(),
            strategies=ModelStrategyOverrides(trainer_cls=None),
        )

    inherited = resolve_model_support(Inheriting())
    cleared = resolve_model_support(Clearing())

    assert inherited.diffusion == _dream_spec()
    assert Inheriting.diffusion == _dream_spec()
    assert inherited.strategies.trainer_cls is not None
    assert inherited.strategies.trainer_cls() is FamilyTrainer
    assert inherited.strategies.collator_cls is not None
    assert inherited.strategies.collator_cls() is FamilyCollator
    assert cleared.strategies.trainer_cls is None
    assert cleared.strategies.collator_cls is not None


def test_diffusion_family_providers_are_lazy_in_a_fresh_process():
    code = """
import sys
from axolotl.model_support import DIFFUSION_LM
assert 'axolotl.core.trainers' not in sys.modules
assert DIFFUSION_LM.strategies.trainer_cls is not None
assert DIFFUSION_LM.strategies.collator_cls is not None
assert 'axolotl.core.trainers' not in sys.modules
"""
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_diffusion_descriptor_discovery_keeps_trainers_and_native_models_unloaded():
    code = """
import sys
from axolotl.model_support import get_model_support
gemma = get_model_support('diffusion_gemma')
dream = get_model_support('Dream')
assert gemma is not None and gemma.diffusion is not None
assert dream is not None and dream.diffusion is not None
assert 'axolotl.core.trainers' not in sys.modules
assert 'axolotl.core.trainers.diffusion_lm.trainer' not in sys.modules
assert 'transformers.models.diffusion_gemma.modeling_diffusion_gemma' not in sys.modules
assert 'axolotl.model_support.dream.compat' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr


def test_canonical_native_options_do_not_change_legacy_schema_or_imply_resize():
    assert "canvas_width" not in DiffusionConfig.model_fields
    cfg = DiffusionLMConfig(
        canvas_width=128,
        t_eps=0.0,
        eos_tail="visible_supervised",
        objective_reduction="example_mean",
        token_reweighting=True,
        alpha=0.3,
        gamma=1.7,
        self_conditioning={"p": 0.5, "train_module": False},
        unroll={"k_max": 2, "grad_through_steps": True},
    )

    assert cfg.allow_native_vocab_resize is False
    assert cfg.t_eps == 0.0
    assert cfg.self_conditioning is not None and cfg.self_conditioning.p == 0.5
    assert cfg.unroll is not None and cfg.unroll.k_max == 2
    assert cfg.eos_tail == "visible_supervised"
    assert cfg.objective_reduction == "example_mean"
    assert cfg.token_reweighting is True
    assert cfg.alpha == 0.3 and cfg.gamma == 1.7
    with pytest.raises(ValidationError):
        DiffusionLMConfig(allow_native_vocab_resize=True)


def test_native_diffusion_rejects_unsupported_attention_implementation():
    cfg = DictDefault(
        {
            "diffusion_lm": {"from_causal_lm": False},
            "adapter": "lora",
            "attn_implementation": "flash_attention_2",
            "lora_target_modules": ["encoder.language_model.decoder"],
        }
    )

    with pytest.raises(ValueError, match="attn_implementation values"):
        validate_native_diffusion_lora(cfg, model_name="DiffusionGemma")


def test_legacy_diffusion_conversion_allows_its_attention_implementation():
    cfg = DictDefault(
        {
            "diffusion_lm": {"from_causal_lm": True},
            "adapter": "lora",
            "attn_implementation": "flash_attention_2",
            "lora_target_modules": ["encoder.language_model.decoder"],
        }
    )

    validate_native_diffusion_lora(cfg, model_name="DiffusionGemma")
