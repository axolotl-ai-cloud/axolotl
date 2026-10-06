"""Native descriptor for Nemotron Labs Diffusion."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from axolotl.model_support.base import ModelSupport, Supported, Unsupported
from axolotl.model_support.diffusion import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
)
from axolotl.model_support.native_adapters import validate_native_diffusion_lora
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import DIFFUSION_LM

if TYPE_CHECKING:
    from .compat import NemotronVariant


def make_auto_model_class(variant: "NemotronVariant") -> type:
    from . import compat

    class AutoNemotronModel:
        def __new__(cls, config, **kwargs):
            return cls.from_config(config, **kwargs)

        @classmethod
        def from_pretrained(cls, model_source, **kwargs):
            model_class = compat.resolve_nemotron_model_class(
                model_source, revision=kwargs.get("revision"), variant=variant
            )
            resolved_revision = getattr(model_class, "_axolotl_resolved_revision", None)
            config = kwargs.get("config")
            if resolved_revision:
                if config is not None:
                    config._commit_hash = resolved_revision
                kwargs["revision"] = resolved_revision
            return model_class.from_pretrained(model_source, **kwargs)

        @classmethod
        def from_config(cls, config, **kwargs):
            source = getattr(config, "_name_or_path", None)
            if not source:
                raise ValueError(
                    "Nemotron from_config requires config._name_or_path for native source resolution."
                )
            kwargs.pop("trust_remote_code", None)
            model_class = compat.resolve_nemotron_model_class(
                source, revision=getattr(config, "_commit_hash", None), variant=variant
            )
            resolved_revision = getattr(model_class, "_axolotl_resolved_revision", None)
            if resolved_revision:
                config._commit_hash = resolved_revision
            return model_class._from_config(config, **kwargs)

    return AutoNemotronModel


def _model_class() -> type:
    from .compat import LM_VARIANT

    return make_auto_model_class(LM_VARIANT)


def _matches_cfg(cfg) -> bool:
    from .compat import LM_VARIANT, VLM_VARIANT

    source = getattr(cfg, "base_model", None)
    if not isinstance(source, str):
        return False
    lowered = source.lower()
    if "nemotron-labs-diffusion" in lowered:
        return "nemotron-labs-diffusion-vlm" not in lowered
    return (Path(source) / LM_VARIANT.modeling_file).is_file() and not (
        Path(source) / VLM_VARIANT.modeling_file
    ).is_file()


def lora_attention_cls_for(cfg, variant: "NemotronVariant") -> type:
    from . import compat

    model_class = compat.resolve_nemotron_model_class(
        cfg.base_model,
        revision=getattr(cfg, "revision_of_model", None),
        variant=variant,
    )
    name = (
        variant.flex_attention_class
        if getattr(cfg, "attn_implementation", None) == "flex_attention"
        else "Ministral3Attention"
    )
    for klass in model_class.__mro__:
        module = sys.modules.get(klass.__module__)
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError(f"{variant.name} native source does not define {name}.")


def _lora_attention_cls(cfg) -> type:
    from .compat import LM_VARIANT

    return lora_attention_cls_for(cfg, LM_VARIANT)


def _before_model_build(context: ModelHookContext) -> None:
    if not getattr(context.cfg, "cut_cross_entropy", False):
        from .cut_cross_entropy import reset_pending_nemotron_cce_options

        reset_pending_nemotron_cce_options()
        return
    from cut_cross_entropy.transformers.patch import PATCH_FNS

    PATCH_FNS["nemotron_labs_diffusion"] = (
        "axolotl.model_support.nemotron_diffusion.cut_cross_entropy",
        "patch_nemotron",
    )


def _validate(context: ModelHookContext) -> None:
    if not getattr(context.cfg, "trust_remote_code", False):
        raise ValueError("Nemotron native support requires trust_remote_code: true.")
    if getattr(context.cfg, "attn_implementation", None) is None:
        context.cfg.attn_implementation = "flex_attention"
    model_config = context.model_config
    if (
        model_config is not None
        and getattr(model_config, "dlm_paradigm", "bidirectional") != "bidirectional"
    ):
        raise ValueError(
            "Native Nemotron support currently requires dlm_paradigm: bidirectional."
        )
    validate_native_diffusion_lora(
        context.cfg, model_name="Nemotron", allow_4bit=True, allow_fsdp=True
    )


@register_model_support
class NemotronDiffusionSupport(ModelSupport):
    """Native absorbing-mask diffusion model in bidirectional mode."""

    model_types = ("nemotron_labs_diffusion",)
    profile = ModelProfile(
        family=DIFFUSION_LM,
        diffusion=DiffusionSpec(
            noise=DiffusionNoise.ABSORBING,
            layout=DiffusionLayout.FULL_SEQUENCE,
            logit_alignment=LogitAlignment.ALIGNED,
            first_position_alignment=FirstPositionAlignment.REQUIRES_PREDECESSOR,
            self_conditioning=False,
            max_canvas=None,
            max_context=262144,
            eos_handling=EosHandling.INDEPENDENT,
            mask_token_policy=MaskTokenPolicy.MODEL,
            default_time_weighting=TimeWeighting.INV_T,
            objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
            time_floor=0.001,
            reduction_scope=ReductionScope.GLOBAL_WINDOW,
            generation_adapter=GenerationAdapter.FULL_SEQUENCE,
        ),
        capabilities={
            "diffusion_varlen": Supported(
                "Native full-sequence varlen attention is supported."
            ),
            "cut_cross_entropy": Supported(
                "Aligned full-sequence loss through the native diffusion head."
            ),
            "fused_attn_kernel": Unsupported(
                "Native attention parity is not verified."
            ),
            "fsdp": Supported("FSDP2 LoRA and QLoRA match DDP step losses."),
            "quantized_lora": Supported("4-bit LoRA; 8-bit is rejected."),
            "lora_kernels": Supported(
                "Fused QKV/O/MLP kernels patch the native attention class."
            ),
        },
        strategies=ModelStrategyOverrides(
            auto_model_cls=_model_class, lora_attention_cls=_lora_attention_cls
        ),
        matchers=ModelMatchers(cfg=_matches_cfg),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.CONFIGURE_RUN: (_validate,),
                ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,),
            }
        ),
    )
