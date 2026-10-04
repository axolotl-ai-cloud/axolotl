"""Native descriptors for Nemotron Labs Diffusion text and vision checkpoints."""

from dataclasses import replace

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
    ModelProfile,
    ModelStrategyOverrides,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import DIFFUSION_LM


def _model_class(*, default_vlm: bool = False) -> type:
    from .compat import resolve_nemotron_model_class, resolve_nemotron_vlm_model_class

    def resolve(config, model_source, revision):
        if (
            default_vlm
            or getattr(config, "model_type", None) == "nemotron_labs_diffusion_vlm"
        ):
            return resolve_nemotron_vlm_model_class(model_source, revision=revision)
        return resolve_nemotron_model_class(model_source, revision=revision)

    class AutoNemotronModel:
        def __new__(cls, config, **kwargs):
            return cls.from_config(config, **kwargs)

        @classmethod
        def from_pretrained(cls, model_source, **kwargs):
            model_class = resolve(
                kwargs.get("config"), model_source, kwargs.get("revision")
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
            model_class = resolve(config, source, getattr(config, "_commit_hash", None))
            resolved_revision = getattr(model_class, "_axolotl_resolved_revision", None)
            if resolved_revision:
                config._commit_hash = resolved_revision
            return model_class._from_config(config, **kwargs)

    return AutoNemotronModel


def _processor_class() -> type:
    from .processing import NemotronVLMProcessor

    return NemotronVLMProcessor


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
    PATCH_FNS["nemotron_labs_diffusion_vlm"] = (
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
    model_type = getattr(model_config, "model_type", None) or getattr(
        context.cfg, "model_config_type", None
    )
    if (
        model_type == "nemotron_labs_diffusion_vlm"
        and not (context.inference or getattr(context.cfg, "inference", False))
        and not getattr(context.cfg, "merge_lora", False)
        and getattr(context.cfg, "decision", None) is None
    ):
        raise ValueError(
            "Nemotron VLM image training currently requires the decision plugin and "
            "a decision configuration; native image SFT collation is not implemented."
        )
    validate_native_diffusion_lora(context.cfg, model_name="Nemotron")


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
            "fsdp": Unsupported("Native Nemotron LoRA has no FSDP validation."),
            "quantized_lora": Unsupported(
                "Native Nemotron LoRA has no quantized validation."
            ),
        },
        strategies=ModelStrategyOverrides(auto_model_cls=_model_class),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.CONFIGURE_RUN: (_validate,),
                ModelHookPhase.BEFORE_MODEL_BUILD: (_before_model_build,),
            }
        ),
    )


@register_model_support
class NemotronDiffusionVLMSupport(NemotronDiffusionSupport):
    """Native image-conditioned Nemotron diffusion model."""

    model_types = ("nemotron_labs_diffusion_vlm",)
    profile = replace(
        NemotronDiffusionSupport.profile,
        is_multimodal=True,
        strategies=replace(
            NemotronDiffusionSupport.profile.strategies,
            auto_model_cls=lambda: _model_class(default_vlm=True),
            auto_processor_cls=_processor_class,
        ),
    )
