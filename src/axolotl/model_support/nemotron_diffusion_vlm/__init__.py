"""Native descriptor for Nemotron Labs Diffusion VLM."""

import logging
import re
from pathlib import Path

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
from axolotl.model_support.native_adapters import (
    _lora_layers,
    validate_native_diffusion_lora,
)
from axolotl.model_support.nemotron_diffusion import (
    lora_attention_cls_for,
    make_auto_model_class,
)
from axolotl.model_support.nemotron_diffusion.compat import VLM_VARIANT
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

LOG = logging.getLogger(__name__)

VISION_MODULE_PREFIXES = ("encoder.vision_tower.", "encoder.multi_modal_projector.")
_TEXT_LAYER_PATTERN = r"^encoder\.layers\.\d+\.(?:self_attn|mlp)\.(?:{names})$"
_PLAIN_TARGET = re.compile(r"^[A-Za-z0-9_]+$")


def _model_class() -> type:
    return make_auto_model_class(VLM_VARIANT)


def _matches_cfg(cfg) -> bool:
    source = getattr(cfg, "base_model", None)
    if not isinstance(source, str):
        return False
    if "nemotron-labs-diffusion-vlm" in source.lower():
        return True
    return (Path(source) / VLM_VARIANT.modeling_file).is_file()


def _lora_attention_cls(cfg) -> type:
    return lora_attention_cls_for(cfg, VLM_VARIANT)


def scope_lora_targets_to_text_layers(targets):
    """Plain module names also match the Pixtral tower's projections; anchor them."""
    if isinstance(targets, str) or not targets:
        return targets
    names = [str(target) for target in targets]
    if not all(_PLAIN_TARGET.fullmatch(name) for name in names):
        return targets
    return _TEXT_LAYER_PATTERN.format(names="|".join(sorted(set(names))))


def _is_vision_parameter(name: str) -> bool:
    bare = name.removeprefix("base_model.model.")
    return bare.startswith(VISION_MODULE_PREFIXES)


def _validate(context: ModelHookContext) -> None:
    cfg = context.cfg
    if not getattr(cfg, "trust_remote_code", False):
        raise ValueError(
            "Nemotron VLM native support requires trust_remote_code: true."
        )
    if getattr(cfg, "attn_implementation", None) is None:
        cfg.attn_implementation = "flex_attention"
    model_config = context.model_config
    if (
        model_config is not None
        and getattr(model_config, "dlm_paradigm", "bidirectional") != "bidirectional"
    ):
        raise ValueError(
            "Native Nemotron VLM support currently requires dlm_paradigm: bidirectional."
        )
    validate_native_diffusion_lora(
        cfg, model_name="Nemotron VLM", allow_4bit=True, allow_fsdp=True
    )
    scoped = scope_lora_targets_to_text_layers(cfg.lora_target_modules)
    if scoped is not cfg.lora_target_modules:
        LOG.info("Scoping Nemotron VLM LoRA targets to text layers: %s", scoped)
        cfg.lora_target_modules = scoped


def _freeze_vision(context: ModelHookContext) -> None:
    for name, parameter in context.model.named_parameters():
        if _is_vision_parameter(name):
            parameter.requires_grad_(False)


def _reject_vision_adapters(context: ModelHookContext) -> None:
    offenders = [
        name for name, _ in _lora_layers(context.model) if _is_vision_parameter(name)
    ]
    if offenders:
        raise ValueError(
            "Nemotron VLM LoRA attached to vision modules "
            f"{offenders[:3]}; restrict lora_target_modules to encoder.layers."
        )


@register_model_support
class NemotronDiffusionVLMSupport(ModelSupport):
    """Nemotron diffusion with a frozen Pixtral vision tower in bidirectional mode."""

    model_types = ("nemotron_labs_diffusion_vlm",)
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
            "vision_inputs": Supported(
                "Image features are scattered into the prompt's image-pad tokens."
            ),
            "diffusion_varlen": Unsupported(
                "Varlen parity is not yet measured on the VLM."
            ),
            "cut_cross_entropy": Unsupported(
                "The selected-logit CCE patch is not yet verified on the VLM."
            ),
            "fused_attn_kernel": Unsupported(
                "Native attention parity is not verified."
            ),
            "fsdp": Unsupported("FSDP2 parity is not yet measured on the VLM."),
            "quantized_lora": Unsupported(
                "4-bit LoRA with the vision stack skipped is not yet measured."
            ),
            "lora_kernels": Unsupported(
                "Fused LoRA kernel parity is not yet measured on the VLM."
            ),
        },
        strategies=ModelStrategyOverrides(
            auto_model_cls=_model_class, lora_attention_cls=_lora_attention_cls
        ),
        matchers=ModelMatchers(cfg=_matches_cfg),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.CONFIGURE_RUN: (_validate,),
                ModelHookPhase.AFTER_BASE_MODEL_BUILD: (_freeze_vision,),
                ModelHookPhase.AFTER_ADAPTER_LOAD: (_reject_vision_adapters,),
            }
        ),
    )
