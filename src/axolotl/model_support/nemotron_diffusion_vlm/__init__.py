"""Native descriptor for Nemotron Labs Diffusion VLM."""

import logging
import os
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
# Decoder projection names Pixtral reuses; the second set exists only in vision.
_SHARED_TEXT_NAMES = frozenset(
    ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
)
_VISION_ONLY_NAMES = frozenset(("linear_1", "linear_2", "merging_layer", "patch_conv"))
DECODER_LAYER_CLS = "Ministral3DecoderLayer"


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


def scope_lora_targets_to_text_layers(targets, layers=None):
    """Plain names shared with the Pixtral tower are anchored to text layers; the
    other entries keep PEFT's list (suffix) semantics as regex alternatives."""
    if isinstance(targets, str) or not targets:
        return targets
    names = sorted({str(target) for target in targets})
    vision_only = [name for name in names if name in _VISION_ONLY_NAMES]
    if vision_only:
        raise ValueError(
            f"Nemotron VLM LoRA targets {vision_only} exist only in the frozen vision "
            "projector/tower; restrict lora_target_modules to encoder.layers."
        )
    shared = [name for name in names if name in _SHARED_TEXT_NAMES]
    if not shared:
        return targets
    others = [name for name in names if name not in _SHARED_TEXT_NAMES]
    if layers:
        prefix = r"encoder\.layers\.(?:{})\.".format("|".join(str(i) for i in layers))
    else:
        prefix = r"encoder\.layers\.\d+\."
    alternatives = [rf"^{prefix}(?:self_attn|mlp)\.(?:{'|'.join(shared)})$"]
    if others:
        escaped = "|".join(re.escape(name) for name in others)
        tail = rf"{prefix}(?:.*\.)?" if layers else r"(?:.*\.)?"
        alternatives.append(rf"^{tail}(?:{escaped})$")
    return "|".join(alternatives)


def _is_vision_parameter(name: str) -> bool:
    bare = name.removeprefix("base_model.model.")
    return bare.startswith(VISION_MODULE_PREFIXES)


def _validate_fsdp_wrap(cfg) -> None:
    """Text-only microbatches skip the frozen vision tower, so a vision or projector
    FSDP unit would all-gather on some ranks only and hang the step."""
    fsdp_config = getattr(cfg, "fsdp_config", None)
    if not fsdp_config:
        return
    policy = str(fsdp_config.get("auto_wrap_policy") or "").upper()
    if policy != "TRANSFORMER_BASED_WRAP":
        return
    names = fsdp_config.get("transformer_layer_cls_to_wrap")
    if not names:
        fsdp_config["transformer_layer_cls_to_wrap"] = DECODER_LAYER_CLS
        # prepare_optim_env exported the FSDP env vars before this hook runs.
        os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = DECODER_LAYER_CLS
        LOG.info(
            "Nemotron VLM FSDP: wrapping only %s (vision stays in the root unit).",
            DECODER_LAYER_CLS,
        )
        return
    rejected = [
        name.strip()
        for name in str(names).split(",")
        if name.strip() != DECODER_LAYER_CLS
    ]
    if rejected:
        raise ValueError(
            "Nemotron VLM FSDP transformer_layer_cls_to_wrap must name only "
            f"{DECODER_LAYER_CLS}; got {rejected}. Wrapping norm, vision or projector "
            "classes hangs ranks whose microbatch has no images."
        )


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
    _validate_fsdp_wrap(cfg)
    scoped = scope_lora_targets_to_text_layers(
        cfg.lora_target_modules, getattr(cfg, "peft_layers_to_transform", None)
    )
    if scoped is not cfg.lora_target_modules:
        LOG.info("Scoping Nemotron VLM LoRA targets to text layers: %s", scoped)
        cfg.lora_target_modules = scoped
        # PEFT rejects layers_to_transform with a str target; the regex carries them.
        cfg.peft_layers_to_transform = None
        cfg.peft_layers_pattern = None


def _freeze_vision(context: ModelHookContext) -> None:
    model = context.model
    assert model is not None
    for name, parameter in model.named_parameters():
        if _is_vision_parameter(name):
            parameter.requires_grad_(False)
    # The remote code doubles image features for its own complementary masking
    # in training mode; the trainer supplies the noised canvas itself.
    config = getattr(model, "config", None)
    if config is not None:
        config.complementary_mask = False


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
                "Only through the diffusion_decision plugin; plain diffusion_lm SFT "
                "on this model is text-only."
            ),
            "diffusion_varlen": Supported(
                "Native full-sequence varlen attention; measured on text-only "
                "batches, image batches are unmeasured."
            ),
            "cut_cross_entropy": Unsupported(
                "The selected-logit CCE patch is not yet verified on the VLM."
            ),
            "fused_attn_kernel": Unsupported(
                "Native attention parity is not verified."
            ),
            "fsdp": Supported(
                "FSDP2 LoRA and QLoRA match DDP step losses on text-only batches; "
                "image batches are unmeasured."
            ),
            "quantized_lora": Supported(
                "4-bit LoRA with the vision stack left unquantized, measured on "
                "text-only batches (image batches unmeasured); 8-bit is rejected."
            ),
            "lora_kernels": Supported(
                "Fused QKV/O/MLP kernels patch the native attention class; measured on text-only batches."
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
