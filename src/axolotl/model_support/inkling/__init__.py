"""Thinking Machines Inkling (`inkling_mm_model`) and Inkling 2 (`inkling2`) support."""

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT

_SHARED_CAPABILITIES = {
    "cut_cross_entropy": Unsupported(
        "The ConditionalGeneration forward scales or softcaps the logits and slices "
        "them to unpadded_vocab_size, which no CCE forward reproduces."
    ),
    "liger": Unsupported("No Liger patch is validated for Inkling."),
    "lora_kernels": Unsupported(
        "The attention runs short convolutions and a relative position bias that "
        "the fused QKV/O LoRA forward drops."
    ),
    "context_parallel": Unsupported(
        "The short convolutions carry state across shard boundaries."
    ),
}


def _refuse_modelopt_checkpoint(context: ModelHookContext) -> None:
    qcfg = getattr(context.model_config, "quantization_config", None) or {}
    if not isinstance(qcfg, dict):
        qcfg = qcfg.to_dict()
    if qcfg.get("quant_method") == "modelopt":
        raise ValueError(
            "This Inkling 2 checkpoint is quantized with modelopt (NVFP4 + MXFP8), "
            "which transformers cannot load. Use thinkingmachines/Inkling-2-Small-MXFP8."
        )


@register_model_support
class InklingSupport(ModelSupport):
    """Descriptor for Inkling (`inkling_mm_model`) text/image/audio MoE checkpoints."""

    model_types = ("inkling_mm_model",)
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities=_SHARED_CAPABILITIES,
    )


@register_model_support
class Inkling2Support(ModelSupport):
    """Descriptor for Inkling 2 (`inkling2`) text/image MoE checkpoints."""

    model_types = ("inkling2",)
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities={
            **_SHARED_CAPABILITIES,
            "expert_kernels": Unsupported(
                "The experts use a softcapped SwiGLU that the fused expert kernels "
                "would compute as plain SwiGLU."
            ),
        },
        hooks=ModelHooks(
            by_phase={ModelHookPhase.CONFIGURE_RUN: (_refuse_modelopt_checkpoint,)}
        ),
    )
