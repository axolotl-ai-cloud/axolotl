"""Thinking Machines Inkling (`inkling_mm_model`) and Inkling 2 (`inkling2`) support."""

from collections.abc import Mapping

from axolotl.model_support.base import ModelSupport, Unsupported
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import IMAGE_TEXT_TO_TEXT
from axolotl.utils.dict import DictDefault

_SHARED_CAPABILITIES = {
    "cut_cross_entropy": Unsupported(
        "The LM head forward scales or softcaps the logits and slices "
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


# the Inkling-Small tokenizer declares neither; ids match config.eos_token_id and
# Inkling 2's generation_config pad_token_id
_DEFAULT_SPECIAL_TOKENS = {
    "pad_token": "<|endoftext|>",  # nosec B105
    "eos_token": "<|content_model_end_sampling|>",  # nosec B105
}


def _default_special_tokens(context: ModelHookContext) -> None:
    cfg = context.cfg
    special_tokens = DictDefault(cfg.special_tokens or {})
    for name, token in _DEFAULT_SPECIAL_TOKENS.items():
        if special_tokens.get(name) is None:
            special_tokens[name] = token
    cfg.special_tokens = special_tokens


def _quant_method(context: ModelHookContext) -> str | None:
    config = context.model_config
    qcfg = (
        config.get("quantization_config")
        if isinstance(config, Mapping)
        else getattr(config, "quantization_config", None)
    )
    return (
        qcfg.get("quant_method")
        if isinstance(qcfg, Mapping)
        else getattr(qcfg, "quant_method", None)
    )


def _refuse_unloadable_checkpoint(context: ModelHookContext) -> None:
    quant_method = _quant_method(context)
    if quant_method == "modelopt":
        raise ValueError(
            "This Inkling 2 checkpoint is quantized with modelopt (NVFP4 + MXFP8), "
            "which transformers cannot load. Use thinkingmachines/Inkling-2-Small-MXFP8."
        )
    cfg = context.cfg
    if quant_method == "fp8" and (cfg.load_in_4bit or cfg.load_in_8bit):
        raise ValueError(
            "This Inkling 2 checkpoint stores its routed experts in MXFP8, and "
            "bitsandbytes cannot requantize FP8 weights. Remove load_in_4bit / "
            "load_in_8bit, or start from a bf16 checkpoint."
        )


@register_model_support
class InklingSupport(ModelSupport):
    """Descriptor for Inkling (`inkling_mm_model`) text/image/audio MoE checkpoints."""

    model_types = ("inkling_mm_model", "inkling_text")
    profile = ModelProfile(
        family=IMAGE_TEXT_TO_TEXT,
        capabilities=_SHARED_CAPABILITIES,
        hooks=ModelHooks(
            by_phase={ModelHookPhase.CONFIGURE_RUN: (_default_special_tokens,)}
        ),
    )


@register_model_support
class Inkling2Support(ModelSupport):
    """Descriptor for Inkling 2 (`inkling2`) text/image MoE checkpoints."""

    model_types = ("inkling2", "inkling2_text")
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
            by_phase={ModelHookPhase.CONFIGURE_RUN: (_refuse_unloadable_checkpoint,)}
        ),
    )
