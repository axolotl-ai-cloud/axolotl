"""Cohere Transcribe (cohere_asr) model support: Parakeet encoder + transformer decoder."""

from axolotl.model_support.base import ModelSupport
from axolotl.model_support.profile import (
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
)
from axolotl.model_support.registry import register_model_support
from axolotl.model_support.templates import SPEECH_SEQ2SEQ
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# `get_decoder_prompt_ids` is 10 single tokens for every language/punctuation choice
_DECODER_PROMPT_LEN = 10


def _matches_processor(processor) -> bool:
    try:
        from transformers.models.cohere_asr import CohereAsrProcessor
    except ImportError:
        return False
    return isinstance(processor, CohereAsrProcessor)


def _validate(context: ModelHookContext) -> None:
    cfg = context.cfg
    if cfg.attn_implementation not in (None, "sdpa", "eager"):
        raise ValueError(
            f"attn_implementation: {cfg.attn_implementation} is not supported for "
            "cohere_asr; the Parakeet encoder supports only sdpa or eager."
        )
    max_positions = getattr(context.model_config, "max_position_embeddings", None)
    for key in ("sequence_len", "eval_sequence_len"):
        if max_positions and cfg[key] is not None and cfg[key] > max_positions:
            raise ValueError(
                f"{key} ({cfg[key]}) exceeds the cohere_asr decoder's "
                f"max_position_embeddings ({max_positions})."
            )
        if cfg[key] is not None and cfg[key] <= _DECODER_PROMPT_LEN:
            raise ValueError(
                f"{key} ({cfg[key]}) must exceed the cohere_asr decoder prompt "
                f"length ({_DECODER_PROMPT_LEN}) to leave room for transcript tokens."
            )
    if cfg.trust_remote_code:
        LOG.warning(
            "cohere_asr is native in transformers; trust_remote_code: true loads the "
            "hub's remote code instead, which axolotl does not validate."
        )


@register_model_support
class CohereAsrSupport(ModelSupport):
    """Descriptor for Cohere Transcribe."""

    model_types = ("cohere_asr",)

    @staticmethod
    def resolve_lora_merge_method(cfg, requested: str) -> str:
        """Merge through from_pretrained, which renames the hub's NeMo-style keys."""
        del cfg
        if requested == "memory_efficient":
            raise ValueError(
                "cohere_asr requires merge_method: legacy; the hub checkpoint uses "
                "NeMo key names that memory_efficient merging cannot match."
            )
        return "legacy"

    profile = ModelProfile(
        family=SPEECH_SEQ2SEQ,
        matchers=ModelMatchers(processor=_matches_processor),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.CONFIGURE_RUN: (_validate,),
            }
        ),
    )
