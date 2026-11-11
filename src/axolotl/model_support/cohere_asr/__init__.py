"""Cohere Transcribe (cohere_asr) model support: Parakeet encoder + transformer decoder."""

import types

from torch import nn

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


def _eval_only_train(self: nn.Module, mode: bool = True) -> nn.Module:
    del mode
    return nn.Module.train(self, False)


def _freeze_batchnorm_stats(context: ModelHookContext) -> None:
    # Small, zero-padded fine-tuning batches would corrupt the running stats, and an
    # adapter does not save them; the trainer calls model.train() every step.
    if context.model is None:
        return
    for module in context.model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train = types.MethodType(_eval_only_train, module)
            module.eval()


@register_model_support
class CohereAsrSupport(ModelSupport):
    """Descriptor for Cohere Transcribe."""

    model_types = ("cohere_asr",)
    profile = ModelProfile(
        family=SPEECH_SEQ2SEQ,
        matchers=ModelMatchers(processor=_matches_processor),
        hooks=ModelHooks(
            by_phase={
                ModelHookPhase.CONFIGURE_RUN: (_validate,),
                ModelHookPhase.AFTER_ADAPTER_LOAD: (_freeze_batchnorm_stats,),
            }
        ),
    )
