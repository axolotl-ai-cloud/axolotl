"""Reusable model-family templates for the vanilla loading path."""

from importlib.util import find_spec
from typing import Any

from .base import Unsupported
from .diffusion import is_native_diffusion
from .profile import (
    ModelFamilyTemplate,
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelStrategies,
)


def _causal_lm_auto_model_cls() -> type:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def _image_text_to_text_auto_model_cls() -> type:
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText


def _speech_seq2seq_auto_model_cls() -> type:
    from transformers import AutoModelForSpeechSeq2Seq

    return AutoModelForSpeechSeq2Seq


def _speech_seq2seq_collator_factory(cfg: Any, tokenizer: Any, is_eval: bool) -> Any:
    from axolotl.utils.collators.speech_seq2seq import build_speech_seq2seq_collator

    return build_speech_seq2seq_collator(cfg, tokenizer, is_eval)


def _validate_speech_seq2seq(context: ModelHookContext) -> None:
    cfg = context.cfg
    if find_spec("librosa") is None or find_spec("soundfile") is None:
        raise ImportError(
            "Speech seq2seq training requires the audio extra: "
            "uv pip install 'axolotl[audio]'"
        )
    if cfg.asr is None:
        from axolotl.utils.dict import DictDefault
        from axolotl.utils.schemas.asr import AsrConfig

        # dataset loading only skips datasets' torchcodec Audio decode when `asr` is set
        cfg.asr = DictDefault(AsrConfig().model_dump())
    # These read `input_ids`, which speech seq2seq batches do not carry.
    for key in ("sample_packing", "group_by_length", "generate_samples"):
        if cfg[key]:
            raise ValueError(
                f"{key} is not supported for speech seq2seq model_type={cfg.model_config_type}."
            )


def _diffusion_lm_trainer_cls(cfg: Any) -> type | None:
    if not is_native_diffusion(cfg):
        return None
    from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer

    return AxolotlDiffusionTrainer


def _diffusion_lm_collator_factory(cfg: Any, tokenizer: Any, is_eval: bool) -> Any:
    if not is_native_diffusion(cfg):
        return None
    from axolotl.core.trainers.diffusion_lm.collator import build_diffusion_collator

    return build_diffusion_collator(cfg, tokenizer, is_eval=is_eval)


VANILLA_CAUSAL_LM = ModelFamilyTemplate(
    name="vanilla_causal_lm",
    strategies=ModelStrategies(auto_model_cls=_causal_lm_auto_model_cls),
)

IMAGE_TEXT_TO_TEXT = ModelFamilyTemplate(
    name="image_text_to_text",
    is_multimodal=True,
    strategies=ModelStrategies(auto_model_cls=_image_text_to_text_auto_model_cls),
)

SPEECH_SEQ2SEQ = ModelFamilyTemplate(
    name="speech_seq2seq",
    is_multimodal=True,
    capabilities={
        "cut_cross_entropy": Unsupported(
            "CCE patches decoder-only CausalLM forwards, not speech seq2seq heads."
        ),
        "liger": Unsupported("Speech seq2seq models have no Liger kernel validation."),
        "lora_kernels": Unsupported(
            "The fused LoRA rewrite targets decoder-only attention and MLP blocks."
        ),
        "context_parallel": Unsupported(
            "Context parallelism shards token sequences, not audio features."
        ),
    },
    strategies=ModelStrategies(
        auto_model_cls=_speech_seq2seq_auto_model_cls,
        collator_factory=_speech_seq2seq_collator_factory,
    ),
    hooks=ModelHooks(
        by_phase={ModelHookPhase.CONFIGURE_RUN: (_validate_speech_seq2seq,)}
    ),
)

DIFFUSION_LM = ModelFamilyTemplate(
    name="diffusion_lm",
    capabilities={
        "cut_cross_entropy": Unsupported(
            "Native diffusion computes its objective from aligned canvas logits."
        ),
        "liger": Unsupported("Native diffusion has no Liger kernel validation."),
        "context_parallel": Unsupported(
            "Native diffusion document and canvas masks have no context-parallel validation."
        ),
        "expert_kernels": Unsupported(
            "Native diffusion has no expert-kernel training validation."
        ),
    },
    strategies=ModelStrategies(
        trainer_cls=_diffusion_lm_trainer_cls,
        collator_factory=_diffusion_lm_collator_factory,
    ),
)
