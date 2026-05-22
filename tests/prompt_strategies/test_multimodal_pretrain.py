"""Multimodal CPT helpers + safety gate tests.

The non-streaming strategy class and ``load()`` factory are deferred to a
follow-on PR (along with the matching ``build_collator`` routing for
``datasets:`` MM CPT batches), so only the helper-level surface is exercised
here in v1.
"""

from __future__ import annotations

import pytest
from transformers import AutoProcessor

from axolotl.prompt_strategies.multimodal_pretrain import (
    _INCOMPATIBLE_PROCESSOR_REASONS,
    ImageTokenSpec,
    build_image_token_spec,
    check_processor_compatibility,
)

from tests.hf_offline_utils import enable_hf_offline

_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"


@pytest.fixture(scope="module", name="smolvlm_processor")
@enable_hf_offline
def fixture_smolvlm_processor(
    download_smolvlm_500m_instruct_model,  # pylint: disable=unused-argument
):
    return AutoProcessor.from_pretrained(_SMOLVLM)


# ---- build_image_token_spec ------------------------------------------------


def test_build_image_token_spec_autodetects_smolvlm(smolvlm_processor):
    spec = build_image_token_spec(smolvlm_processor)
    assert isinstance(spec, ImageTokenSpec)
    assert spec.image_token == "<image>"
    assert spec.image_token_id > 0
    assert spec.image_token_id in spec.image_family_token_ids


def test_build_image_token_spec_honors_override(smolvlm_processor):
    spec = build_image_token_spec(smolvlm_processor, override="<image>")
    assert spec.image_token == "<image>"


def test_build_image_token_spec_rejects_bad_override(smolvlm_processor):
    with pytest.raises(ValueError, match="not a registered special token"):
        build_image_token_spec(smolvlm_processor, override="<definitely-not-real>")


def test_build_image_token_spec_rejects_plain_word_override(smolvlm_processor):
    # Plain words BPE-tokenize but aren't placeholders.
    with pytest.raises(ValueError, match="not a registered special token"):
        build_image_token_spec(smolvlm_processor, override="image")


def test_build_image_token_spec_keeps_image_token_when_no_soft_token_in_name(
    smolvlm_processor,
):
    """Non-Gemma-3 processors: the boi-swap heuristic only fires when
    `image_token` name contains "soft_token" (Gemma-3 convention). Otherwise
    `image_token` IS the user-facing placeholder (Gemma-4 convention) and
    must not be silently replaced by `boi_token`."""
    tok = smolvlm_processor.tokenizer
    image_id = tok.convert_tokens_to_ids("<image>")
    boi_id = tok.convert_tokens_to_ids("<fake_token_around_image>")
    assert boi_id != image_id, (
        "fixture assumption broken: SmolVLM tokenizer should map these to distinct ids"
    )

    class _FakeGemma4Like:
        image_token = "<image>"  # no 'soft_token' in name → must not swap
        boi_token = "<fake_token_around_image>"
        tokenizer = tok

    spec = build_image_token_spec(_FakeGemma4Like())
    assert spec.image_token == "<image>"
    assert spec.image_token_id == image_id
    assert spec.image_token_id != boi_id


# ---- check_processor_compatibility (startup-time gate) ---------------------


@pytest.mark.parametrize("cls_name", list(_INCOMPATIBLE_PROCESSOR_REASONS.keys()))
def test_check_processor_compatibility_rejects_incompatible(cls_name):
    fake = type(cls_name, (), {})()
    with pytest.raises(ValueError) as exc:
        check_processor_compatibility(fake)
    assert cls_name in str(exc.value)
    assert _INCOMPATIBLE_PROCESSOR_REASONS[cls_name] in str(exc.value)


def test_check_processor_compatibility_rejects_subclass():
    # MRO-name fallback must catch user-defined subclasses.
    class BaseMllama:
        pass

    BaseMllama.__name__ = "MllamaProcessor"

    class CustomUserProcessor(BaseMllama):
        pass

    CustomUserProcessor.__name__ = "CustomUserProcessor"

    with pytest.raises(ValueError, match="MllamaProcessor"):
        check_processor_compatibility(CustomUserProcessor())


def test_check_processor_compatibility_accepts_supported(smolvlm_processor):
    check_processor_compatibility(smolvlm_processor)
