"""Multimodal CPT prompt strategy + safety gate tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from transformers import AutoProcessor

from axolotl.prompt_strategies.multimodal_pretrain import (
    _INCOMPATIBLE_PROCESSOR_REASONS,
    ImageTokenSpec,
    MultimodalPretrainTokenizationStrategy,
    build_image_token_spec,
    check_processor_compatibility,
    load,
)
from axolotl.prompt_strategies.pretrain import PretrainTokenizer

from tests.hf_offline_utils import enable_hf_offline

_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"


@pytest.fixture(scope="module", name="smolvlm_processor")
@enable_hf_offline
def fixture_smolvlm_processor(
    download_smolvlm_500m_instruct_model,  # pylint: disable=unused-argument
):
    return AutoProcessor.from_pretrained(_SMOLVLM)


@pytest.fixture(scope="module", name="tiny_image_path")
def fixture_tiny_image_path(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("mm_pretrain_imgs")
    p = d / "dummy.png"
    arr = np.random.default_rng(0).integers(0, 255, (64, 64, 3)).astype("uint8")
    Image.fromarray(arr).save(p)
    return p


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


def test_build_image_token_spec_prefers_boi_token_over_expansion_token(
    smolvlm_processor,
):
    """Gemma-3-style autodetect: `boi_token` is preferred over `image_token`
    when they differ."""

    class _FakeGemma3Like:
        image_token = "<image>"
        boi_token = "<fake_token_around_image>"
        tokenizer = smolvlm_processor.tokenizer

    spec = build_image_token_spec(_FakeGemma3Like())
    assert spec.image_token == "<fake_token_around_image>"


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


# ---- MultimodalPretrainTokenizationStrategy --------------------------------


def _make_strategy(
    smolvlm_processor: Any,
    text_column: str = "text",
    image_column: str = "images",
) -> MultimodalPretrainTokenizationStrategy:
    spec = build_image_token_spec(smolvlm_processor)
    return MultimodalPretrainTokenizationStrategy(
        PretrainTokenizer(),
        smolvlm_processor.tokenizer,
        False,  # train_on_inputs
        2048,  # sequence_len
        text_column=text_column,
        image_column=image_column,
        image_base_dir=None,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        max_length=2048,
    )


def test_strategy_preserves_images_and_text(smolvlm_processor, tiny_image_path):
    strat = _make_strategy(smolvlm_processor)
    out = strat.tokenize_prompt(
        {
            "text": "<image>\nsample transcription text",
            "images": [str(tiny_image_path)],
        }
    )
    assert "input_ids" in out
    assert "images" in out and "_mm_text" in out
    assert len(out["input_ids"]) == 1
    assert len(out["images"]) == 1
    assert len(out["_mm_text"]) == 1
    assert out["images"][0] == [str(tiny_image_path)]
    assert out["_mm_text"][0].startswith("<image>")


def test_strategy_rejects_placeholder_count_mismatch(
    smolvlm_processor, tiny_image_path
):
    strat = _make_strategy(smolvlm_processor)
    with pytest.raises(ValueError, match="occurrence"):
        strat.tokenize_prompt(
            {
                "text": "<image><image>\ntwo placeholders one image",
                "images": [str(tiny_image_path)],
            }
        )


def test_strategy_rejects_row_exceeding_max_length(smolvlm_processor, tiny_image_path):
    spec = build_image_token_spec(smolvlm_processor)
    strat = MultimodalPretrainTokenizationStrategy(
        PretrainTokenizer(),
        smolvlm_processor.tokenizer,
        False,
        128,
        text_column="text",
        image_column="images",
        image_base_dir=None,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        max_length=128,
    )
    huge = "word " * 5000
    with pytest.raises(ValueError, match="exceeds sequence_len"):
        strat.tokenize_prompt(
            {
                "text": f"{spec.image_token} {huge}",
                "images": [str(tiny_image_path)],
            }
        )


def test_strategy_rejects_non_list_image_column(smolvlm_processor, tiny_image_path):
    strat = _make_strategy(smolvlm_processor)
    with pytest.raises(ValueError, match="list"):
        strat.tokenize_prompt(
            {
                "text": "<image>\nbad image field",
                "images": str(tiny_image_path),  # should be a list
            }
        )


@pytest.mark.parametrize("bad_value", ["", 0, False])
def test_strategy_rejects_falsy_non_none_image_column(smolvlm_processor, bad_value):
    """Falsy non-None image cells (e.g. "") are rejected, not coerced to []."""
    strat = _make_strategy(smolvlm_processor)
    with pytest.raises(ValueError, match="list"):
        strat.tokenize_prompt(
            {
                "text": "no placeholder, but bad images cell",
                "images": bad_value,
            }
        )


def test_strategy_treats_none_image_column_as_empty(smolvlm_processor):
    """images=None is the only falsy value treated as a text-only row."""
    strat = _make_strategy(smolvlm_processor)
    out = strat.tokenize_prompt(
        {
            "text": "plain text-only row, no placeholder",
            "images": None,
        }
    )
    assert out["images"][0] == []


# ---- load() factory --------------------------------------------------------


def test_load_requires_processor(smolvlm_processor):
    class _Cfg:
        train_on_inputs = False
        sequence_len = 2048

    with pytest.raises(ValueError, match="processor"):
        load(smolvlm_processor.tokenizer, _Cfg(), ds_cfg={}, processor=None)


def test_load_returns_strategy_with_spec(smolvlm_processor):
    class _Cfg:
        train_on_inputs = False
        sequence_len = 2048

    strat = load(
        smolvlm_processor.tokenizer,
        _Cfg(),
        ds_cfg={"text_column": "text", "image_column": "images"},
        processor=smolvlm_processor,
    )
    assert isinstance(strat, MultimodalPretrainTokenizationStrategy)
    assert hasattr(strat, "image_token_spec")
    assert strat.image_token_spec.image_token == "<image>"
