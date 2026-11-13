"""Tests for the speech seq2seq (ASR) collator."""

import io
from unittest.mock import patch

import numpy as np
import pytest
import torch

librosa = pytest.importorskip("librosa")
soundfile = pytest.importorskip("soundfile")

from axolotl.utils.collators import speech_seq2seq  # noqa: E402
from axolotl.utils.collators.core import IGNORE_INDEX  # noqa: E402
from axolotl.utils.collators.speech_seq2seq import (  # noqa: E402
    build_speech_seq2seq_collator,
    load_audio,
)
from axolotl.utils.dict import DictDefault  # noqa: E402

from tests.hf_offline_utils import enable_hf_offline  # noqa: E402

MIRROR = "onnx-community/cohere-transcribe-03-2026-ONNX"
SR = 16000
MODEL_KEYS = {
    "input_features",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "labels",
}


def _tone(seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _row(text: str, seconds: float = 1.0, **extra) -> dict:
    return {
        "audio": {"array": _tone(seconds), "sampling_rate": SR},
        "text": text,
        **extra,
    }


@pytest.fixture(name="tokenizer", scope="module")
@enable_hf_offline
def fixture_tokenizer(download_cohere_transcribe_processor_fixture):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MIRROR)


@pytest.fixture(name="make_collator")
def fixture_make_collator(tokenizer):
    @enable_hf_offline
    def _make(asr=None, **cfg):
        cfg = DictDefault(
            {
                "base_model": MIRROR,
                "sequence_len": 448,
                "asr": DictDefault({"language": "en", **(asr or {})}),
                **cfg,
            }
        )
        return build_speech_seq2seq_collator(cfg, tokenizer, is_eval=False)

    return _make


def _prompt(collator, language="en"):
    return collator.processor.get_decoder_prompt_ids(
        language=language, punctuation=collator.punctuation
    )


def test_alignment_masking_and_padding(make_collator, tokenizer):
    collator = make_collator()
    batch = collator(
        [_row(" Hi. "), _row("A much longer transcript, with more words.")]
    )
    prompt = _prompt(collator)

    assert set(batch) == MODEL_KEYS
    assert batch["attention_mask"].shape == batch["input_features"].shape[:2]
    for ids, mask, labels in zip(
        batch["decoder_input_ids"].tolist(),
        batch["decoder_attention_mask"].tolist(),
        batch["labels"].tolist(),
        strict=True,
    ):
        length = sum(mask)
        assert mask == [1] * length + [0] * (len(ids) - length)
        assert ids[: len(prompt)] == prompt
        assert ids[length:] == [tokenizer.pad_token_id] * (len(ids) - length)
        assert labels[: len(prompt) - 1] == [IGNORE_INDEX] * (len(prompt) - 1)
        assert labels[length:] == [IGNORE_INDEX] * (len(ids) - length)
        assert labels[length - 1] == tokenizer.eos_token_id
        for i in range(len(prompt) - 1, length - 1):
            assert labels[i] == ids[i + 1]


def test_train_on_inputs(make_collator):
    collator = make_collator(train_on_inputs=True)
    labels = collator([_row("hello world")])["labels"][0].tolist()
    assert labels[: len(_prompt(collator)) - 1] == _prompt(collator)[1:]
    assert IGNORE_INDEX not in labels


@pytest.mark.parametrize("punctuation", [True, False])
def test_prefix_matches_inference_prompt(make_collator, punctuation):
    collator = make_collator(
        asr={"language_column": "lang", "punctuation": punctuation}
    )
    audio = _tone(1.0)
    batch = collator([_row("bonjour", lang="fr"), _row("hello", lang=None)])
    for row_idx, language in enumerate(["fr", "en"]):
        expected = collator.processor(
            audio, language=language, punctuation=punctuation, sampling_rate=SR
        )["decoder_input_ids"][0].tolist()
        prefix = batch["decoder_input_ids"][row_idx, : len(expected)].tolist()
        assert prefix == expected
    pnc = collator.tokenizer.convert_tokens_to_ids(
        "<|pnc|>" if punctuation else "<|nopnc|>"
    )
    assert pnc in prefix


def test_missing_language_raises(make_collator):
    collator = make_collator(asr={"language": None})
    with pytest.raises(ValueError, match="No language"):
        collator([_row("hello")])


def test_truncation_drops_eos(make_collator, tokenizer):
    collator = make_collator()
    max_len = len(_prompt(collator)) + 3
    collator.max_decoder_len = max_len
    batch = collator([_row("one two three four five six seven eight")])
    assert batch["decoder_input_ids"].shape[1] == max_len
    assert tokenizer.eos_token_id not in batch["labels"][0].tolist()


def test_out_of_range_clips_dropped_once_then_all_dropped_raises(make_collator):
    collator = make_collator()
    with patch.object(speech_seq2seq.LOG, "warning") as warning:
        for _ in range(2):
            batch = collator(
                [
                    _row("kept", seconds=35.0),
                    _row("long", seconds=36.0),
                    _row("tiny", seconds=0.01),
                ]
            )
            assert batch["input_features"].shape[0] == 1
            assert batch["labels"].shape[0] == 1
    warning.assert_called_once()
    with pytest.raises(ValueError, match="Every clip"):
        collator([_row("long", seconds=36.0)])


def test_features_cast_to_compute_dtype(make_collator):
    collator = make_collator(torch_dtype=torch.bfloat16)
    assert collator([_row("hello")])["input_features"].dtype == torch.bfloat16


@pytest.mark.parametrize("orig_sr", [8000, 44100])
@pytest.mark.parametrize("source", ["array", "stereo_array", "bytes", "path"])
def test_load_audio_sources(tmp_path, orig_sr, source):
    mono = _tone(0.5, orig_sr)
    stereo = np.stack([mono, mono], axis=1)
    if source in ("bytes", "path"):
        wav = tmp_path / "clip.wav"
        soundfile.write(wav, stereo, orig_sr, subtype="FLOAT")
        audio = (
            {"bytes": wav.read_bytes(), "path": None}
            if source == "bytes"
            else {"bytes": None, "path": str(wav)}
        )
    else:
        array = stereo.T if source == "stereo_array" else mono
        audio = {"array": array, "sampling_rate": orig_sr}

    decoded = load_audio(audio, SR)
    expected = librosa.resample(mono, orig_sr=orig_sr, target_sr=SR)
    assert decoded.ndim == 1
    np.testing.assert_allclose(decoded, expected, atol=1e-5)


@enable_hf_offline
def test_dataset_load_skips_decoding_and_filters_clip_lengths(tokenizer):
    from datasets import Audio, Dataset

    from axolotl.utils.data.sft import _load_and_process_single_dataset

    clips = []
    for seconds in (0.1, 0.01):
        buffer = io.BytesIO()
        soundfile.write(buffer, _tone(seconds), SR, format="WAV")
        clips.append({"bytes": buffer.getvalue(), "path": None})
    dataset = Dataset.from_dict({"audio": clips, "text": ["hi", "tiny"]}).cast_column(
        "audio", Audio()
    )
    cfg = DictDefault(
        base_model=MIRROR,
        skip_prepare_dataset=True,
        asr=DictDefault(audio_column="audio", language="en"),
    )
    with patch("axolotl.utils.data.sft.load_dataset_with_config", return_value=dataset):
        loaded, _ = _load_and_process_single_dataset(
            DictDefault(path="unused"), cfg, tokenizer, "train", seed=42
        )
    assert loaded.features["audio"].decode is False
    assert loaded["text"] == ["hi"]
    assert loaded[0]["audio"]["bytes"] == clips[0]["bytes"]


@enable_hf_offline
@pytest.mark.parametrize(
    "asr, match",
    [
        ({"text_column": "transcript"}, "no column"),
        ({"language_column": "lng"}, "no column"),
        ({"language": "english"}, "Unsupported language"),
        ({"language_column": "lang"}, "Unsupported language: 'English'"),
    ],
)
def test_dataset_load_rejects_bad_asr_columns_and_languages(tokenizer, asr, match):
    from datasets import Dataset

    from axolotl.utils.collators.speech_seq2seq import filter_audio_lengths

    rows = [_row("hi", lang="en"), _row("hello", lang="English")]
    cfg = DictDefault(
        base_model=MIRROR, asr=DictDefault({"language": "en", **asr}), sequence_len=448
    )
    with pytest.raises(ValueError, match=match):
        filter_audio_lengths(Dataset.from_list(rows), cfg, tokenizer)
