"""Collator for speech seq2seq (ASR) models: raw audio + transcript to encoder
features and teacher-forced decoder targets."""

import importlib
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, ProcessorMixin

from axolotl.utils.collators.core import IGNORE_INDEX
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PROCESSORS: dict[tuple[str, str | None], ProcessorMixin] = {}


def _import_audio_lib(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as err:
        raise ImportError(
            f"{name} is required for speech training; install the audio extra: "
            "uv pip install 'axolotl[audio]'"
        ) from err


def load_audio(audio: Any, sampling_rate: int) -> np.ndarray:
    """Decode an undecoded `datasets.Audio` value, bytes, or path to mono float32 at `sampling_rate`."""
    if isinstance(audio, dict) and audio.get("array") is not None:
        array = np.asarray(audio["array"], dtype=np.float32)
        orig_sr = audio["sampling_rate"]
        if array.ndim > 1:
            array = array.mean(axis=int(np.argmin(array.shape)))
    else:
        soundfile = _import_audio_lib("soundfile")
        if isinstance(audio, dict):
            audio = audio.get("bytes") or audio["path"]
        if isinstance(audio, bytes):
            audio = io.BytesIO(audio)
        array, orig_sr = soundfile.read(audio, dtype="float32", always_2d=True)
        array = array.mean(axis=1)
    if orig_sr != sampling_rate:
        librosa = _import_audio_lib("librosa")
        array = librosa.resample(array, orig_sr=orig_sr, target_sr=sampling_rate)
    return array


@dataclass
class SpeechSeq2SeqCollator:
    """Builds `decoder_input_ids = prompt + text` and pre-shifted `labels = prompt[1:] + text + [eos]`;
    the model's loss does not shift labels."""

    processor: ProcessorMixin
    tokenizer: PreTrainedTokenizerBase
    audio_column: str = "audio"
    text_column: str = "text"
    language: str | None = None
    language_column: str | None = None
    punctuation: bool = True
    max_decoder_len: int | None = None
    train_on_inputs: bool = False
    features_dtype: torch.dtype = torch.float32
    _prompts: dict[str, list[int]] = field(default_factory=dict, init=False)
    _warned: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        if not hasattr(self.processor, "get_decoder_prompt_ids"):
            raise ValueError(
                f"{type(self.processor).__name__} has no get_decoder_prompt_ids; "
                "the speech seq2seq collator cannot build the decoder prompt."
            )
        if self.tokenizer.eos_token_id is None or self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define eos and pad tokens.")
        feature_extractor = self.processor.feature_extractor
        self.sampling_rate = feature_extractor.sampling_rate
        # longer clips get split into extra feature rows, desyncing them from the labels
        max_clip_s = getattr(feature_extractor, "max_audio_clip_s", None)
        self.max_audio_s = (
            None
            if max_clip_s is None
            else max_clip_s - getattr(feature_extractor, "overlap_chunk_second", 0.0)
        )

    def _warn_once(self, key: str, message: str):
        if key not in self._warned:
            self._warned.add(key)
            LOG.warning(message)

    def _prompt_ids(self, row: dict) -> list[int]:
        language = (
            row.get(self.language_column) if self.language_column else None
        ) or self.language
        if not language:
            raise ValueError(
                "No language for this row: set `asr.language` or provide a value in "
                "`asr.language_column`."
            )
        if language not in self._prompts:
            self._prompts[language] = list(
                self.processor.get_decoder_prompt_ids(
                    language=language, punctuation=self.punctuation
                )
            )
        return self._prompts[language]

    def _build_sequence(self, row: dict) -> tuple[list[int], list[int]]:
        prompt = self._prompt_ids(row)
        text_ids = self.tokenizer(
            row[self.text_column].strip(), add_special_tokens=False
        )["input_ids"]
        if self.max_decoder_len is not None:
            max_text_len = max(self.max_decoder_len - len(prompt), 0)
            if len(text_ids) > max_text_len:
                self._warn_once(
                    "truncate",
                    f"Truncating transcripts to {max_text_len} tokens to fit "
                    f"sequence_len={self.max_decoder_len} with the decoder prompt.",
                )
                text_ids = text_ids[:max_text_len]
        seq = prompt + text_ids + [self.tokenizer.eos_token_id]
        labels = seq[1:]
        if not self.train_on_inputs:
            num_masked = len(prompt) - 1
            labels = [IGNORE_INDEX] * num_masked + labels[num_masked:]
        return seq[:-1], labels

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        audios, rows = [], []
        for row in examples:
            audio = load_audio(row[self.audio_column], self.sampling_rate)
            if (
                self.max_audio_s is not None
                and len(audio) / self.sampling_rate > self.max_audio_s
            ):
                self._warn_once(
                    "long_clip",
                    f"Dropping clips longer than {self.max_audio_s:g}s; filter them "
                    "out of the dataset to avoid smaller batches.",
                )
                continue
            audios.append(audio)
            rows.append(row)
        if not rows:
            raise ValueError(
                f"Every clip in the batch is longer than {self.max_audio_s:g}s."
            )

        features = self.processor.feature_extractor(
            audios,
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        features.pop("audio_chunk_index", None)
        if features["input_features"].shape[0] != len(rows):
            raise ValueError(
                "The feature extractor returned a different number of rows than "
                "the batch; was a clip split into chunks?"
            )

        decoder_input_ids, labels = zip(
            *(self._build_sequence(row) for row in rows), strict=True
        )
        max_len = max(len(ids) for ids in decoder_input_ids)
        pad_id = self.tokenizer.pad_token_id

        def _pad(seqs, value):
            return torch.tensor([s + [value] * (max_len - len(s)) for s in seqs])

        return {
            "input_features": features["input_features"].to(self.features_dtype),
            "attention_mask": features["attention_mask"],
            "decoder_input_ids": _pad(decoder_input_ids, pad_id),
            "decoder_attention_mask": _pad(
                [[1] * len(ids) for ids in decoder_input_ids], 0
            ),
            "labels": _pad(labels, IGNORE_INDEX),
        }


def build_speech_seq2seq_collator(
    cfg: DictDefault, tokenizer: PreTrainedTokenizerBase, is_eval: bool
) -> SpeechSeq2SeqCollator:
    from axolotl.loaders.processor import load_processor

    if not cfg.processor_config:
        cfg = DictDefault(
            {**cfg, "processor_config": cfg.base_model_config or cfg.base_model}
        )
    cache_key = (cfg.processor_config, cfg.revision_of_model)
    if cache_key not in _PROCESSORS:
        _PROCESSORS[cache_key] = load_processor(cfg, tokenizer)

    asr = cfg.asr or DictDefault()
    return SpeechSeq2SeqCollator(
        processor=_PROCESSORS[cache_key],
        tokenizer=tokenizer,
        audio_column=asr.audio_column or "audio",
        text_column=asr.text_column or "text",
        language=asr.language,
        language_column=asr.language_column,
        punctuation=asr.punctuation is not False,
        max_decoder_len=(cfg.eval_sequence_len if is_eval else None)
        or cfg.sequence_len,
        train_on_inputs=bool(cfg.train_on_inputs),
        features_dtype=cfg.torch_dtype or torch.float32,
    )
