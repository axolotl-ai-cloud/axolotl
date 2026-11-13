"""Collator for speech seq2seq (ASR) models: raw audio + transcript to encoder
features and teacher-forced decoder targets."""

import importlib
import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from datasets import Dataset, IterableDataset
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


def audio_num_samples(audio: Any, sampling_rate: int) -> int:
    """Length of `audio` at `sampling_rate`, read from the file header when undecoded."""
    if isinstance(audio, dict) and audio.get("array") is not None:
        num_samples, orig_sr = max(np.shape(audio["array"])), audio["sampling_rate"]
    else:
        soundfile = _import_audio_lib("soundfile")
        if isinstance(audio, dict):
            audio = audio.get("bytes") or audio["path"]
        if isinstance(audio, bytes):
            audio = io.BytesIO(audio)
        info = soundfile.info(audio)
        num_samples, orig_sr = info.frames, info.samplerate
    # librosa.resample rounds the output length up
    return math.ceil(num_samples * sampling_rate / orig_sr)


def _within(num_samples: int, bounds: tuple[int, int | None]) -> bool:
    min_samples, max_samples = bounds
    return num_samples >= min_samples and (
        max_samples is None or num_samples <= max_samples
    )


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
        # fewer than two hops leaves a single frame, whose normalization divides by zero
        self.length_bounds = (
            2 * getattr(feature_extractor, "hop_length", 0),
            None if max_clip_s is None else round(max_clip_s * self.sampling_rate),
        )

    def describe_bounds(self) -> str:
        min_samples, max_samples = self.length_bounds
        upper = "inf" if max_samples is None else max_samples
        return f"[{min_samples}, {upper}] samples at {self.sampling_rate} Hz"

    def _warn_once(self, key: str, message: str):
        if key not in self._warned:
            self._warned.add(key)
            LOG.warning(message)

    def _prompt_ids(self, row: dict) -> list[int]:
        language = row.get(self.language_column) if self.language_column else None
        if language is None:
            language = self.language
        if language is None:
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
        eos = [self.tokenizer.eos_token_id]
        if self.max_decoder_len is not None:
            max_text_len = max(self.max_decoder_len - len(prompt), 0)
            if len(text_ids) > max_text_len:
                self._warn_once(
                    "truncate",
                    f"Truncating transcripts to {max_text_len} tokens to fit "
                    f"sequence_len={self.max_decoder_len} with the decoder prompt.",
                )
                # the audio continues past the cut, so do not teach the model to stop here
                text_ids, eos = text_ids[: max_text_len + 1], []
        seq = prompt + text_ids + eos
        labels = seq[1:]
        if not self.train_on_inputs:
            num_masked = len(prompt) - 1
            labels = [IGNORE_INDEX] * num_masked + labels[num_masked:]
        return seq[:-1], labels

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        audios, rows = [], []
        for row in examples:
            audio = load_audio(row[self.audio_column], self.sampling_rate)
            if not _within(len(audio), self.length_bounds):
                self._warn_once(
                    "clip_length",
                    f"Dropping clips outside {self.describe_bounds()}; filter them "
                    "out of the dataset to avoid smaller batches.",
                )
                continue
            audios.append(audio)
            rows.append(row)
        if not rows:
            raise ValueError(
                f"Every clip in the batch is outside {self.describe_bounds()}."
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


def _keep_clip(row: dict, column: str, sampling_rate: int, bounds: tuple) -> bool:
    return _within(audio_num_samples(row[column], sampling_rate), bounds)


def filter_audio_lengths(
    dataset: Dataset | IterableDataset,
    cfg: DictDefault,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset | IterableDataset:
    """Drop clips the collator would skip, so no batch can end up empty."""
    collator = build_speech_seq2seq_collator(cfg, tokenizer, is_eval=False)
    columns = dataset.column_names
    if columns is not None:
        wanted = (collator.audio_column, collator.text_column, collator.language_column)
        missing = [col for col in wanted if col and col not in columns]
        if missing:
            raise ValueError(
                f"Dataset has no column(s) {missing} named in `asr`; it has {columns}."
            )
    # surface unsupported language codes now rather than at the first batch
    if collator.language is not None:
        collator._prompt_ids({})
    if collator.language_column and isinstance(dataset, Dataset):
        for language in dataset.unique(collator.language_column):
            collator._prompt_ids({collator.language_column: language})

    fn_kwargs = {
        "column": collator.audio_column,
        "sampling_rate": collator.sampling_rate,
        "bounds": collator.length_bounds,
    }
    if isinstance(dataset, IterableDataset):
        return dataset.filter(_keep_clip, fn_kwargs=fn_kwargs)

    num_rows = len(dataset)
    dataset = dataset.filter(
        _keep_clip,
        fn_kwargs=fn_kwargs,
        num_proc=cfg.dataset_num_proc,
        desc="Dropping out-of-range audio clips",
    )
    if len(dataset) < num_rows:
        LOG.warning(
            f"Dropped {num_rows - len(dataset)} of {num_rows} clips outside "
            f"{collator.describe_bounds()}."
        )
    return dataset
