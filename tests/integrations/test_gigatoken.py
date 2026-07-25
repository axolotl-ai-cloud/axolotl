# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the gigatoken integration plugin."""

import sys
from types import ModuleType

import pytest

from axolotl.integrations.gigatoken import GigatokenPlugin
from axolotl.prompt_strategies.pretrain import (
    PretrainTokenizationStrategy,
    PretrainTokenizer,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.tokenization import get_fast_encoder, set_fast_encoder


class Encoder:
    """Stand-in that encodes like the tokenizer but without overflow support."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = 0

    def __call__(self, texts, **kwargs):
        self.calls += 1
        return self.tokenizer(texts)


def fake_gigatoken(monkeypatch, as_hf=Encoder):
    module = ModuleType("gigatoken")

    class Tokenizer:  # pylint: disable=too-few-public-methods
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def as_hf(self):
            return as_hf(self.tokenizer)

    module.Tokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "gigatoken", module)


class TestGigatokenPlugin:
    """Encoder attachment and opt-out."""

    def test_attaches_encoder(self, monkeypatch, tokenizer_huggyllama):
        fake_gigatoken(monkeypatch)
        GigatokenPlugin().post_tokenizer_load(
            DictDefault({"gigatoken": True}), tokenizer_huggyllama
        )
        assert isinstance(get_fast_encoder(tokenizer_huggyllama), Encoder)
        set_fast_encoder(tokenizer_huggyllama, None)

    def test_disabled(self, monkeypatch, tokenizer_huggyllama):
        fake_gigatoken(monkeypatch)
        GigatokenPlugin().post_tokenizer_load(
            DictDefault({"gigatoken": False}), tokenizer_huggyllama
        )
        assert get_fast_encoder(tokenizer_huggyllama) is None

    def test_unwrappable_tokenizer_raises(self, monkeypatch, tokenizer_huggyllama):
        def _raise(_tokenizer):
            raise RuntimeError("Byte remapping failed")

        fake_gigatoken(monkeypatch, _raise)
        with pytest.raises(RuntimeError, match="gigatoken: false"):
            GigatokenPlugin().post_tokenizer_load(
                DictDefault({"gigatoken": True}), tokenizer_huggyllama
            )


class TestPretrainEncoder:
    """The packed pretraining path uses the encoder without losing overflow."""

    @staticmethod
    def _tokenize(tokenizer, texts, max_length, encoder=None):
        strategy = PretrainTokenizationStrategy(
            PretrainTokenizer(),
            tokenizer,
            False,
            max_length,
            text_column="text",
            max_length=max_length,
        )
        set_fast_encoder(tokenizer, encoder)
        try:
            return strategy._tokenize(texts)  # pylint: disable=protected-access
        finally:
            set_fast_encoder(tokenizer, None)

    def test_encoder_matches_tokenizer(self, tokenizer_huggyllama):
        texts = ["short text", "another short text"]
        encoder = Encoder(tokenizer_huggyllama)

        expected = self._tokenize(tokenizer_huggyllama, texts, 512)
        actual = self._tokenize(tokenizer_huggyllama, texts, 512, encoder)

        assert encoder.calls == 1
        assert actual["input_ids"] == expected["input_ids"]
        assert actual["attention_mask"] == expected["attention_mask"]

    def test_overflow_falls_back_to_tokenizer(self, tokenizer_huggyllama):
        texts = ["hello world. " * 500]

        expected = self._tokenize(tokenizer_huggyllama, texts, 512)
        actual = self._tokenize(
            tokenizer_huggyllama, texts, 512, Encoder(tokenizer_huggyllama)
        )

        assert len(expected["input_ids"]) > 1
        assert actual["input_ids"] == expected["input_ids"]
