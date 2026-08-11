"""
tests for the Muse Glimmer (Harmony-style) chat template via the chat_template strategy
"""

import pytest
from datasets import Dataset

from axolotl.prompt_strategies.chat_template import load
from axolotl.prompters import IGNORE_TOKEN_ID
from axolotl.utils.dict import DictDefault

EOT_TOKENS = ["<|eot|>", "<|eom|>"]


@pytest.fixture(name="muse_glimmer_dataset")
def fixture_muse_glimmer_dataset():
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "welcome"},
                    {"role": "user", "content": "bye"},
                    {"role": "assistant", "content": "goodbye"},
                ]
            }
        ]
    )


@pytest.fixture(name="muse_glimmer_reasoning_dataset")
def fixture_muse_glimmer_reasoning_dataset():
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "reasoning_content": "ponder",
                        "content": "welcome",
                    },
                ]
            }
        ]
    )


def _strategy(tokenizer, **cfg_extra):
    return load(
        tokenizer,
        DictDefault({"train_on_inputs": False, "sequence_len": 512, **cfg_extra}),
        DictDefault(
            {
                "chat_template": "tokenizer_default",
                "message_property_mappings": {"role": "role", "content": "content"},
                "field_messages": "messages",
            }
        ),
    )


def _trained_text(tokenizer, strategy, sample):
    res = strategy.tokenize_prompt(sample)
    kept = [
        token_id
        for token_id, label in zip(res["input_ids"], res["labels"], strict=True)
        if label != IGNORE_TOKEN_ID
    ]
    return tokenizer.decode(kept)


class TestMuseGlimmerChatTemplate:
    """Muse Glimmer ships its own Harmony-style template; axolotl uses it as-is."""

    def test_assistant_only_masking(self, muse_glimmer_tokenizer, muse_glimmer_dataset):
        strategy = _strategy(muse_glimmer_tokenizer, eot_tokens=EOT_TOKENS)
        trained = _trained_text(
            muse_glimmer_tokenizer, strategy, muse_glimmer_dataset[0]
        )

        assert trained == "welcome<|eot|>goodbye<|eot|>"

    def test_eot_tokens_required_to_train_the_terminator(
        self, muse_glimmer_tokenizer, muse_glimmer_dataset
    ):
        """The template closes turns with <|eot|>, which is not the tokenizer's
        eos_token (<|end_of_text|>). Without eot_tokens the terminator is never
        trained and the model never learns to stop."""
        strategy = _strategy(muse_glimmer_tokenizer)
        trained = _trained_text(
            muse_glimmer_tokenizer, strategy, muse_glimmer_dataset[0]
        )

        assert trained == "welcomegoodbye"
        assert "<|eot|>" not in trained

    def test_reasoning_content_is_trained(
        self, muse_glimmer_tokenizer, muse_glimmer_reasoning_dataset
    ):
        """reasoning_content renders as a `to=self` block closed by <|eom|>, and is
        assistant-authored, so it belongs in the loss alongside the content turn."""
        strategy = _strategy(muse_glimmer_tokenizer, eot_tokens=EOT_TOKENS)
        trained = _trained_text(
            muse_glimmer_tokenizer, strategy, muse_glimmer_reasoning_dataset[0]
        )

        assert "ponder<|eom|>" in trained
        assert trained.endswith("welcome<|eot|>")

    def test_prompt_text_is_masked(self, muse_glimmer_tokenizer):
        """The auto-injected system block (knowledge cutoff, reasoning strength, valid
        recipients) is prompt text and must not reach the loss."""
        sample = {
            "messages": [
                {"role": "user", "content": "USERTEXT"},
                {"role": "assistant", "content": "welcome"},
            ]
        }
        strategy = _strategy(muse_glimmer_tokenizer, eot_tokens=EOT_TOKENS)
        trained = _trained_text(muse_glimmer_tokenizer, strategy, sample)

        for prompt in (
            "USERTEXT",
            "Reasoning strength",
            "Valid recipients",
            "Knowledge cutoff",
        ):
            assert prompt not in trained
