"""
Differential fuzz and regression tests for Messages.tokenized.

Messages.tokenized tokenizes the concatenated content values in a single pass
and attributes tokens to their content item via char offsets. These tests lock
that behavior: the output must always equal the tokenization of the final
string, never lose tokens, and attribute labels per item weight.

Run: PYTHONPATH=src pytest tests/core/chat/test_messages_fuzz.py
"""

import os
import random

import pytest
from transformers import AddedToken, AutoTokenizer

from axolotl.core.chat.format.chatml import format_message
from axolotl.core.chat.messages import ChatFormattedChats, Messages

from tests.hf_offline_utils import enable_hf_offline  # noqa

TOKENIZER = "NousResearch/Meta-Llama-3-8B"
N_FUZZ = int(os.getenv("AXOLOTL_FUZZ_ITERATIONS", "200"))


@pytest.fixture(scope="session", name="llama_tokenizer")
@enable_hf_offline
def llama_tokenizer_fixture():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokenizer.add_special_tokens(
        {
            "eos_token": AddedToken(
                "<|im_end|>", rstrip=False, lstrip=False, normalized=False
            )
        }
    )
    tokenizer.add_tokens(
        [AddedToken("<|im_start|>", rstrip=False, lstrip=False, normalized=False)]
    )
    return tokenizer


def _naive_full_string_tokenized(messages, tokenizer, ignore_index=-100):
    """Reference: build the same string, tokenize once, attribute by offsets."""
    content_str = ""
    spans = []
    for msg_content in messages.content:
        if msg_content.type in ("text", "tool_call", "tool_response"):
            start = len(content_str)
            content_str += str(msg_content)
            spans.append((start, len(content_str), msg_content))
    tok_results = tokenizer(
        content_str, add_special_tokens=False, return_offsets_mapping=True
    )
    input_ids = tok_results["input_ids"]
    labels = []
    item_idx = 0
    for token_id, (_, end) in zip(
        input_ids, tok_results["offset_mapping"], strict=True
    ):
        while item_idx < len(spans) and end > spans[item_idx][1]:
            item_idx += 1
        if messages.weight and spans[item_idx][2].weight not in [0, 0.0]:
            labels.append(token_id)
        else:
            labels.append(ignore_index)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _rand_text(rng, vocab, max_words=12):
    return " ".join(rng.choice(vocab) for _ in range(rng.randint(1, max_words)))


def _rand_content(rng, vocab, allow_tool=True, allow_nontext=True):
    r = rng.random()
    if r < 0.55:
        return {"type": "text", "value": _rand_text(rng, vocab)}
    if allow_tool and r < 0.8:
        return {
            "type": "tool_call",
            "value": {
                "name": rng.choice(["get_date", "get_stock_price", "move"]),
                "arguments": {"symbol": rng.choice(["AAPL", "TSLA"])},
            },
        }
    if r < 0.92:
        return {
            "type": "tool_response",
            "value": {
                "name": "get_date",
                "content": {"date": f"2024-{rng.randint(1, 12):02d}-01"},
            },
        }
    if allow_nontext and r < 0.96:
        return {"type": "image", "value": "/tmp/fake.png"}
    if allow_nontext:
        return {"type": "audio", "value": "/tmp/fake.wav"}
    return {"type": "text", "value": _rand_text(rng, vocab)}


def _rand_conversation(rng, vocab, n_messages=4):
    conv = []
    roles = ["system", "user", "assistant", "tool", "assistant", "user", "assistant"]
    for i in range(rng.randint(2, n_messages)):
        msg = {"role": roles[i % len(roles)], "content": []}
        for _ in range(rng.randint(1, 5)):
            allow_tool = msg["role"] in ("assistant", "tool")
            msg["content"].append(_rand_content(rng, vocab, allow_tool=allow_tool))
        if rng.random() < 0.5:
            msg["weight"] = rng.choice([0, 1])
        conv.append(msg)
    return {"conversation": conv}


VOCAB = [
    "The",
    "stock",
    "price",
    "of",
    "Apple",
    "on",
    "September",
    "9",
    "2024",
    "is",
    "$123.45",
    "What",
    "today's",
    "weather",
    "in",
    "Paris",
    "?",
    "move",
    "to",
    "(0,",
    "1)",
    "celsius",
    "fahrenheit",
    "reflect",
    "answer",
    "thanks",
]


@pytest.fixture(scope="session", name="fuzz_rng")
def fuzz_rng_fixture():
    return random.Random(42)


class TestTokenizedFuzz:
    @pytest.mark.parametrize("trial", range(N_FUZZ))
    def test_matches_full_string_tokenization(self, llama_tokenizer, fuzz_rng, trial):
        chat = _rand_conversation(fuzz_rng, VOCAB)
        chat_obj = ChatFormattedChats(**chat, formatter=format_message)
        for message in chat_obj.conversation:
            got = message.tokenized(llama_tokenizer)
            expected = _naive_full_string_tokenized(message, llama_tokenizer)
            assert got == expected

    @pytest.mark.parametrize("trial", range(N_FUZZ))
    def test_token_count_matches_final_string(self, llama_tokenizer, fuzz_rng, trial):
        chat = _rand_conversation(fuzz_rng, VOCAB)
        chat_obj = ChatFormattedChats(**chat, formatter=format_message)
        for message in chat_obj.conversation:
            content_str = "".join(
                str(c)
                for c in message.content
                if c.type in ("text", "tool_call", "tool_response")
            )
            got = message.tokenized(llama_tokenizer)
            expected_ids = llama_tokenizer(content_str, add_special_tokens=False)[
                "input_ids"
            ]
            assert got["input_ids"] == expected_ids
            assert len(got["labels"]) == len(got["input_ids"])
            assert len(got["attention_mask"]) == len(got["input_ids"])
            assert got["attention_mask"] == [1] * len(got["input_ids"])


class TestTokenizedBoundaryMerges:
    def test_adjacent_json_merge_not_dropped(self, llama_tokenizer):
        # JSON payloads concatenated as `..."}}` + `{"name"...` trigger a BPE
        # merge across the content item boundary; the legacy incremental
        # tokenization froze tokens at item boundaries and could drop or
        # duplicate them here. The output must equal the final string's
        # tokenization.
        message = Messages(
            role="tool",
            content=[
                {
                    "type": "tool_response",
                    "value": {
                        "name": "get_date",
                        "content": {"date": "2024-12-01"},
                    },
                },
                {
                    "type": "tool_response",
                    "value": {
                        "name": "get_date",
                        "content": {"date": "2024-06-01"},
                    },
                },
            ],
        )
        content_str = str(message)
        expected = llama_tokenizer(content_str, add_special_tokens=False)["input_ids"]
        got = message.tokenized(llama_tokenizer)
        assert got["input_ids"] == expected
        assert len(got["input_ids"]) == len(got["labels"])

    def test_merge_across_weight_boundary(self, llama_tokenizer):
        # a token spanning a weight-0 item and a trainable item is attributed
        # to the item containing its end offset
        message = Messages(
            role="assistant",
            weight=1,
            content=[
                {"type": "text", "value": "move", "weight": 0},
                {"type": "text", "value": " to (0, 1)"},
            ],
        )
        got = message.tokenized(llama_tokenizer)
        expected_ids = llama_tokenizer("move to (0, 1)", add_special_tokens=False)[
            "input_ids"
        ]
        assert got["input_ids"] == expected_ids
        merge_start = len(
            llama_tokenizer("move", add_special_tokens=False)["input_ids"]
        )
        assert got["labels"][:merge_start] == [-100] * merge_start
        assert all(label != -100 for label in got["labels"][merge_start:])

    def test_single_content_item(self, llama_tokenizer):
        message = Messages(
            role="user",
            content=[{"type": "text", "value": "What is 2 + 2?"}],
        )
        expected = llama_tokenizer("What is 2 + 2?", add_special_tokens=False)[
            "input_ids"
        ]
        got = message.tokenized(llama_tokenizer)
        assert got["input_ids"] == expected
        assert got["labels"] == [-100] * len(expected)

    def test_nontext_only_message(self, llama_tokenizer):
        message = Messages(
            role="user",
            content=[
                {"type": "image", "value": "/tmp/a.png"},
                {"type": "audio", "value": "/tmp/b.wav"},
            ],
        )
        assert message.tokenized(llama_tokenizer) == {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
