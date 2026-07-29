"""
End-to-end check that KD online teaching works over a dataset rendered in a runtime mix
of reasoning-on / reasoning-off modes: the teacher scores exactly the ids each example
was rendered with, and the returned targets land on those ids.
"""

from unittest.mock import MagicMock

import orjson
import pytest
import requests
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from axolotl.datasets import wrap_dataset_for_tokenized_prompt
from axolotl.integrations.kd.collator_online_teacher import OnlineTeacherCollator
from axolotl.prompt_strategies.chat_template import load
from axolotl.utils.dict import DictDefault

TOP_K = 2
REASONING_TEMPLATE = (
    "{%- for m in messages %}"
    "{{ '<|im_start|> ' + m['role'] + ' ' }}"
    "{%- if m['role'] == 'assistant' and enable_thinking %}"
    "{{ '<think> ' + (m['reasoning_content'] | default('pondering')) + ' </think> ' }}"
    "{%- endif %}"
    "{{ m['content'] + ' <|im_end|> ' }}"
    "{%- endfor %}"
)
VOCAB_WORDS = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "user",
    "assistant",
    "hello",
    "world",
    "pondering",
    "answer",
    "[UNK]",
    "[PAD]",
]
THINK_OPEN_ID = VOCAB_WORDS.index("<think>")
MIX = [
    {"kwargs": {"enable_thinking": True}, "weight": 0.5},
    {"kwargs": {"enable_thinking": False}, "weight": 0.5},
]


@pytest.fixture(name="reasoning_tokenizer")
def reasoning_tokenizer_fixture():
    backend = Tokenizer(
        WordLevel({w: i for i, w in enumerate(VOCAB_WORDS)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
    )
    tokenizer.chat_template = REASONING_TEMPLATE
    return tokenizer


def mixed_dataset(tokenizer, size=8):
    strategy = load(
        tokenizer,
        DictDefault({"train_on_inputs": False, "sequence_len": 512, "seed": 42}),
        DictDefault(
            {
                "chat_template": "tokenizer_default",
                "field_messages": "messages",
                "roles_to_train": ["assistant"],
                "chat_template_kwargs_mix": MIX,
            }
        ),
    )
    raw = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
            for _ in range(size)
        ]
    )
    return wrap_dataset_for_tokenized_prompt(strategy, raw, keep_in_memory=True)


def teacher_for(request_payload):
    """
    Answer with a distribution whose top-1 token is the prompt token at that position,
    so the alignment of the returned targets is observable from the ids alone.
    """
    choices = []
    for prompt_ids in request_payload["prompt"]:
        rows = [None]
        for token_id in prompt_ids[1:]:
            rows.append(
                {
                    str(token_id): {"logprob": -0.1},
                    str(len(VOCAB_WORDS) + token_id): {"logprob": -2.0},
                }
            )
        choices.append({"prompt_logprobs": rows})
    return {"choices": choices}


def make_collator(tokenizer, requests_log):
    collator = OnlineTeacherCollator(
        tokenizer,
        kd_online_server_base_url="http://teacher:8000",
        kd_online_topk=TOP_K,
        kd_online_preflight=False,
    )

    def post(_endpoint, json=None, **_kwargs):
        requests_log.append(json)
        response = MagicMock(spec=requests.Response)
        response.status_code = 200
        response.content = orjson.dumps(teacher_for(json))
        response.raise_for_status.return_value = None
        return response

    collator.http_session.post = post
    return collator


def test_online_teacher_scores_each_examples_own_rendering(reasoning_tokenizer):
    dataset = mixed_dataset(reasoning_tokenizer)
    features = [
        {
            "input_ids": list(row["input_ids"]),
            "labels": list(row["labels"]),
            "attention_mask": list(row["attention_mask"]),
        }
        for row in dataset
    ]
    rendered = [feature["input_ids"] for feature in features]

    thinking = [THINK_OPEN_ID in ids for ids in rendered]
    assert any(thinking) and not all(thinking), "the batch should carry both renderings"

    requests_log = []
    batch = make_collator(reasoning_tokenizer, requests_log)(
        [dict(feature) for feature in features]
    )

    # the teacher was asked to score exactly what each example was rendered as
    assert len(requests_log) == 1
    assert requests_log[0]["prompt"] == rendered
    assert requests_log[0]["max_tokens"] == 0
    assert requests_log[0]["prompt_logprobs"] == TOP_K

    padded_len = batch["input_ids"].shape[1]
    assert batch["target_token_ids"].shape == (len(features), padded_len, TOP_K)
    assert batch["target_logprobs"].shape == (len(features), padded_len, TOP_K)
    assert torch.isfinite(batch["target_logprobs"]).all()

    for sample, (ids, labels) in enumerate(
        zip(rendered, [feature["labels"] for feature in features], strict=True)
    ):
        for position in range(len(ids) - 1):
            if labels[position + 1] == -100:
                continue
            assert (
                batch["target_token_ids"][sample][position][0].item()
                == ids[position + 1]
            ), f"sample {sample} row {position} is not aligned to its own rendering"
            assert batch["target_mask"][sample][position].tolist() == [1] * TOP_K

        # the final rendered position, and any right padding, carry no teacher target
        for position in range(len(ids) - 1, padded_len):
            assert batch["target_mask"][sample][position].tolist() == [0] * TOP_K


def test_packed_mixed_batch_keeps_per_sequence_targets(reasoning_tokenizer):
    dataset = mixed_dataset(reasoning_tokenizer)
    rows = [
        {
            "input_ids": list(row["input_ids"]),
            "labels": list(row["labels"]),
            "attention_mask": list(row["attention_mask"]),
        }
        for row in dataset
    ]
    thinking_row = next(r for r in rows if THINK_OPEN_ID in r["input_ids"])
    plain_row = next(r for r in rows if THINK_OPEN_ID not in r["input_ids"])
    sub_batch = [dict(thinking_row), dict(plain_row)]

    requests_log = []
    batch = make_collator(reasoning_tokenizer, requests_log)([sub_batch])

    assert requests_log[0]["prompt"] == [
        thinking_row["input_ids"],
        plain_row["input_ids"],
    ]

    offset = 0
    for row in (thinking_row, plain_row):
        ids, labels = row["input_ids"], row["labels"]
        for position in range(len(ids) - 1):
            if labels[position + 1] == -100:
                continue
            assert (
                batch["target_token_ids"][0][offset + position][0].item()
                == ids[position + 1]
            )
        assert batch["target_mask"][0][offset + len(ids) - 1].tolist() == [0] * TOP_K
        offset += len(ids)


def test_offline_kd_strategy_inherits_the_mix(reasoning_tokenizer):
    """The KD chat_template strategies are ChatTemplateStrategy subclasses."""
    from axolotl.integrations.kd.chat_template import load as load_kd

    strategy = load_kd(
        reasoning_tokenizer,
        DictDefault({"train_on_inputs": False, "sequence_len": 512, "seed": 42}),
        DictDefault(
            {
                "chat_template": "tokenizer_default",
                "field_messages": "messages",
                "roles_to_train": ["assistant"],
                "chat_template_kwargs_mix": MIX,
            }
        ),
    )

    assert strategy.supports_indices is True
    assert strategy.prompter.chat_template_kwargs_mix is not None
    assert strategy.prompter.chat_template_kwargs_mix.weights == [0.5, 0.5]
