"""
Tests for per-dataset chat-template kwargs and the deterministic reasoning-mode mix.
"""

import pytest
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from axolotl.datasets import wrap_dataset_for_tokenized_prompt
from axolotl.prompt_strategies.chat_template import load
from axolotl.utils.chat_templates import (
    ChatTemplateKwargsMix,
    build_chat_template_kwargs_mix,
)
from axolotl.utils.dict import DictDefault

# a minimal reasoning-capable template: the assistant turn carries a <think> span only
# when the render is asked for it, exactly like the Qwen3-family enable_thinking switch
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
    "system",
    "hello",
    "world",
    "pondering",
    "answer",
    "[UNK]",
    "[PAD]",
]
THINK_OPEN_ID = VOCAB_WORDS.index("<think>")


@pytest.fixture(name="reasoning_tokenizer")
def reasoning_tokenizer_fixture():
    """Hermetic tokenizer whose chat template has an enable_thinking conditional."""
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


def conversation(index=0):
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
    ]


def make_dataset(size=8):
    return Dataset.from_list([{"messages": conversation(i)} for i in range(size)])


def build_strategy(tokenizer, ds_overrides=None, cfg_overrides=None):
    cfg = DictDefault(
        {
            "train_on_inputs": False,
            "sequence_len": 512,
            "seed": 42,
            **(cfg_overrides or {}),
        }
    )
    ds_cfg = DictDefault(
        {
            "chat_template": "tokenizer_default",
            "field_messages": "messages",
            "roles_to_train": ["assistant"],
            **(ds_overrides or {}),
        }
    )
    return load(tokenizer, cfg, ds_cfg)


def thinking_flags(dataset):
    """Whether each tokenized row was rendered with the <think> span."""
    return [THINK_OPEN_ID in row["input_ids"] for row in dataset]


def tokenize(strategy, dataset, num_proc=None):
    return wrap_dataset_for_tokenized_prompt(
        strategy, dataset, process_count=num_proc, keep_in_memory=True
    )


MIX_30_70 = [
    {"kwargs": {"enable_thinking": True}, "weight": 0.3},
    {"kwargs": {"enable_thinking": False}, "weight": 0.7},
]


# --- mix helper -------------------------------------------------------------------


def test_weights_are_normalized():
    mix = ChatTemplateKwargsMix(MIX_30_70)

    assert mix.weights == pytest.approx([0.3, 0.7])
    assert len(mix) == 2


def test_weights_default_to_uniform():
    mix = ChatTemplateKwargsMix(
        [{"kwargs": {"enable_thinking": True}}, {"kwargs": {"enable_thinking": False}}]
    )

    assert mix.weights == pytest.approx([0.5, 0.5])


def test_missing_kwargs_means_template_default():
    mix = ChatTemplateKwargsMix([{"weight": 1.0}])

    assert mix.kwargs_for_index(0) == {}


@pytest.mark.parametrize(
    "entries, match",
    [
        ([], "at least one entry"),
        ([{"kwargs": {}, "weight": 0.0}], "sum to more than 0"),
        ([{"kwargs": {}, "weight": -1.0}], "non-negative"),
        ([{"kwargs": [], "weight": 1.0}], "kwargs must be a mapping"),
        ([{"kwargs": {}, "weight": "heavy"}], "weight must be a number"),
        (["enable_thinking"], "must be a mapping"),
    ],
)
def test_invalid_mixes_are_rejected(entries, match):
    with pytest.raises(ValueError, match=match):
        ChatTemplateKwargsMix(entries)


def test_build_returns_none_without_a_mix():
    assert build_chat_template_kwargs_mix(None) is None
    assert build_chat_template_kwargs_mix([]) is None


# --- determinism ------------------------------------------------------------------


def assignments(seed, count=512):
    mix = ChatTemplateKwargsMix(MIX_30_70, seed=seed)
    return [mix.kwargs_for_index(i)["enable_thinking"] for i in range(count)]


def test_assignment_is_reproducible_for_a_seed():
    assert assignments(42) == assignments(42)


def test_assignment_changes_with_the_seed():
    first, second = assignments(42), assignments(43)

    assert first != second
    # not merely a shifted copy: the two disagree on a substantial share of examples
    disagreement = sum(a != b for a, b in zip(first, second, strict=True)) / len(first)
    assert 0.1 < disagreement < 0.9


def test_assignment_does_not_depend_on_order_of_access():
    mix = ChatTemplateKwargsMix(MIX_30_70, seed=7)
    forward = [mix.kwargs_for_index(i) for i in range(64)]
    backward = [mix.kwargs_for_index(i) for i in reversed(range(64))]

    assert forward == list(reversed(backward))


def test_weighted_proportions_are_honored():
    sample = assignments(42, count=4000)
    share_on = sum(sample) / len(sample)

    # 4000 draws at p=0.3 has a standard error of ~0.0072; 5 sigma is ~0.036
    assert share_on == pytest.approx(0.3, abs=0.036)


def test_three_way_mix_proportions():
    mix = ChatTemplateKwargsMix(
        [
            {"kwargs": {"mode": "a"}, "weight": 1},
            {"kwargs": {"mode": "b"}, "weight": 2},
            {"kwargs": {"mode": "c"}, "weight": 1},
        ],
        seed=11,
    )
    modes = [mix.kwargs_for_index(i)["mode"] for i in range(4000)]

    assert modes.count("a") / 4000 == pytest.approx(0.25, abs=0.035)
    assert modes.count("b") / 4000 == pytest.approx(0.50, abs=0.04)
    assert modes.count("c") / 4000 == pytest.approx(0.25, abs=0.035)


# --- rendering --------------------------------------------------------------------


def test_mix_entries_render_differently(reasoning_tokenizer):
    strategy = build_strategy(
        reasoning_tokenizer, {"chat_template_kwargs_mix": MIX_30_70}
    )
    tokenized = tokenize(strategy, make_dataset(64))
    flags = thinking_flags(tokenized)

    assert any(flags) and not all(flags), "both renderings should appear"

    with_think = next(row for row, flag in zip(tokenized, flags, strict=True) if flag)
    without_think = next(
        row for row, flag in zip(tokenized, flags, strict=True) if not flag
    )
    assert with_think["input_ids"] != without_think["input_ids"]
    assert len(with_think["input_ids"]) > len(without_think["input_ids"])
    assert len(with_think["labels"]) == len(with_think["input_ids"])


def test_rendered_mix_matches_the_declared_assignment(reasoning_tokenizer):
    strategy = build_strategy(
        reasoning_tokenizer, {"chat_template_kwargs_mix": MIX_30_70}
    )
    tokenized = tokenize(strategy, make_dataset(64))

    mix = ChatTemplateKwargsMix(MIX_30_70, seed=42)
    expected = [mix.kwargs_for_index(i)["enable_thinking"] for i in range(64)]

    assert thinking_flags(tokenized) == expected


def test_tokenization_is_stable_across_num_proc(reasoning_tokenizer):
    dataset = make_dataset(64)
    strategy = build_strategy(
        reasoning_tokenizer, {"chat_template_kwargs_mix": MIX_30_70}
    )

    single = thinking_flags(tokenize(strategy, dataset, num_proc=1))
    two = thinking_flags(tokenize(strategy, dataset, num_proc=2))
    three = thinking_flags(tokenize(strategy, dataset, num_proc=3))

    assert single == two == three
    assert any(single) and not all(single)


def test_per_dataset_kwargs_without_a_mix(reasoning_tokenizer):
    on = build_strategy(
        reasoning_tokenizer, {"chat_template_kwargs": {"enable_thinking": True}}
    )
    off = build_strategy(
        reasoning_tokenizer, {"chat_template_kwargs": {"enable_thinking": False}}
    )

    assert on.supports_indices is False
    assert all(thinking_flags(tokenize(on, make_dataset(4))))
    assert not any(thinking_flags(tokenize(off, make_dataset(4))))


def test_dataset_kwargs_override_the_global_setting(reasoning_tokenizer):
    strategy = build_strategy(
        reasoning_tokenizer,
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert all(thinking_flags(tokenize(strategy, make_dataset(4))))


def test_global_kwargs_still_apply_without_a_dataset_override(reasoning_tokenizer):
    strategy = build_strategy(
        reasoning_tokenizer, None, {"chat_template_kwargs": {"enable_thinking": True}}
    )

    assert all(thinking_flags(tokenize(strategy, make_dataset(4))))


def test_mix_overrides_the_base_kwargs_per_example(reasoning_tokenizer):
    strategy = build_strategy(
        reasoning_tokenizer,
        {
            "chat_template_kwargs": {"enable_thinking": True},
            "chat_template_kwargs_mix": MIX_30_70,
        },
    )
    flags = thinking_flags(tokenize(strategy, make_dataset(64)))

    assert any(flags) and not all(flags)


def test_dataset_seed_overrides_the_run_seed(reasoning_tokenizer):
    default_seed = build_strategy(
        reasoning_tokenizer, {"chat_template_kwargs_mix": MIX_30_70}
    )
    other_seed = build_strategy(
        reasoning_tokenizer,
        {"chat_template_kwargs_mix": MIX_30_70, "chat_template_kwargs_mix_seed": 1234},
    )

    assert thinking_flags(tokenize(default_seed, make_dataset(64))) != thinking_flags(
        tokenize(other_seed, make_dataset(64))
    )


# --- eval isolation ---------------------------------------------------------------


def test_eval_dataset_does_not_inherit_a_training_mix(reasoning_tokenizer):
    """A separately configured (eval) dataset entry only renders how it declares."""
    train = build_strategy(reasoning_tokenizer, {"chat_template_kwargs_mix": MIX_30_70})
    evaluation = build_strategy(
        reasoning_tokenizer, {"chat_template_kwargs": {"enable_thinking": False}}
    )

    assert train.supports_indices is True
    assert evaluation.supports_indices is False
    assert not any(thinking_flags(tokenize(evaluation, make_dataset(16))))


def test_strategy_without_a_mix_does_not_request_indices(reasoning_tokenizer):
    strategy = build_strategy(reasoning_tokenizer)

    assert strategy.supports_indices is False
    assert strategy.prompter.chat_template_kwargs_mix is None
