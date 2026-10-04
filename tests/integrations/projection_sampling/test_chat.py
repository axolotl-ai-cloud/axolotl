"""Chat dataset normalization, target context, and standard SFT mask parity."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_projection_sampling import ScriptedBackend
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from axolotl.integrations.projection_sampling.args import ProjectionSamplingConfig
from axolotl.integrations.projection_sampling.chat import sample_chat
from axolotl.integrations.projection_sampling.plugin import (
    ProjectionSamplingPlugin,
    cache_path,
)
from axolotl.integrations.projection_sampling.sampler import ProjectionSampler
from axolotl.integrations.projection_sampling.tokenization import load as cache_strategy
from axolotl.prompt_strategies.chat_template import load
from axolotl.utils.dict import DictDefault


@pytest.fixture
def tokenizer():
    vocab = {
        text: i
        for i, text in enumerate(
            [
                "<unk>",
                "<pad>",
                "<eos>",
                "<system>",
                "<user>",
                "<assistant>",
                "<tool>",
                "system",
                "question",
                "followup",
                "expert",
                "rewritten",
                "masked",
                "result",
            ]
        )
    }
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<system>", "<user>", "<assistant>", "<tool>"]}
    )
    tokenizer.chat_template = "{% for message in messages %}{{ '<' + message['role'] + '> ' + message['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    return tokenizer


@pytest.fixture
def cfg():
    return DictDefault(
        sequence_len=1024,
        train_on_inputs=False,
        chat_template="tokenizer_default",
        chat_template_kwargs={},
    )


def setup_sampler(tokenizer, count=1):
    backend = ScriptedBackend(
        [[tokenizer.convert_tokens_to_ids("rewritten"), tokenizer.eos_token_id]] * count
    )
    backend.tokenizer = tokenizer
    backend.eos_token_ids = {tokenizer.eos_token_id}
    sampler = ProjectionSampler(
        backend, ProjectionSamplingConfig(block_size=2, max_new_tokens=2, mcmc_steps=0)
    )
    return sampler, backend


@pytest.mark.parametrize(
    "gain,margin,retained",
    [(0.25, 0.125, True), (0.125, 0.125, False), (0, 0, False), (-0.25, 0, False)],
)
@pytest.mark.parametrize("train_on_inputs", [False, True])
def test_final_margin_compares_original_labeled_reply(
    tokenizer, cfg, gain, margin, retained, train_on_inputs
):
    cfg.train_on_inputs = train_on_inputs
    strategy = load(tokenizer, cfg, {"train_on_eos": "none"})
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    sampler, backend = setup_sampler(tokenizer)
    sampler.config.min_logprob_improvement = margin
    rewritten_id = tokenizer.convert_tokens_to_ids("rewritten")
    expert_id = tokenizer.convert_tokens_to_ids("expert")
    backend.target = lambda tokens: sum(
        -1 + gain if token == rewritten_id else -1 if token == expert_id else -100
        for token in tokens
    )
    record = sample_chat(row, strategy, sampler)
    metadata = record["sampling"][0]
    assert metadata["original_labeled_tokens"] == 1
    assert metadata["rewritten_labeled_tokens"] == 1
    assert metadata["original_mean_logprob"] == -1
    assert metadata["logprob_improvement"] == gain
    assert metadata["logprob_margin_passed"] is retained
    assert metadata["fallback_to_expert"] is not retained
    assert record["messages"][-1]["content"] == ("rewritten" if retained else "expert")
    if not retained:
        assert metadata["fallback_reason"] == "insufficient_logprob_improvement"
        assert record["labels"] == strategy.tokenize_prompt(row)["labels"]
    margin_calls = [call for call in backend.calls if call[0] == "target"][1:]
    assert [call[2] for call in margin_calls] == [[expert_id], [rewritten_id]]


def test_final_margin_counts_only_current_turn_with_rewritten_history(tokenizer, cfg):
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
            {"role": "user", "content": "followup"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    strategy = load(
        tokenizer, cfg, {"roles_to_train": ["assistant"], "train_on_eos": "none"}
    )
    sampler, backend = setup_sampler(tokenizer, count=2)
    sampler.config.min_logprob_improvement = 0.125
    rewritten_id = tokenizer.convert_tokens_to_ids("rewritten")
    backend.target = lambda tokens: sum(
        -0.5 if token == rewritten_id else -1 for token in tokens
    )
    record = sample_chat(row, strategy, sampler)
    assert all(not item["fallback_to_expert"] for item in record["sampling"])
    assert all(
        item["original_labeled_tokens"] == item["rewritten_labeled_tokens"] == 1
        for item in record["sampling"]
    )
    last_comparison = [call for call in backend.calls if call[0] == "target"][-2:]
    assert all(rewritten_id in call[1] for call in last_comparison)
    assert all(len(call[2]) == 1 for call in last_comparison)


def test_final_margin_respects_last_eos_policy_in_full_conversation(tokenizer, cfg):
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
            {"role": "user", "content": "followup"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    strategy = load(
        tokenizer, cfg, {"roles_to_train": ["assistant"], "train_on_eos": "last"}
    )
    sampler, backend = setup_sampler(tokenizer, count=2)
    sampler.config.min_logprob_improvement = 0
    rewritten_id = tokenizer.convert_tokens_to_ids("rewritten")
    backend.target = lambda tokens: sum(
        -0.5 if token == rewritten_id else -1 for token in tokens
    )
    record = sample_chat(row, strategy, sampler)
    assert [item["original_labeled_tokens"] for item in record["sampling"]] == [1, 2]
    assert [item["rewritten_labeled_tokens"] for item in record["sampling"]] == [1, 2]


@pytest.mark.parametrize("eos_policy,expected_tokens", [("none", 1), ("all", 2)])
def test_final_margin_locates_original_and_rewritten_headers_independently(
    tokenizer, cfg, eos_policy, expected_tokens
):
    tokenizer.chat_template = "{% for message in messages %}{{ '<' + message['role'] + '> ' }}{% if message['role'] == 'assistant' and message['content'] != 'expert' %}{{ 'masked ' }}{% endif %}{{ message['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> masked ' }}{% endif %}"
    strategy = load(
        tokenizer, cfg, {"roles_to_train": ["assistant"], "train_on_eos": eos_policy}
    )
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    sampler, backend = setup_sampler(tokenizer)
    sampler.config.min_logprob_improvement = 0
    rewritten_id = tokenizer.convert_tokens_to_ids("rewritten")
    backend.target = lambda tokens: sum(
        -0.5 if token == rewritten_id else -1 for token in tokens
    )
    record = sample_chat(row, strategy, sampler)
    metadata = record["sampling"][0]
    assert metadata["original_labeled_tokens"] == expected_tokens
    assert metadata["rewritten_labeled_tokens"] == expected_tokens
    assert metadata["logprob_improvement"] == 0.5 / expected_tokens
    assert not metadata["fallback_to_expert"]


def test_disabled_margin_does_not_score_expert_reply(tokenizer, cfg):
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    sampler, backend = setup_sampler(tokenizer)
    record = sample_chat(row, load(tokenizer, cfg), sampler)
    assert "logprob_improvement" not in record["sampling"][0]
    assert len([call for call in backend.calls if call[0] == "target"]) == 1


@pytest.mark.parametrize("ceiling,retained", [(0.5, True), (0.49, False)])
@pytest.mark.parametrize("train_on_inputs", [False, True])
@pytest.mark.parametrize("eos_policy,count", [("none", 1), ("all", 2)])
def test_final_kl_gate_uses_only_labeled_sampled_reply_positions(
    tokenizer, cfg, ceiling, retained, train_on_inputs, eos_policy, count
):
    from unittest.mock import Mock

    cfg.train_on_inputs = train_on_inputs
    strategy = load(tokenizer, cfg, {"train_on_eos": eos_policy})
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    sampler, backend = setup_sampler(tokenizer)
    sampler.config.max_proposal_kl = ceiling
    backend.proposal_kl = Mock(return_value=[0.5] * count)
    record = sample_chat(row, strategy, sampler)
    metadata = record["sampling"][0]
    assert metadata["proposal_to_base_mean_kl"] == 0.5
    assert metadata["proposal_kl_labeled_tokens"] == count
    assert metadata["proposal_kl_gate_passed"] is retained
    assert metadata["fallback_to_expert"] is not retained
    target, proposal, tokens, positions = backend.proposal_kl.call_args.args
    assert positions == list(range(count))
    assert target == [call[1] for call in backend.calls if call[0] == "target"][0]
    assert tokens == [
        tokenizer.convert_tokens_to_ids("rewritten"),
        tokenizer.eos_token_id,
    ]
    assert proposal != target
    if not retained:
        assert metadata["fallback_reason"] == "excessive_proposal_kl"
        assert record["labels"] == strategy.tokenize_prompt(row)["labels"]


def test_failed_margin_skips_kl_scoring(tokenizer, cfg):
    from unittest.mock import Mock

    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    sampler, backend = setup_sampler(tokenizer)
    sampler.config.min_logprob_improvement = 1
    sampler.config.max_proposal_kl = 1
    backend.proposal_kl = Mock(side_effect=AssertionError("margin already failed"))
    record = sample_chat(row, load(tokenizer, cfg), sampler)
    assert (
        record["sampling"][0]["fallback_reason"] == "insufficient_logprob_improvement"
    )
    backend.proposal_kl.assert_not_called()


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("train_on_inputs", [False, True])
def test_custom_fields_roles_and_parser_mask_parity(
    tokenizer, cfg, serialized, train_on_inputs
):
    cfg.train_on_inputs = train_on_inputs
    ds_cfg = {
        "field_messages": "conversation",
        "message_property_mappings": {"role": "speaker", "content": "text"},
        "roles": {"user": ["human"], "assistant": ["bot"]},
    }
    messages = [
        {"speaker": "human", "text": "question"},
        {"speaker": "bot", "text": "expert"},
    ]
    row = {"conversation": json.dumps(messages) if serialized else messages}
    original = deepcopy(row)
    strategy = load(tokenizer, cfg, ds_cfg)
    sampler, backend = setup_sampler(tokenizer)
    record = sample_chat(row, strategy, sampler)
    assert row == original
    expected = strategy.tokenize_prompt(
        {"conversation": [messages[0], {"speaker": "bot", "text": "rewritten"}]}
    )
    assert {key: record[key] for key in expected} == expected
    assert record["sampling"][0]["fallback_to_expert"] is False
    assert record["messages"][-1]["content"] == "rewritten"
    target = next(call[1] for call in backend.calls if call[0] == "target")
    assert tokenizer.convert_tokens_to_ids("expert") not in target
    assert target == strategy.prompter.build_prompt(
        [{"role": "user", "content": "question"}], add_generation_prompt=True
    )
    assert cache_strategy(tokenizer, cfg).tokenize_prompt(record) == expected


def test_multiturn_uses_rewritten_history_and_keeps_system(tokenizer, cfg):
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
            {"role": "user", "content": "followup"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    sampler, backend = setup_sampler(tokenizer, count=2)
    strategy = load(tokenizer, cfg)
    record = sample_chat(row, strategy, sampler)
    assert [turn["content"] for turn in record["messages"]] == [
        "system",
        "question",
        "rewritten",
        "followup",
        "rewritten",
    ]
    contexts = [call[1] for call in backend.calls if call[0] == "target"]
    assert tokenizer.convert_tokens_to_ids("system") in contexts[0]
    assert tokenizer.convert_tokens_to_ids("rewritten") in contexts[1]
    assert tokenizer.convert_tokens_to_ids("expert") not in contexts[1]
    expected_row = deepcopy(row)
    expected_row["messages"][2]["content"] = "rewritten"
    expected_row["messages"][4]["content"] = "rewritten"
    assert record["labels"] == strategy.tokenize_prompt(expected_row)["labels"]


def test_proposal_embeds_messages_without_template_control_tokens(
    tokenizer, cfg, monkeypatch
):
    row = {
        "messages": [
            {"role": "system", "content": "system", "learn": False},
            {"role": "user", "content": "question", "learn": False},
            {"role": "assistant", "content": "expert", "learn": True},
        ]
    }
    strategy = load(tokenizer, cfg, {"message_field_training": "learn"})
    sampler, _ = setup_sampler(tokenizer)
    sample = sampler.sample
    contexts = []

    def capture(question, expert, **kwargs):
        assert tokenizer.eos_token not in question
        contexts.append(json.loads(question))
        return sample(question, expert, **kwargs)

    monkeypatch.setattr(sampler, "sample", capture)
    record = sample_chat(row, strategy, sampler)
    assert contexts == [
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ]
    ]
    assert not record["sampling"][0]["fallback_to_expert"]


def test_per_message_flags_and_span_masks_are_preserved(tokenizer, cfg):
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "masked", "learn": False},
            {"role": "user", "content": "followup"},
            {
                "role": "assistant",
                "content": "expert",
                "learn": True,
                "spans": [{"begin_offset": 0, "end_offset": 5, "train": True}],
            },
        ]
    }
    strategy = load(
        tokenizer,
        cfg,
        {"message_field_training": "learn", "message_field_training_detail": "spans"},
    )
    sampler, backend = setup_sampler(tokenizer, count=0)
    record = sample_chat(row, strategy, sampler)
    assert not backend.calls
    assert record["labels"] == strategy.tokenize_prompt(row)["labels"]
    assert record["sampling"] == [
        {"message_index": 3, "skipped": "partial_training_mask"}
    ]


def test_drop_system_and_eos_policy_follow_parser(tokenizer, cfg):
    strategy = load(
        tokenizer, cfg, {"drop_system_message": True, "train_on_eos": "none"}
    )
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    sampler, backend = setup_sampler(tokenizer)
    record = sample_chat(row, strategy, sampler)
    assert record["messages"][0]["role"] == "user"
    assert all(
        label == -100
        for token, label in zip(record["input_ids"], record["labels"], strict=True)
        if token == tokenizer.eos_token_id
    )
    context = next(call[1] for call in backend.calls if call[0] == "target")
    assert tokenizer.convert_tokens_to_ids("system") not in context


@pytest.mark.parametrize("template_uses_thinking", [False, True])
def test_split_thinking_checks_transformed_reply_tokens(
    tokenizer, cfg, template_uses_thinking
):
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    if template_uses_thinking:
        tokenizer.chat_template = (
            "{% for message in messages %}{{ '<' + message['role'] + '> ' }}"
            "{% if message.get('reasoning_content') %}"
            "{{ '<think> ' + message['reasoning_content'] + ' </think> ' }}"
            "{% endif %}{{ message['content'] + eos_token }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
        )
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    strategy = load(tokenizer, cfg, {"split_thinking": True})
    generated = tokenizer.encode(
        "<think> system </think> rewritten", add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    backend = ScriptedBackend([generated])
    backend.tokenizer = tokenizer
    backend.eos_token_ids = {tokenizer.eos_token_id}
    sampler = ProjectionSampler(
        backend,
        ProjectionSamplingConfig(
            block_size=len(generated), max_new_tokens=len(generated), mcmc_steps=0
        ),
    )
    record = sample_chat(row, strategy, sampler)
    metadata = record["sampling"][0]
    assert metadata["fallback_to_expert"] is not template_uses_thinking
    if template_uses_thinking:
        context = strategy.prompter.build_prompt(
            row["messages"][:-1], add_generation_prompt=True
        )
        assert record["input_ids"][len(context) :] == generated
        assert record["messages"][-1]["reasoning_content"] == "system"
    else:
        assert metadata["fallback_reason"] == "template_token_mismatch"
        expected = strategy.tokenize_prompt(row)
        assert {key: record[key] for key in expected} == expected


@pytest.mark.parametrize("reason", ["verification", "template"])
def test_chat_fallback_keeps_original_parser_labels(tokenizer, cfg, reason):
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    strategy = load(tokenizer, cfg)
    sampler, backend = setup_sampler(tokenizer)
    verifier = (lambda **kwargs: False) if reason == "verification" else None
    if reason == "template":
        backend.tokenizer.decode = lambda *args, **kwargs: "result"
    record = sample_chat(row, strategy, sampler, verifier)
    assert record["sampling"][0]["fallback_to_expert"]
    assert record["labels"] == strategy.tokenize_prompt(row)["labels"]
    if reason == "template":
        assert record["sampling"][0]["fallback_reason"] == "template_token_mismatch"


def test_cache_fingerprint_tracks_chat_labels_but_not_learning_rate(tmp_path, cfg):
    path = tmp_path / "chat.jsonl"
    path.write_text("{}\n")
    cfg.datasets = [{"path": str(path), "type": "chat_template"}]
    config = ProjectionSamplingConfig()
    original = cache_path(cfg, config)
    cfg.learning_rate = 0.01
    assert cache_path(cfg, config) == original
    cfg.train_on_inputs = True
    assert cache_path(cfg, config) != original
    cfg.train_on_inputs = False
    cfg.chat_template_kwargs = {"enable_thinking": False}
    assert cache_path(cfg, config) != original


def test_plugin_chat_cache_and_training_reuse(tmp_path, tokenizer, cfg, monkeypatch):
    import axolotl.common.datasets as common
    from axolotl.integrations.projection_sampling.backends.transformers import (
        TransformersBackend,
    )

    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps(row) + "\n")
    cfg.output_dir = str(tmp_path / "output")
    cfg.datasets = [
        {
            "path": str(path),
            "ds_type": "json",
            "type": "chat_template",
            "split": "train",
        }
    ]
    cfg.projection_sampling = {
        "cache_dir": str(tmp_path / "cache"),
        "device": "cpu",
        "block_size": 2,
        "max_new_tokens": 2,
        "mcmc_steps": 0,
    }
    sampler, backend = setup_sampler(tokenizer)
    monkeypatch.setattr(TransformersBackend, "from_config", lambda *args: backend)
    monkeypatch.setattr(common, "load_datasets", lambda **kwargs: "prepared")
    plugin = ProjectionSamplingPlugin()
    assert plugin.load_datasets(cfg, preprocess=True) == "prepared"
    assert backend.closed
    record = json.loads(
        cache_path(
            cfg, ProjectionSamplingConfig.model_validate(cfg.projection_sampling)
        ).read_text()
    )
    assert record["messages"][-1]["content"] == "rewritten"
    assert len(record["input_ids"]) == len(record["labels"])
    readable = json.loads(
        (Path(cfg.output_dir) / "projection-sampling" / "rewritten.jsonl").read_text()
    )
    assert readable["messages"] == record["messages"]
    assert readable["sampling"][0]["expert_response"] == "expert"
    assert readable["sampling_seed"] == 42
    assert not {"input_ids", "labels", "attention_mask"} & readable.keys()
    assert "sampled_token_ids" not in readable["sampling"][0]
    monkeypatch.setattr(
        TransformersBackend, "from_config", lambda *args: pytest.fail("cache miss")
    )
    assert plugin.load_datasets(cfg) == "prepared"


def test_legacy_last_reply_policy(tokenizer, cfg):
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
            {"role": "user", "content": "followup"},
            {"role": "assistant", "content": "expert"},
        ]
    }
    strategy = load(tokenizer, cfg, {"roles_to_train": None, "train_on_eos": None})
    sampler, backend = setup_sampler(tokenizer)
    record = sample_chat(row, strategy, sampler)
    assert [entry["message_index"] for entry in record["sampling"]] == [3]
    assert record["messages"][1]["content"] == "expert"
    assert record["messages"][3]["content"] == "rewritten"
    assert sum(call[0] == "sample" for call in backend.calls) == 1


def test_template_kwargs_and_tools_reach_target_and_proposal(tokenizer, cfg):
    tokenizer.chat_template = (
        "{% if include_marker %}{{ 'system ' }}{% endif %}{% if tools %}{{ 'result ' }}{% endif %}"
        + tokenizer.chat_template
    )
    cfg.chat_template_kwargs = {"include_marker": True}
    row = {
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "expert"},
        ],
    }
    sampler, backend = setup_sampler(tokenizer)
    record = sample_chat(row, load(tokenizer, cfg), sampler)
    assert not record["sampling"][0]["fallback_to_expert"]
    marker = [
        tokenizer.convert_tokens_to_ids("system"),
        tokenizer.convert_tokens_to_ids("result"),
    ]
    for _kind, context, _ in backend.calls:
        assert context[:2] == marker
