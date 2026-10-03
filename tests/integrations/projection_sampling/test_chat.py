"""Chat dataset normalization, target context, and standard SFT mask parity."""

import json
from copy import deepcopy

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
