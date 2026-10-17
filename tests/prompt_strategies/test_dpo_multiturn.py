"""Tests for the `chat_template.multiturn` trajectory-level preference strategy."""

import copy
import json

import pytest

from axolotl.prompt_strategies.dpo.chat_template import multiturn
from axolotl.utils.dict import DictDefault

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    }
]

PREFIX = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Find X."},
]

CHOSEN = [
    {
        "role": "assistant",
        "reasoning_content": "I should search.",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": {"q": "X"}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "TOOLRESULTALPHA"},
    {"role": "assistant", "reasoning_content": "Got it.", "content": "X is 42."},
]

REJECTED = [
    {
        "role": "assistant",
        "reasoning_content": "Guess a query.",
        "content": "",
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "Y"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_2", "content": "TOOLRESULTBETA"},
    {
        "role": "assistant",
        "reasoning_content": "Search again.",
        "content": "SKIPPEDTURN",
        "train": False,
    },
    {"role": "assistant", "reasoning_content": "Give up.", "content": "No idea."},
]


def _transform(tokenizer, ds_extra=None, **cfg_extra):
    cfg = DictDefault(
        {
            "chat_template": "tokenizer_default",
            "sequence_len": 4096,
            "datasets": [{"type": "chat_template.multiturn", **(ds_extra or {})}],
            **cfg_extra,
        }
    )
    transform_fn, map_kwargs = multiturn(cfg)
    assert map_kwargs == {"remove_columns": "__all__"}
    return lambda sample: transform_fn(copy.deepcopy(sample), tokenizer=tokenizer)


def _sample(**overrides):
    return {
        "messages": PREFIX,
        "chosen": CHOSEN,
        "rejected": REJECTED,
        "tools": TOOLS,
        **overrides,
    }


def _masked_text(tokenizer, ids, mask, keep):
    return tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if m == keep])


def _assert_well_formed(result):
    assert set(result) == {
        "prompt_ids",
        "chosen_ids",
        "rejected_ids",
        "chosen_loss_mask",
        "rejected_loss_mask",
    }
    for side in ("chosen", "rejected"):
        assert len(result[f"{side}_ids"]) == len(result[f"{side}_loss_mask"])
        assert set(result[f"{side}_loss_mask"]) == {0, 1}


class TestMultiTurnQwen3:
    """Qwen3 renders reasoning in <think> and tool calls as <tool_call> json."""

    def test_prompt_and_completions_match_full_render(self, qwen3_tokenizer):
        result = _transform(qwen3_tokenizer)(_sample())
        _assert_well_formed(result)

        prompt = qwen3_tokenizer.apply_chat_template(
            PREFIX, tools=TOOLS, add_generation_prompt=True, tokenize=False
        )
        assert qwen3_tokenizer.decode(result["prompt_ids"]) == prompt
        full = qwen3_tokenizer.apply_chat_template(
            PREFIX + CHOSEN, tools=TOOLS, tokenize=False
        )
        assert (
            qwen3_tokenizer.decode(result["prompt_ids"] + result["chosen_ids"]) == full
        )

    def test_mask_covers_only_assistant_turns(self, qwen3_tokenizer):
        result = _transform(qwen3_tokenizer)(_sample())
        trained = _masked_text(
            qwen3_tokenizer, result["chosen_ids"], result["chosen_loss_mask"], 1
        )
        assert trained == (
            "<think>\nI should search.\n</think>\n\n<tool_call>\n"
            '{"name": "search", "arguments": {"q": "X"}}\n</tool_call><|im_end|>'
            "Got it.\n</think>\n\nX is 42.<|im_end|>"
        )
        masked = _masked_text(
            qwen3_tokenizer, result["chosen_ids"], result["chosen_loss_mask"], 0
        )
        assert "TOOLRESULTALPHA" in masked
        assert "<tool_response>" in masked
        assert "<|im_start|>assistant" in masked

    def test_tool_results_and_train_false_turn_masked(self, qwen3_tokenizer):
        result = _transform(qwen3_tokenizer)(_sample())
        trained = _masked_text(
            qwen3_tokenizer, result["rejected_ids"], result["rejected_loss_mask"], 1
        )
        masked = _masked_text(
            qwen3_tokenizer, result["rejected_ids"], result["rejected_loss_mask"], 0
        )
        assert "TOOLRESULTBETA" not in trained
        assert "TOOLRESULTBETA" in masked
        assert "SKIPPEDTURN" not in trained
        assert "SKIPPEDTURN" in masked
        assert '"q": "Y"' in trained
        assert "No idea.<|im_end|>" in trained

    def test_string_and_dict_arguments_render_the_same(self, qwen3_tokenizer):
        transform = _transform(qwen3_tokenizer)
        as_string = copy.deepcopy(CHOSEN)
        as_string[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"q": "X"})
        assert transform(_sample()) == transform(_sample(chosen=as_string))

    def test_tools_as_json_string(self, qwen3_tokenizer):
        transform = _transform(qwen3_tokenizer)
        assert transform(_sample()) == transform(_sample(tools=json.dumps(TOOLS)))

    def test_shared_prefix_split_from_full_conversations(self, qwen3_tokenizer):
        transform = _transform(qwen3_tokenizer)
        split = transform(
            {"chosen": PREFIX + CHOSEN, "rejected": PREFIX + REJECTED, "tools": TOOLS}
        )
        assert split == transform(_sample())

    def test_mismatched_prefix_raises(self, qwen3_tokenizer):
        other = [PREFIX[0], {"role": "user", "content": "Find Z."}]
        with pytest.raises(ValueError, match="same non-assistant messages"):
            _transform(qwen3_tokenizer)(
                {"chosen": PREFIX + CHOSEN, "rejected": other + REJECTED}
            )

    def test_side_without_trainable_tokens_raises(self, qwen3_tokenizer):
        untrained = copy.deepcopy(CHOSEN)
        for message in untrained:
            message["train"] = False
        with pytest.raises(ValueError, match="no trainable assistant tokens"):
            _transform(qwen3_tokenizer)(_sample(chosen=untrained))

    def test_custom_field_names(self, qwen3_tokenizer):
        transform = _transform(
            qwen3_tokenizer,
            ds_extra={
                "field_messages": "context",
                "field_chosen": "better",
                "field_rejected": "worse",
                "field_tools": "functions",
            },
        )
        renamed = transform(
            {
                "context": PREFIX,
                "better": CHOSEN,
                "worse": REJECTED,
                "functions": TOOLS,
            }
        )
        assert renamed == _transform(qwen3_tokenizer)(_sample())


class TestMultiTurnHarmony:
    """Muse Glimmer's Harmony-style template splits a turn into recipient blocks."""

    def test_masks(self, muse_glimmer_tokenizer):
        transform = _transform(
            muse_glimmer_tokenizer, eot_tokens=["<|eot|>", "<|eom|>"]
        )
        result = transform(_sample())
        _assert_well_formed(result)

        full = muse_glimmer_tokenizer.apply_chat_template(
            PREFIX + CHOSEN, tools=TOOLS, tokenize=False
        )
        assert (
            muse_glimmer_tokenizer.decode(result["prompt_ids"] + result["chosen_ids"])
            == full
        )

        trained = _masked_text(
            muse_glimmer_tokenizer, result["chosen_ids"], result["chosen_loss_mask"], 1
        )
        masked = _masked_text(
            muse_glimmer_tokenizer, result["chosen_ids"], result["chosen_loss_mask"], 0
        )
        assert "I should search.<|eom|>" in trained
        assert '<atem:invoke name="search">' in trained
        assert "X is 42.<|eot|>" in trained
        assert trained.count("<|eot|>") == 2
        assert "TOOLRESULTALPHA" in masked
        assert "<|start|>tool search<|message|>" in masked

        rejected_trained = _masked_text(
            muse_glimmer_tokenizer,
            result["rejected_ids"],
            result["rejected_loss_mask"],
            1,
        )
        assert "TOOLRESULTBETA" not in rejected_trained
        assert "SKIPPEDTURN" not in rejected_trained
        assert "No idea.<|eot|>" in rejected_trained
