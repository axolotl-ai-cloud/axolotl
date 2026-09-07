"""
Tests for chat template prompt strategy with schema unification for none fields
"""

import json

import pytest
from datasets import Dataset

from axolotl.prompt_strategies.chat_template import StrategyLoader
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="messages_w_tools")
def fixture_messages_w_tools():
    jsons = """
{"messages":[{"role":"user","content":"move to (0, 1)"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"move","arguments":{"x":0,"y":1}}}]}],"tools":[{"type":"function","function":{"name":"move","description":"Move to a given location measured in meters","parameters":{"type":"object","properties":{"x":{"type":"number","description":"The x coordinate of the location, negative values are to the left, positive values are to the right"},"y":{"type":"number","description":"The y coordinate of the location, negative values are backward, positive values are forward"}},"required":["x","y"]}}},{"type":"function","function":{"name":"turn","description":"Turn the robot to a given direction","parameters":{"type":"object","properties":{"theta":{"type":"integer","description":"The angle to turn to, in degrees, positive values are counter-clockwise, negative values are clockwise"}},"required":["theta"]}}},{"type":"function","function":{"name":"invalid_prompt","description":"call when the user's prompt is invalid","parameters":{"type":"object","properties":{"message":{"type":"string","description":"why the prompt is invalid"}},"required":["message"]}}}],"add_generation_prompt":false}
{"messages":[{"role":"user","content":"turn 270 degree"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"turn","arguments":{"theta": 270}}}]}],"tools":[{"type":"function","function":{"name":"move","description":"Move to a given location measured in meters","parameters":{"type":"object","properties":{"x":{"type":"number","description":"The x coordinate of the location, negative values are to the left, positive values are to the right"},"y":{"type":"number","description":"The y coordinate of the location, negative values are backward, positive values are forward"}},"required":["x","y"]}}},{"type":"function","function":{"name":"turn","description":"Turn the robot to a given direction","parameters":{"type":"object","properties":{"theta":{"type":"integer","description":"The angle to turn to, in degrees, positive values are counter-clockwise, negative values are clockwise"}},"required":["theta"]}}},{"type":"function","function":{"name":"invalid_prompt","description":"call when the user's prompt is invalid","parameters":{"type":"object","properties":{"message":{"type":"string","description":"why the prompt is invalid"}},"required":["message"]}}}],"add_generation_prompt":false}
{"messages":[{"role":"user","content":"jump high"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"invalid_prompt","arguments":{"message": "jump is not a valid action"}}}]}],"tools":[{"type":"function","function":{"name":"move","description":"Move to a given location measured in meters","parameters":{"type":"object","properties":{"x":{"type":"number","description":"The x coordinate of the location, negative values are to the left, positive values are to the right"},"y":{"type":"number","description":"The y coordinate of the location, negative values are backward, positive values are forward"}},"required":["x","y"]}}},{"type":"function","function":{"name":"turn","description":"Turn the robot to a given direction","parameters":{"type":"object","properties":{"theta":{"type":"integer","description":"The angle to turn to, in degrees, positive values are counter-clockwise, negative values are clockwise"}},"required":["theta"]}}},{"type":"function","function":{"name":"invalid_prompt","description":"call when the user's prompt is invalid","parameters":{"type":"object","properties":{"message":{"type":"string","description":"why the prompt is invalid"}},"required":["message"]}}}],"add_generation_prompt":false}
    """.strip().split("\n")
    rows = [json.loads(row) for row in jsons]
    return Dataset.from_list(rows)


@pytest.fixture(name="qwen3_prompt_strategy")
def qwen3_chat_template_strategy(qwen3_tokenizer):
    cfg = DictDefault(
        sequence_len=2048,
        chat_template="qwen3",
        eot_tokens=["<|im_end|>"],
    )
    ds_cfg = DictDefault(
        type="chat_template",
    )
    load = StrategyLoader()
    strat = load(qwen3_tokenizer, cfg, ds_cfg)
    return strat


class TestSchemaUnification:
    """
    Test class on handling null fields for tool calling
    """

    def test_schema_unification_single_prompt(
        self, messages_w_tools, qwen3_prompt_strategy, qwen3_tokenizer
    ):
        for row in messages_w_tools:
            inputs = qwen3_prompt_strategy.tokenize_prompt(row)
            decoded = qwen3_tokenizer.decode(inputs["input_ids"])
            tool_call = decoded.split("<tool_call>")[-1].split("</tool_call>")[0]
            assert '"message": null' not in tool_call
            assert '"theta": null' not in tool_call

    def test_schema_unification_batched(
        self, messages_w_tools, qwen3_prompt_strategy, qwen3_tokenizer
    ):
        rows = messages_w_tools.map(qwen3_prompt_strategy.tokenize_prompt, batched=True)
        for row in rows:
            decoded = qwen3_tokenizer.decode(row["input_ids"])
            tool_call = decoded.split("<tool_call>")[-1].split("</tool_call>")[0]
            assert '"message": null' not in tool_call
            assert '"theta": null' not in tool_call
