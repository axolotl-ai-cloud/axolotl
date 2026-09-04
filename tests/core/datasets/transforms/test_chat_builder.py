import pytest

from axolotl.core.datasets.transforms.chat_builder import chat_message_transform_builder


@pytest.fixture(name="toolcalling_sample")
def fixture_toolcalling_sample():
    return {
        "messages": [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "What's the temperature in Paris?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_temperature",
                            "arguments": {
                                "location": "Paris, France",
                                "unit": "celsius",
                            },
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_current_temperature",
                "content": "22.0",
            },
            {
                "role": "assistant",
                "content": "The temperature is 22.0 degrees Celsius.",
            },
        ]
    }


class TestChatBuilderText:
    def test_str_content_becomes_text_item(self):
        transform = chat_message_transform_builder()
        out = transform({"messages": [{"role": "user", "content": "hello"}]})
        assert out["conversation"][0]["content"] == [{"type": "text", "value": "hello"}]

    def test_sharegpt_fields(self):
        transform = chat_message_transform_builder(conversations_field="conversations")
        out = transform(
            {
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "hi"},
                ]
            }
        )
        assert out["conversation"][0] == {
            "role": "user",
            "content": [{"type": "text", "value": "hello"}],
            "weight": 0,
        }
        assert out["conversation"][1]["role"] == "assistant"
        assert out["conversation"][1]["weight"] == 1

    def test_message_weight_field(self):
        transform = chat_message_transform_builder(message_field_training="train")
        out = transform({"messages": [{"role": "user", "content": "x", "train": 1}]})
        assert out["conversation"][0]["weight"] == 1


class TestChatBuilderToolCalls:
    def test_openai_tool_calls(self, toolcalling_sample):
        transform = chat_message_transform_builder()
        out = transform(toolcalling_sample)
        conv = out["conversation"]
        assert conv[2]["content"] == [
            {
                "type": "tool_call",
                "value": {
                    "name": "get_current_temperature",
                    "arguments": {"location": "Paris, France", "unit": "celsius"},
                },
            }
        ]
        assert conv[3]["content"] == [
            {
                "type": "tool_response",
                "value": {
                    "name": "get_current_temperature",
                    "content": "22.0",
                },
            }
        ]

    def test_tool_calls_without_type_field(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {"role": "user", "content": "move to (0, 1)"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "move",
                                    "arguments": {"x": 0, "y": 1},
                                }
                            }
                        ],
                    },
                ]
            }
        )
        assert out["conversation"][1]["content"] == [
            {
                "type": "tool_call",
                "value": {"name": "move", "arguments": {"x": 0, "y": 1}},
            }
        ]

    def test_tool_calls_with_string_arguments(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_stock_price",
                                    "arguments": '{"symbol": "AAPL"}',
                                },
                            }
                        ],
                    }
                ]
            }
        )
        assert out["conversation"][0]["content"] == [
            {
                "type": "tool_call",
                "value": {
                    "name": "get_stock_price",
                    "arguments": {"symbol": "AAPL"},
                    "id": "call_1",
                },
            }
        ]

    def test_malformed_string_arguments_fail_naming_the_tool_call(self):
        transform = chat_message_transform_builder()
        with pytest.raises(ValueError, match="get_stock_price"):
            transform(
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {
                                        "name": "get_stock_price",
                                        "arguments": '{"symbol": ',
                                    },
                                }
                            ],
                        }
                    ]
                }
            )
        with pytest.raises(ValueError, match="call_9"):
            transform(
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {
                                        "name": "get_stock_price",
                                        "arguments": '{"symbol": ',
                                    },
                                }
                            ],
                        }
                    ]
                }
            )

    def test_content_function_items_converted(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "move",
                                    "arguments": {"x": 1, "y": 2},
                                },
                            }
                        ],
                    }
                ]
            }
        )
        assert out["conversation"][0]["content"] == [
            {
                "type": "tool_call",
                "value": {"name": "move", "arguments": {"x": 1, "y": 2}},
            }
        ]

    def test_text_and_tool_calls_coexist(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "f", "arguments": {}},
                            }
                        ],
                    }
                ]
            }
        )
        assert out["conversation"][0]["content"] == [
            {"type": "text", "value": "Let me check."},
            {"type": "tool_call", "value": {"name": "f", "arguments": {}}},
        ]

    def test_tool_message_without_content_field(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "f", "arguments": {}},
                            }
                        ],
                    },
                ]
            }
        )
        assert out["conversation"][1]["content"] == [
            {"type": "tool_call", "value": {"name": "f", "arguments": {}}}
        ]

    def test_tool_call_id_on_response(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "tool",
                        "name": "move",
                        "tool_call_id": "call_1",
                        "content": "ok",
                    }
                ]
            }
        )
        assert out["conversation"][0]["content"] == [
            {
                "type": "tool_response",
                "value": {"name": "move", "content": "ok", "id": "call_1"},
            }
        ]

    def test_tool_text_key_content(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "tool",
                        "name": "get_temp",
                        "content": [{"type": "text", "text": "22.0"}],
                    }
                ]
            }
        )
        assert out["conversation"][0]["content"] == [
            {
                "type": "tool_response",
                "value": {"name": "get_temp", "content": "22.0"},
            }
        ]

    def test_tool_scalar_list_content_becomes_response(self):
        transform = chat_message_transform_builder()
        out = transform(
            {"messages": [{"role": "tool", "name": "get_temp", "content": ["ok"]}]}
        )
        assert out["conversation"][0]["content"] == [
            {
                "type": "tool_response",
                "value": {"name": "get_temp", "content": "ok"},
            }
        ]

    def test_tool_dict_content(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "tool",
                        "name": "get_date",
                        "content": {"date": "2024-09-09"},
                    }
                ]
            }
        )
        assert out["conversation"][0]["content"] == [
            {
                "type": "tool_response",
                "value": {"name": "get_date", "content": {"date": "2024-09-09"}},
            }
        ]

    def test_tool_message_without_name_keeps_text(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": "ok",
                        "tool_call_id": "call_1",
                    }
                ]
            }
        )
        assert out["conversation"][0]["content"] == [{"type": "text", "value": "ok"}]

    def test_nontext_items_pass_through(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "value": "/tmp/a.png"},
                            {"type": "text", "value": "what is this?"},
                        ],
                    }
                ]
            }
        )
        assert out["conversation"][0]["content"] == [
            {"type": "image", "value": "/tmp/a.png"},
            {"type": "text", "value": "what is this?"},
        ]

    def test_empty_content(self):
        transform = chat_message_transform_builder()
        out = transform({"messages": [{"role": "user", "content": ""}]})
        assert out["conversation"][0]["content"] == []

    def test_dict_content_wrapped_in_list(self):
        transform = chat_message_transform_builder()
        out = transform(
            {"messages": [{"role": "user", "content": {"type": "text", "value": "hi"}}]}
        )
        assert out["conversation"][0]["content"] == [{"type": "text", "value": "hi"}]

    def test_missing_content_field_raises(self):
        transform = chat_message_transform_builder()
        with pytest.raises(ValueError, match="message_content"):
            transform({"messages": [{"role": "user"}]})

    def test_content_field_detected_in_later_message(self):
        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_temperature",
                                    "arguments": {},
                                },
                            }
                        ],
                    },
                    {"role": "assistant", "content": "The temperature is 22.0."},
                ]
            }
        )
        assert out["conversation"][0]["content"][0]["type"] == "tool_call"
        assert out["conversation"][1]["content"] == [
            {"type": "text", "value": "The temperature is 22.0."}
        ]

    def test_weight_field_detected_in_later_message(self):
        transform = chat_message_transform_builder(message_field_training="weight")
        out = transform(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi", "weight": 0},
                ]
            }
        )
        assert out["conversation"][1]["weight"] == 0

    def test_tool_contents_validate_as_message_contents(self):
        from axolotl.core.chat.messages import MessageContents

        transform = chat_message_transform_builder()
        out = transform(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_temperature",
                                    "arguments": {"location": "Paris"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "get_temperature",
                        "content": "22.0",
                        "tool_call_id": "call_1",
                    },
                ]
            }
        )
        for content in out["conversation"][0]["content"]:
            assert content["type"] == "tool_call"
            MessageContents(**content)
        response = out["conversation"][1]["content"][0]
        assert response["type"] == "tool_response"
        MessageContents(**response)
