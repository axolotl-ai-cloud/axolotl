"""
This module contains a function that builds a transform that takes a row from the
dataset and converts it to a Chat.
"""

import json
from typing import Any, Mapping


def chat_message_transform_builder(
    train_on_inputs=False,
    conversations_field: str = "messages",
    message_field_role: str | list[str] | None = None,  # commonly "role"
    message_field_content: str | list[str] | None = None,  # commonly "content"
    message_field_training: str | list[str] | None = None,  # commonly "weight"
):
    """Builds a transform that takes a row from the dataset and converts it to a Chat

    Args:
        train_on_inputs (bool, optional):
            If True, the transform will train on the inputs. If False, the transform will train on the targets.
            Defaults to False.
        conversations_field (str, optional):
            The field name of the conversations. Defaults to "messages".
        message_field_role (str | list[str], optional):
            The field name of the role.
        message_field_content (str | list[str], optional):
            The field name of the message content.
        message_field_training (str | list[str], optional):
            The field name of the train/weight.

    Returns:
        Callable:
            A function that takes a list of conversations and returns a list of messages.
    """

    if message_field_training is None:
        message_field_training = ["train", "weight"]
    if message_field_content is None:
        message_field_content = ["value", "text", "content"]
    if message_field_role is None:
        message_field_role = ["role", "from"]
    message_field_role = (
        [message_field_role]
        if isinstance(message_field_role, str)
        else message_field_role
    )
    message_field_content = (
        [message_field_content]
        if isinstance(message_field_content, str)
        else message_field_content
    )
    message_weight_fields = (
        [message_field_training]
        if isinstance(message_field_training, str)
        else message_field_training
    )

    role_value_mappings = {
        "system": "system",
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "tool": "tool",
        "ipython": "ipython",
    }
    if train_on_inputs:
        role_default_weights_mappings = {
            "system": 1,
            "user": 1,
            "assistant": 1,
            "tool": 1,
            "ipython": 1,
        }
    else:
        role_default_weights_mappings = {
            "system": 0,
            "user": 0,
            "assistant": 1,
            "tool": 0,
            "ipython": 0,
        }

    def transform_builder(sample: Mapping[str, Any]):
        if conversations_field not in sample:
            raise ValueError(f"Field '{conversations_field}' not found in sample.")
        # if none of the role fields are in the message, raise an error
        if not any(
            role in sample[conversations_field][0] for role in message_field_role
        ):
            raise ValueError("No role field found in message.")
        role_field = next(
            role
            for role in message_field_role
            if role in sample[conversations_field][0]
        )
        has_content_field = any(
            field in message
            for message in sample[conversations_field]
            for field in message_field_content
        )
        has_tool_calls = any(
            "tool_calls" in message for message in sample[conversations_field]
        )
        if not has_content_field and not has_tool_calls:
            raise ValueError("No message_content field found in message.")
        message_content_field = next(
            (
                field
                for field in message_field_content
                if any(field in message for message in sample[conversations_field])
            ),
            message_field_content[0],
        )
        if not any(
            field in message
            for message in sample[conversations_field]
            for field in message_weight_fields
        ):
            message_weight_field = None
        else:
            message_weight_field = next(
                field
                for field in message_weight_fields
                if any(field in message for message in sample[conversations_field])
            )

        messages = []
        for message in sample[conversations_field]:
            role = role_value_mappings[message[role_field]]
            weight = (
                int(message[message_weight_field])
                if message_weight_field and message_weight_field in message
                else role_default_weights_mappings[role]
            )
            messages.append(
                {
                    "role": role,
                    "content": _normalize_content(message, message_content_field, role),
                    "weight": weight,
                }
            )

        return {"conversation": messages}

    return transform_builder


def _convert_openai_tool_call(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a raw OpenAI tool call entry to the internal tool_call content format."""
    function = tool_call.get("function", tool_call)
    tool_call_id = tool_call.get("id") or function.get("id")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        if not arguments:
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                raise ValueError(
                    "Could not parse arguments JSON for tool call "
                    f"name={function.get('name', '')!r} "
                    f"id={tool_call_id!r}: {arguments!r}"
                ) from exc
    value: dict[str, Any] = {"name": function.get("name", ""), "arguments": arguments}
    if tool_call_id is not None:
        value["id"] = tool_call_id
    return {"type": "tool_call", "value": value}


def _normalize_content(
    message: Mapping[str, Any], message_content_field: str, role: str
) -> list[dict[str, Any]]:
    """Normalize raw dataset message content into the internal content format.

    Handles OpenAI-style tool calls (top-level `tool_calls` field or
    `{"type": "function", ...}` content items), string-encoded JSON arguments,
    and tool messages with a `name`, which become `tool_response` contents.
    """
    content = message.get(message_content_field, [])
    if content is None or content == "":
        content = []
    if isinstance(content, str):
        content = [{"type": "text", "value": content}]
    elif not isinstance(content, list):
        content = [content]

    normalized: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, Mapping) and item.get("type") == "function":
            normalized.append(_convert_openai_tool_call(item))
        else:
            normalized.append(item)

    if isinstance(message.get("tool_calls"), list):
        normalized.extend(
            _convert_openai_tool_call(tool_call) for tool_call in message["tool_calls"]
        )

    if role in ("tool", "ipython") and message.get("name"):
        if len(normalized) == 1:
            item = normalized[0]
            if not isinstance(item, Mapping):
                response_content = item
            elif item.get("type") == "text":
                response_content = item.get("value", item.get("text", ""))
            elif "type" not in item:
                response_content = item
            else:
                return normalized
            tool_response: dict[str, Any] = {
                "name": message["name"],
                "content": response_content,
            }
            if message.get("tool_call_id"):
                tool_response["id"] = message["tool_call_id"]
            return [{"type": "tool_response", "value": tool_response}]
        return normalized

    return normalized
