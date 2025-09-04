"""
This module contains a function that builds a transform that takes a row from the dataset and converts it to a Chat.
"""

from typing import Any, Mapping, Union


def chat_message_transform_builder(  # pylint: disable=dangerous-default-value
    train_on_inputs=False,
    conversations_field: str = "conversations",
    message_field_role: Union[str, list[str]] = ["role", "from"],  # commonly "role"
    message_field_content: Union[str, list[str]] = [
        "value",
        "text",
        "content",
    ],  # commonly "content"
    message_field_training: Union[str, list[str]] = [
        "train",
        "weight",
    ],  # commonly "weight"
):
    """Builds a transform that takes a row from the dataset and converts it to a Chat

    Args:
        train_on_inputs (bool, optional):
            If True, the transform will train on the inputs. If False, the transform will train on the targets.
            Defaults to False.
        conversations_field (str, optional):
            The field name of the conversations. Defaults to "conversations".
        message_field_role (str | list[str], optional):
            The field name of the role. Defaults to "role".
        message_field_content (str | list[str], optional):
            The field name of the message content. Defaults to "content".
        message_field_training (str | list[str], optional):
            The field name of the train/weight. Defaults to "weight".

    Returns:
        Callable:
            A function that takes a list of conversations and returns a list of messages.
    """

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
        if not any(
            field in sample[conversations_field][0] for field in message_field_content
        ):
            raise ValueError("No message_content field found in message.")
        message_content_field = next(
            field
            for field in message_field_content
            if field in sample[conversations_field][0]
        )
        if not any(
            field in sample[conversations_field][0] for field in message_field_training
        ):
            message_weight_field = None
        else:
            message_weight_field = next(
                field
                for field in message_weight_fields
                if field in sample[conversations_field][0]
            )

        messages = []
        for message in sample[conversations_field]:
            role = role_value_mappings[message[role_field]]
            weight = (
                int(message[message_weight_field])
                if message_weight_field
                else role_default_weights_mappings[role]
            )

            # TODO if "tool_calls" in message[message_content_field]: then convert tool call to ToolCallContents
            if isinstance(message[message_content_field], str):
                messages.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "text",
                                "value": message[message_content_field],
                            }
                        ],
                        "weight": weight,
                    }
                )
            else:
                messages.append(
                    {
                        "role": role,
                        "content": message[message_content_field],
                        "weight": weight,
                    }
                )

        return {"conversation": messages}

    return transform_builder
