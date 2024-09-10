from typing import Mapping


def chat_message_transform_builder(
    train_on_inputs=False,
    conversations_field: str = "conversations",
    role_fields: str | list[str] = "role",  # commonly ["role", "from"]
    message_field_content: str | list[str] = "content",  # also ["value", "text", "content"]
):
    """Builds a transform that takes a row from the dataset and converts it to a Chat

    Args:
        train_on_inputs (bool, optional):
            If True, the transform will train on the inputs. If False, the transform will train on the targets.
            Defaults to False.
        conversations_field (str, optional):
            The field name of the conversations. Defaults to "conversations".
        role_fields (str | list[str], optional):
            The field name of the role. Defaults to "role".
        message_field_content (str | list[str], optional):
            The field name of the message content. Defaults to "content".

    Returns:
        Callable:
            A function that takes a list of conversations and returns a list of messages.
    """

    role_fields = [role_fields] if isinstance(role_fields, str) else role_fields
    message_field_content = (
        [message_field_content] if isinstance(message_field_content, str) else message_field_content
    )

    role_value_mappings = {
        "system": "system",
        "user": "user",
        "human": "user",
        "assistant": "assistant",
        "gpt": "assistant",
    }

    def transform_builder(sample: Mapping[str, any]):
        if conversations_field not in sample:
            raise ValueError(f"Field '{conversations_field}' not found in sample.")
        for message in sample[conversations_field]:
            for role in role_fields:
                if role not in message:
                    raise ValueError(f"Field '{role}' not found in message.")
            for message_field in message_field_content:
                if message_field not in message:
                    raise ValueError(f"Field '{message_field}' not found in message.")
        messages = []
        for conversation in conversations:
            for role in role_fields:
                for message_field in message_field_content:
                    messages.extend(
                        [
                            message[message_field]
                            for message in conversation[conversations_field]
                            if message[role] == "user" if train_on_inputs
                        ]
                    )
                    messages.extend(
                        [
                            message[message_field]
                            for message in conversation[conversations_field]
                            if message[role] == "bot" if not train_on_inputs
                        ]
                    )
        return messages

    return transform_builder
