"""
Llama 3.x chat formatting functions for MessageContents
"""

from typing import Optional

from ..messages import MessageContents, Messages
from .shared import wrap_tools


def format_message(message: Messages, message_index: Optional[int] = None) -> Messages:
    if message.is_chat_formatted:
        return message

    message_role = message.role
    if message.role == "tool":
        message_role = "ipython"

    # prepend the role prefix within a MessageContents to message.content
    message.content.insert(
        0,
        MessageContents(
            type="text",
            value=f"<|start_header_id|>{message_role}<|end_header_id|>\n\n",
            weight=0,
        ),
    )

    message.content.append(
        MessageContents(type="text", value="<|eot_id|>", weight=message.weight)
    )

    message = wrap_tools(message)

    if message_index == 0:
        message.content.insert(
            0,
            MessageContents(
                type="text",
                value="<|begin_of_text|>",
                weight=0,
            ),
        )

    message.is_chat_formatted = True
    return message
