"""
ChatML transformation functions for MessageContents
"""

from typing import Optional

from ..messages import MessageContents, Messages
from .shared import wrap_tools


def format_message(
    message: Messages,
    message_index: Optional[int] = None,  # pylint: disable=unused-argument
) -> Messages:
    if message.is_chat_formatted:
        return message

    # prepend the role prefix within a MessageContents to message.content
    message.content.insert(
        0,
        MessageContents(
            type="text",
            value=f"<|im_start|>{message.role}\n",
            weight=0,
        ),
    )
    message.content.append(
        MessageContents(type="text", value="<|im_end|>", weight=message.weight)
    )
    message.content.append(MessageContents(type="text", value="\n", weight=0))

    message = wrap_tools(message)

    message.is_chat_formatted = True
    return message
