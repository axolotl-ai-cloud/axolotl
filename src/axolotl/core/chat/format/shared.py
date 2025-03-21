"""
shared functions for format transforms
"""

from axolotl.core.chat.messages import MessageContents, Messages


def wrap_tools(message: Messages):
    # loop over message.content by index to find tool calls, we need to wrap each with tags,
    # so be wary of indexing issues when changing the list while iterating.
    # iterate over the range in reverse order to avoid index shifting
    for i in range(len(message.content) - 1, -1, -1):
        if message.content[i].type == "tool_call":
            # append a </tool_call> MessageContents text tag after
            message.content.insert(
                i + 1,
                MessageContents(
                    type="text", value="</tool_call>\n", weight=message.weight
                ),
            )
            # make sure the actual tool call content ends with a newline
            message.content[i].has_newline = True
            # prepend a <tool_call> MessageContents text tag before
            message.content.insert(
                i,
                MessageContents(
                    type="text", value="<tool_call>\n", weight=message.weight
                ),
            )
        elif message.content[i].type == "tool_response":
            # append a </tool_call> MessageContents text tag after
            message.content.insert(
                i + 1,
                MessageContents(
                    type="text", value="</tool_response>\n", weight=message.weight
                ),
            )
            # make sure the actual tool response content ends with a newline
            message.content[i].has_newline = True
            # prepend a <tool_call> MessageContents text tag before
            message.content.insert(
                i,
                MessageContents(
                    type="text", value="<tool_response>\n", weight=message.weight
                ),
            )

    return message
