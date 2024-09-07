from ..messages import Messages, MessageContents


def format_message(message: Messages) -> Messages:
    if message.is_chat_formatted:
        return message

    # prepend the role prefix within a MessageContents to message.content
    message.content.insert(0, MessageContents(type="text", value=f"<|im_start|>{message.role}\n", train=False))
    message.content.append(MessageContents(type="text", value="<|im_end|>\n", train=message.train))

     # loop over message.content by index to find tool calls, we need to wrap each with tags,
    # so be wary of indexing issues when changing the list while iterating
    # iterate over the range in reverse order to avoid index shifting
    for i in range(len(message.content) - 1, -1, -1):
        if message.content[i].type == "tool_call":
            # append a </tool_call> MessageContents text tag after
            message.content.insert(i + 1, MessageContents(type="text", value="</tool_call>", train=message.train))
            # prepend a <tool_call> MessageContents text tag before
            message.content.insert(i, MessageContents(type="text", value="<tool_call>"))

    message.is_chat_formatted = True
    return message
