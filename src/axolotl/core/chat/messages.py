"""
internal message representations of chat messages
"""

import json
from enum import Enum
from typing import Any, Callable, List, Optional, Union

from pydantic import BaseModel
from transformers import PreTrainedTokenizer


class MessageRoles(str, Enum):
    """
    Message roles for the system, user, assistant, and tools
    """

    system = "system"  # pylint: disable=invalid-name
    user = "user"  # pylint: disable=invalid-name
    assistant = "assistant"  # pylint: disable=invalid-name
    tool = "tool"  # pylint: disable=invalid-name
    ipython = (  # pylint: disable=invalid-name
        # for responses from builtin tools
        "ipython"
    )


class MessageContentTypes(str, Enum):
    """
    Message content types for text, image, audio, tool calls, and tool responses
    """

    special_token = "special_token"  # pylint: disable=invalid-name  # nosec B105
    text = "text"  # pylint: disable=invalid-name
    image = "image"  # pylint: disable=invalid-name
    audio = "audio"  # pylint: disable=invalid-name
    tool_call = "tool_call"  # pylint: disable=invalid-name  # to differentiate regular responses from tool calls from the assistant
    tool_response = "tool_response"  # pylint: disable=invalid-name


class SpecialToken(str, Enum):
    """
    Special tokens for beginning of string and end of string
    """

    bos_token = "bos_token"  # pylint: disable=invalid-name  # nosec B105
    eos_token = "eos_token"  # pylint: disable=invalid-name  # nosec B105


class ToolCallFunction(BaseModel):
    """
    Tool call function with name and arguments
    """

    name: str
    arguments: dict[str, str]


class Tool(BaseModel):
    """
    Tool with description, function, and parameters
    """

    description: str
    function: ToolCallFunction
    parameters: dict[str, str]  # .properties


class ToolCallContents(BaseModel):
    """
    Tool call contents with name, arguments, and optional id
    """

    name: str
    arguments: dict[str, Union[str, int]]
    id: Optional[str] = None  # pylint: disable=invalid-name

    def __str__(self) -> str:
        data = {"name": self.name, "arguments": self.arguments}
        if self.id is not None:
            data["id"] = self.id
        return json.dumps(data)


class ToolResponseContents(BaseModel):
    """
    Tool response contents with name, content, and optional id
    """

    name: str
    content: Union[str, dict[str, Union[str, int, float]]]
    id: Optional[str] = None  # pylint: disable=invalid-name

    def __str__(self) -> str:
        data = {"name": self.name, "content": self.content}
        if self.id is not None:
            data["id"] = self.id
        return json.dumps(data)


class MessageContents(BaseModel):
    """
    Message contents with type, value, metadata, weight, newline, and end of contents
    """

    type: Union[str, MessageContentTypes]
    value: Union[str, ToolCallContents, ToolResponseContents, SpecialToken]
    meta: Optional[dict[str, Any]] = None  # support additional arbitrary metadata
    weight: Optional[Union[int, float]] = None
    has_newline: bool = False
    eoc: bool = False  # end of contents

    def __str__(self) -> str:
        str_val = str(self.value)
        if self.has_newline and not str_val.endswith("\n"):
            str_val += "\n"
        return str_val


class Messages(BaseModel):
    """
    Messages with role, content, metadata, weight, and chat formatting
    """

    role: Union[MessageRoles, str]  # allows for arbitrary roles
    content: List["MessageContents"]
    meta: Optional[dict[str, Any]] = None  # support additional arbitrary metadata
    weight: Optional[Union[int, float]] = None
    is_chat_formatted: bool = False

    def __str__(self) -> str:
        return "".join(str(c) for c in self.content)

    def tokenized(
        self, tokenizer: PreTrainedTokenizer, ignore_index=-100
    ) -> dict[str, List[int]]:
        # iterate over the contents, tokenizing the concatenated string values up to the current MessageContents
        # returns a dictionary mapping w input_ids, attention_mask, and labels
        input_ids: List[int] = []
        labels: List[int] = []
        pending_input_ids: List[int] = []
        pending_weight = self.weight
        running_content = ""
        for _, msg_content in enumerate(self.content):
            # TODO also handle non-text content types
            if msg_content.type in [
                MessageContentTypes.text.value,
                MessageContentTypes.tool_call.value,
                MessageContentTypes.tool_response.value,
            ]:
                running_content += str(msg_content)
                tok_results = tokenizer(running_content, add_special_tokens=False)
                tok_input_ids = tok_results["input_ids"]
                if pending_input_ids:
                    new_pending_inputs = tok_input_ids[
                        len(input_ids) : len(input_ids) + len(pending_input_ids)
                    ]
                    if new_pending_inputs != pending_input_ids:
                        # logging.warning("tokenization mismatch from concatenation.")
                        pending_input_ids = new_pending_inputs
                    input_ids.extend(pending_input_ids)
                    if pending_weight:
                        labels.extend(pending_input_ids)
                    else:
                        labels.extend([ignore_index] * len(pending_input_ids))
                pending_input_ids = tok_results["input_ids"][len(input_ids) :]
                pending_weight = self.weight and msg_content.weight not in [0, 0.0]
        input_ids.extend(pending_input_ids)
        if pending_weight:
            labels.extend(pending_input_ids)
        else:
            labels.extend([ignore_index] * len(pending_input_ids))
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class Chats(BaseModel):
    """
    top level data structure for chat conversations
    """

    conversation: List[Messages]

    def __str__(self) -> str:
        return "".join(str(c) for c in self.conversation)

    def tokenized(
        self, tokenizer: Callable[[str], dict[str, List[int]]], ignore_index=-100
    ) -> dict[str, List[int]]:
        input_ids = []
        attention_mask = []
        labels = []
        for msg in self.conversation:
            msg_results = msg.tokenized(tokenizer, ignore_index)
            input_ids.extend(msg_results["input_ids"])
            attention_mask.extend(msg_results["attention_mask"])
            labels.extend(msg_results["labels"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ChatFormattedChats(Chats):
    """
    Chat formatted chats with formatter and optional train on inputs
    """

    formatter: Callable  # [[Union[dict, Chats]], Chats]
    train_on_inputs: bool = False

    def model_post_init(self, __context):
        for i, msg in enumerate(self.conversation):
            self.conversation[i] = self.formatter(msg, message_index=i)
            if self.train_on_inputs:
                self.conversation[i].weight = 1


class PreferenceChats(BaseModel):
    """
    representation for preference data for chat
    """

    prompt: List[Messages]
    chosen: Messages
    rejected: Messages
