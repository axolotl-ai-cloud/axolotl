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

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
    ipython = (
        # for responses from builtin tools
        "ipython"
    )


class MessageContentTypes(str, Enum):
    """
    Message content types for text, image, audio, tool calls, and tool responses
    """

    special_token = "special_token"  # nosec B105
    text = "text"
    image = "image"
    audio = "audio"
    tool_call = "tool_call"
    tool_response = "tool_response"


class SpecialToken(str, Enum):
    """
    Special tokens for beginning of string and end of string
    """

    bos_token = "bos_token"  # nosec B105
    eos_token = "eos_token"  # nosec B105


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
    id: Optional[str] = None

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
    id: Optional[str] = None

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
        # tokenize the concatenated content values once, attributing tokens to
        # their content item by char offset; avoids re-tokenizing the running
        # string per item (O(n) instead of O(n^2)) and guarantees the output
        # equals the tokenization of the final string
        content_str = ""
        spans: List[tuple[int, int, MessageContents]] = []
        for msg_content in self.content:
            # TODO also handle non-text content types
            if msg_content.type in [
                MessageContentTypes.text.value,
                MessageContentTypes.tool_call.value,
                MessageContentTypes.tool_response.value,
            ]:
                start = len(content_str)
                content_str += str(msg_content)
                spans.append((start, len(content_str), msg_content))
        if not spans:
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
        tok_results = tokenizer(
            content_str, add_special_tokens=False, return_offsets_mapping=True
        )
        input_ids = tok_results["input_ids"]
        offsets = tok_results["offset_mapping"]
        labels: List[int] = []
        item_idx = 0
        for token_id, (_, end) in zip(input_ids, offsets, strict=True):
            while item_idx < len(spans) and end > spans[item_idx][1]:
                item_idx += 1
            msg_content = spans[item_idx][2]
            if self.weight and msg_content.weight not in [0, 0.0]:
                labels.append(token_id)
            else:
                labels.append(ignore_index)
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
