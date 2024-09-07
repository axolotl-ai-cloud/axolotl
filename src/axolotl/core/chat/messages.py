import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Union


class MessageRoles(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
    ipython = "ipython"  # for responses from builtin tools


class MessageContentTypes(str, Enum):
    text = "text"
    image = "image"
    audio = "audio"
    tool_call = "tool_call"  # to differentiate regular responses from tool calls from the assistant
    tool_response = "tool_response"


@dataclass
class ToolCallFunction:
    name: str
    arguments: dict[str, str]


@dataclass
class Tool:
    description: str
    function: ToolCallFunction
    parameters: dict[str, str]  # .properties


@dataclass
class ToolCallContents:
    name: str
    arguments: dict[str, Union[str, int]]
    id: Optional[str] = None

    def __str__(self) -> str:
        data = {"name": self.name, "arguments": self.arguments}
        if self.id is not None:
            data["id"] = self.id
        return json.dumps(data)


@dataclass
class MessageContents:
    type: Union[str, MessageContentTypes]
    value: Union[str, ToolCallContents]
    meta: Optional[dict[str, Any]] = None  # support additional arbitrary metadata
    train: bool = False
    eoc: bool = False  # end of contents

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class Messages:
    role: Union[MessageRoles, str]  # allows for arbitrary roles
    content: List["MessageContents"]
    meta: Optional[dict[str, Any]] = None  # support additional arbitrary metadata
    train: bool = False
    is_chat_formatted: bool = False

    def __str__(self) -> str:
        return "".join(str(c) for c in self.content)


@dataclass
class Chats:
    conversation: List[Messages]

    def __str__(self) -> str:
        return "".join(str(c) for c in self.conversation)


@dataclass
class ChatFormattedChats(Chats):
    formatter: Callable[[Chats], Chats]

    def __post_init__(self):
        for i, msg in enumerate(self.conversation):
            self.conversation[i] = self.formatter(msg)


@dataclass
class PreferenceChats:
    prompt: List[Messages]
    chosen: Messages
    rejected: Messages
