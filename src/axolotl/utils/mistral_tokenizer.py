"""Wrapper for MistralTokenizer from mistral-common"""

import os
from shutil import copyfile
from typing import TYPE_CHECKING, Optional

from huggingface_hub import hf_hub_download
from mistral_common.tokens.tokenizers.mistral import (
    MistralTokenizer as _MistralTokenizer,
)

if TYPE_CHECKING:
    from mistral_common.protocol.instruct.request import ChatCompletionRequest


def _get_file_path(path_or_repo_id: str, filename: str) -> str:
    """Get the file path from local or HF Hub"""
    if os.path.exists(path_or_repo_id):
        if os.path.exists(os.path.join(path_or_repo_id, filename)):
            return path_or_repo_id

        raise FileNotFoundError(f"File not found at {path_or_repo_id}")

    return hf_hub_download(repo_id=path_or_repo_id, filename=filename)


class HFMistralTokenizer:
    """
    Wraps mistral_common.tokens.tokenizers.mistral.MistralTokenizer
    and exposes HuggingFace API for special tokens.
    """

    def __init__(self, mistral: _MistralTokenizer, path_or_repo_id: str):
        """
        Args:
            tokenizer: The tokenizer to wrap.
            path: The path to the tokenizer files.
        """
        self._mistral = mistral
        self._padding_side = "right"
        self.chat_template = None
        self._tokenizer_path = _get_file_path(path_or_repo_id, "tekken.json")

        # Try to load system prompt if available
        try:
            self._system_prompt = self._load_system_prompt(
                path_or_repo_id=path_or_repo_id
            )
        except FileNotFoundError:
            pass

        # Make sure special tokens will be kept when decoding
        tokenizer_ = self._mistral.instruct_tokenizer.tokenizer
        from mistral_common.tokens.tokenizers.tekken import (
            SpecialTokenPolicy,
            Tekkenizer,
        )

        is_tekken = isinstance(tokenizer_, Tekkenizer)
        if is_tekken:
            tokenizer_._special_token_policy = SpecialTokenPolicy.KEEP  # type: ignore  # pylint: disable=protected-access
        else:
            raise NotImplementedError(f"Tokenizer {path_or_repo_id} not supported yet")

    def _load_system_prompt(self, path_or_repo_id: str) -> str:
        """Load system prompt from local or HF Hub"""
        file_path = _get_file_path(path_or_repo_id, "SYSTEM_PROMPT.txt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"System prompt file not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @property
    def bos_token_id(self) -> int:
        return self._mistral.instruct_tokenizer.tokenizer.bos_id

    @property
    def eos_token_id(self) -> int:
        return self._mistral.instruct_tokenizer.tokenizer.eos_id

    @property
    def pad_token_id(self) -> Optional[int]:
        if hasattr(self._mistral.instruct_tokenizer.tokenizer, "pad_id"):
            return self._mistral.instruct_tokenizer.tokenizer.pad_id
        return None

    @property
    def bos_token(self) -> str:
        return self._mistral.instruct_tokenizer.tokenizer.id_to_piece(self.bos_token_id)

    @property
    def eos_token(self) -> str:
        return self._mistral.instruct_tokenizer.tokenizer.id_to_piece(self.eos_token_id)

    @property
    def pad_token(self) -> Optional[str]:
        pid = self.pad_token_id
        return (
            self._mistral.instruct_tokenizer.tokenizer.id_to_piece(pid)
            if pid is not None
            else None
        )

    @property
    def padding_side(self) -> str:
        return self._padding_side

    def __len__(self) -> int:
        return self._mistral.instruct_tokenizer.tokenizer.n_words

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str,
        *,
        revision: Optional[str] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> "HFMistralTokenizer":
        """
        Download a mistral tokenizer from HF Hub and wrap it.
        """

        if revision:
            raise NotImplementedError("Revision not supported yet")

        # check if tokenizer_config is a valid local path
        base = _MistralTokenizer.from_file(
            _get_file_path(path_or_repo_id, "tekken.json")
        )
        return cls(base, path_or_repo_id=path_or_repo_id)

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the Tekken/SentencePiece model file so that from_pretrained can pick it up again.
        """
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer

        inner = self._mistral.instruct_tokenizer.tokenizer
        if isinstance(inner, Tekkenizer):
            # Create the directory and save the model
            os.makedirs(save_directory, exist_ok=True)
            copyfile(self._tokenizer_path, os.path.join(save_directory, "tekken.json"))

        raise RuntimeError(f"Unsupported tokenizer type: {type(inner)}")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._mistral.instruct_tokenizer.tokenizer.encode(
            text,
            bos=add_special_tokens,
            eos=add_special_tokens,
        )

    def decode(self, ids: int | list[int], skip_special_tokens: bool = True) -> str:
        if not skip_special_tokens:
            raise NotImplementedError("skip_special_tokens not supported yet")

        if isinstance(ids, int):
            return self.decode([ids])

        return self._mistral.instruct_tokenizer.tokenizer.decode(ids)

    def _create_mistral_chat_completion_request(
        self, conversation: list[dict], tools: list[dict] | None = None
    ) -> "ChatCompletionRequest":
        from mistral_common.protocol.instruct.messages import (
            AssistantMessage,
            SystemMessage,
            ToolMessage,
            UserMessage,
        )
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.protocol.instruct.tool_calls import Function, Tool

        messages: list[UserMessage | AssistantMessage | ToolMessage | SystemMessage] = (
            []
        )
        for turn in conversation:
            role = turn.get("role")

            if role == "user":
                messages.append(UserMessage(content=turn["content"]))
            elif role == "assistant":
                messages.append(
                    AssistantMessage(
                        content=turn.get("content"),
                        tool_calls=turn.get("tool_calls"),
                    )
                )
            elif role == "tool":
                messages.append(
                    ToolMessage(
                        content=turn.get("content"),
                        tool_call_id=turn.get("tool_call_id"),
                        name=turn.get("name"),
                    )
                )
            elif role == "system":
                messages.append(SystemMessage(content=turn["content"]))
            else:
                raise ValueError(
                    f"Unknown role for use with mistral-common tokenizer: {turn['role']}"
                )

        # set prefix to True for the last message if it is an assistant message
        if messages[-1].role == "assistant":
            messages[-1].prefix = True

        tool_calls: list[Tool] = []
        if tools:
            # convert to Tool
            for tool in tools:
                if tool["type"] != "function":
                    continue

                function = tool["function"]

                tool_calls.append(
                    Tool(
                        function=Function(
                            name=function["name"],
                            description=function["description"],
                            # set parameters to empty dict if not provided
                            parameters=function.get("parameters", {}),
                        )
                    )
                )

        chat_completion: ChatCompletionRequest = ChatCompletionRequest(
            messages=messages,
            tools=tool_calls,
        )

        return chat_completion

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = True,
        tools: list[dict] | None = None,
        chat_template: str | None = None,  # pylint: disable=unused-argument
        add_generation_prompt: bool = False,  # pylint: disable=unused-argument
    ) -> list[int] | str:
        if chat_template:
            raise NotImplementedError("chat_template not supported yet")

        if add_generation_prompt:
            raise NotImplementedError("add_generation_prompt not supported yet")

        chat_completion: ChatCompletionRequest = (
            self._create_mistral_chat_completion_request(messages, tools)
        )

        tokens: list[int] = self._mistral.encode_chat_completion(chat_completion).tokens

        if tokenize:
            return tokens

        return self._mistral.decode(tokens)
