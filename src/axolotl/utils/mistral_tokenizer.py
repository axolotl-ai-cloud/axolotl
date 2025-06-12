"""Wrapper for MistralTokenizer from mistral-common"""

import math
import os
from shutil import copyfile
from typing import TYPE_CHECKING, Optional

import numpy as np
from huggingface_hub import hf_hub_download
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import Tekkenizer
from torch import Tensor
from transformers.utils import PaddingStrategy

from axolotl.utils.collators.core import IGNORE_INDEX

if TYPE_CHECKING:
    from mistral_common.protocol.instruct.request import ChatCompletionRequest


def _get_file_path(path_or_repo_id: str, filename: str) -> str:
    """Get the file path from local or HF Hub"""
    if os.path.exists(path_or_repo_id):
        maybe_file_path = os.path.join(path_or_repo_id, filename)
        if os.path.exists(maybe_file_path):
            return maybe_file_path

        raise FileNotFoundError(f"File not found at {path_or_repo_id}")

    return hf_hub_download(repo_id=path_or_repo_id, filename=filename)


class HFMistralTokenizer:
    """
    Wraps mistral_common.tokens.tokenizers.mistral.MistralTokenizer
    and exposes HuggingFace API for special tokens.
    """

    def __init__(
        self, mistral: MistralTokenizer, name_or_path: str, tokenizer_path: str
    ):
        """
        Args:
            mistral: The mistral-common tokenizer to wrap.
            name_or_path: The name or path to the tokenizer files or the repo id.
        """
        self._mistral = mistral
        self._padding_side = "right"
        self._name_or_path = name_or_path
        self._tokenizer_path = tokenizer_path

        # Manual set to training mode
        from mistral_common.protocol.instruct.validator import (
            MistralRequestValidator,
            ValidationMode,
        )

        # Check if MistralRequestValidator has a _mode attribute.
        # This is a private API and may change in the future.
        # pylint: disable=protected-access
        if not (
            hasattr(self._mistral, "_chat_completion_request_validator")
            and isinstance(
                self._mistral._chat_completion_request_validator,
                MistralRequestValidator,
            )
            and hasattr(self._mistral._chat_completion_request_validator, "_mode")
        ):
            raise RuntimeError(
                "Unable to switch mistral tokenizer to finetuning mode – "
                "private API `_chat_completion_request_validator._mode` missing."
            )

        self._mistral._chat_completion_request_validator._mode = (
            ValidationMode.finetuning
        )

    def _load_system_prompt(self, path_or_repo_id: str) -> str:
        """Load system prompt from local or HF Hub.

        Note: Unused for now as we don't want to explicitly set the system prompt if a user does
        not provide one.

        Args:
            path_or_repo_id: The path to the tokenizer files or the repo id.

        Returns:
            The system prompt.
        """
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
    def pad_token_id(self) -> int:
        return self._mistral.instruct_tokenizer.tokenizer.pad_id

    @property
    def unk_token_id(self) -> int:
        return self._mistral.instruct_tokenizer.tokenizer.unk_id

    @property
    def bos_token(self) -> str:
        return self._mistral.instruct_tokenizer.tokenizer.id_to_piece(self.bos_token_id)

    @property
    def eos_token(self) -> str:
        return self._mistral.instruct_tokenizer.tokenizer.id_to_piece(self.eos_token_id)

    @property
    def pad_token(self) -> str:
        return self._mistral.instruct_tokenizer.tokenizer.id_to_piece(self.pad_token_id)

    @property
    def unk_token(self) -> str:
        return self._mistral.instruct_tokenizer.tokenizer.id_to_piece(self.unk_token_id)

    @property
    def padding_side(self) -> str:
        return self._padding_side

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    @property
    def chat_template(self) -> str | None:
        """Chat template is not supported. Dummy method to satisfy HuggingFace API."""
        return None

    def __len__(self) -> int:
        return self._mistral.instruct_tokenizer.tokenizer.n_words

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        revision: Optional[str] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> "HFMistralTokenizer":
        """
        Load a mistral tekken tokenizer from a local file or HF Hub and wrap it.

        Args:
            path_or_repo_id: The path to the tokenizer files or the repo id.
            revision: The revision of the tokenizer to download.
            kwargs: Additional keyword arguments.

        Returns:
            A HFMistralTokenizer instance.
        """
        if revision:
            raise NotImplementedError(
                "Revision not supported yet for mistral-common tokenizer"
            )

        # only support Tekken tokenizer for now
        # downloads from HF Hub if not local
        tokenizer_path = _get_file_path(name_or_path, "tekken.json")

        base = MistralTokenizer.from_file(tokenizer_path)

        return cls(
            base,
            name_or_path=name_or_path,
            tokenizer_path=tokenizer_path,
        )

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the Tekken/SentencePiece model file so that from_pretrained can pick it up again.

        Only Tekken models are supported.

        Args:
            save_directory: The directory to save the tokenizer files.
        """
        inner = self._mistral.instruct_tokenizer.tokenizer
        if isinstance(inner, Tekkenizer):
            # Create the directory and save the model
            try:
                os.makedirs(save_directory, exist_ok=True)

                # Verify directory was created
                if not os.path.exists(save_directory):
                    raise RuntimeError(f"Failed to create directory: {save_directory}")

                # Verify source file exists
                if not os.path.exists(self._tokenizer_path):
                    raise FileNotFoundError(
                        f"Source tokenizer file not found: {self._tokenizer_path}"
                    )

                destination_path = os.path.join(save_directory, "tekken.json")
                copyfile(self._tokenizer_path, destination_path)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to save tokenizer to {save_directory}: {e}. "
                    f"Source path: {self._tokenizer_path}, "
                    f"Directory exists: {os.path.exists(save_directory)}"
                ) from e

        else:
            raise RuntimeError(f"Unknown tokenizer type: {type(inner)}")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode a text string into a list of token IDs.

        Args:
            text: The text string to encode.
            add_special_tokens: Whether to add special tokens to the encoded tokens.

        Returns:
            A list of token IDs.
        """
        return self._mistral.instruct_tokenizer.tokenizer.encode(
            text,
            bos=add_special_tokens,
            eos=add_special_tokens,
        )

    def decode(
        self, token_ids: int | list[int], skip_special_tokens: bool = False
    ) -> str:
        """
        Decode a list of token IDs into a text string.

        Args:
            token_ids: The int or list of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in the decoded text.

        Returns:
            The decoded text string.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if skip_special_tokens:
            return self._mistral.instruct_tokenizer.tokenizer.decode(token_ids)

        # to_string returns a string with special tokens
        return self._mistral.instruct_tokenizer.tokenizer.to_string(token_ids)

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

        return self.decode(tokens)

    def pad(
        self,
        features: list[dict[str, list[int] | np.ndarray]],
        *,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = None,  # "np", "pt", or "tf"
    ) -> dict[str, np.ndarray | Tensor]:
        """
        HF-style pad method that properly handles all sequence-related features:
        - pad 'input_ids' & 'labels' to the longest (or to max_length)
        """
        import torch
        from torch.nn import functional as F

        # Check for unsupported fields
        if any("token_type_ids" in f for f in features):
            raise ValueError("token_type_ids is not supported by this tokenizer")

        # Determine desired sequence length
        lengths = [len(f["input_ids"]) for f in features]
        if padding in (True, "longest", PaddingStrategy.LONGEST):
            target_length = max(lengths)
        elif padding in ("max_length", PaddingStrategy.MAX_LENGTH):
            if max_length is None:
                raise ValueError("max_length must be set for 'max_length' padding")
            target_length = max_length
        elif padding in (False, "do_not_pad", PaddingStrategy.DO_NOT_PAD):
            target_length = None
        else:
            raise ValueError(f"Unknown padding strategy: {padding}")

        # Apply pad_to_multiple_of
        if target_length is not None and pad_to_multiple_of is not None:
            target_length = (
                math.ceil(target_length / pad_to_multiple_of) * pad_to_multiple_of
            )

        # If no padding requested, just stack tensors
        do_pad = target_length is not None

        # Pad sequences using torch.nn.utils.rnn.pad_sequence
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["input_ids"], dtype=torch.long) for x in features],
            batch_first=True,
            padding_value=self.pad_token_id if self.pad_token_id is not None else 0,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["labels"], dtype=torch.long) for x in features],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["attention_mask"], dtype=torch.long) for x in features],
            batch_first=True,
            padding_value=0,
        )

        # Handle position_ids - pad with sequential values for right padding, 0s for left padding
        if "position_ids" in features[0]:
            if self.padding_side == "left":
                # Likely not needed, but keeping for now
                # For left padding, we'll pad with 0s using pad_sequence, then handle manually
                position_ids = torch.nn.utils.rnn.pad_sequence(
                    [
                        torch.tensor(x["position_ids"], dtype=torch.long)
                        for x in features
                    ],
                    batch_first=True,
                    padding_value=0,
                )
            else:
                # For right padding, continue the sequence
                max_pos_len = max(len(f["position_ids"]) for f in features)
                position_ids_list = []
                for f in features:
                    pos_seq = torch.tensor(f["position_ids"], dtype=torch.long)
                    if len(pos_seq) < max_pos_len:
                        # Continue the sequence
                        last_pos = pos_seq[-1].item() if len(pos_seq) > 0 else -1
                        pad_len = max_pos_len - len(pos_seq)
                        pad_positions = torch.arange(
                            last_pos + 1, last_pos + 1 + pad_len, dtype=torch.long
                        )
                        pos_seq = torch.cat([pos_seq, pad_positions])
                    position_ids_list.append(pos_seq)
                position_ids = torch.stack(position_ids_list)
        else:
            # Create position_ids if not present
            seq_len = input_ids.size(1)
            position_ids = (
                torch.arange(seq_len, dtype=torch.long)
                .unsqueeze(0)
                .expand(input_ids.size(0), -1)
            )

        # Ensure all tensors have the same sequence length
        max_seq_len = max(
            input_ids.size(1),
            labels.size(1),
            attention_mask.size(1),
            position_ids.size(1),
        )

        # TODO: check if trimming is needed? and correct.

        if do_pad and target_length is not None:
            max_seq_len = target_length

        # Pad all tensors to the same length
        if input_ids.size(1) < max_seq_len:
            pad_len = max_seq_len - input_ids.size(1)
            if self.padding_side == "right":
                input_ids = F.pad(
                    input_ids,
                    (0, pad_len),
                    value=self.pad_token_id if self.pad_token_id is not None else 0,
                )
            else:
                input_ids = F.pad(
                    input_ids,
                    (pad_len, 0),
                    value=self.pad_token_id if self.pad_token_id is not None else 0,
                )
        elif input_ids.size(1) > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]

        if labels.size(1) < max_seq_len:
            pad_len = max_seq_len - labels.size(1)
            if self.padding_side == "right":
                labels = F.pad(labels, (0, pad_len), value=IGNORE_INDEX)
            else:
                labels = F.pad(labels, (pad_len, 0), value=IGNORE_INDEX)
        elif labels.size(1) > max_seq_len:
            labels = labels[:, :max_seq_len]

        if attention_mask.size(1) < max_seq_len:
            pad_len = max_seq_len - attention_mask.size(1)
            if self.padding_side == "right":
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            else:
                attention_mask = F.pad(attention_mask, (pad_len, 0), value=0)
        elif attention_mask.size(1) > max_seq_len:
            attention_mask = attention_mask[:, :max_seq_len]

        if position_ids.size(1) < max_seq_len:
            pad_len = max_seq_len - position_ids.size(1)
            if self.padding_side == "right":
                batch_size = position_ids.size(0)
                new_position_ids = []
                for i in range(batch_size):
                    seq = position_ids[i]
                    if len(seq) > 0:
                        # get last position and pad with sequential values
                        last_pos = seq[-1].item()
                        pad_positions = torch.arange(
                            last_pos + 1, last_pos + 1 + pad_len, dtype=torch.long
                        )
                        new_seq = torch.cat([seq, pad_positions])
                    else:
                        new_seq = torch.arange(pad_len, dtype=torch.long)
                    new_position_ids.append(new_seq)
                position_ids = torch.stack(new_position_ids)
            else:
                position_ids = F.pad(position_ids, (pad_len, 0), value=0)
        elif position_ids.size(1) > max_seq_len:
            position_ids = position_ids[:, :max_seq_len]

        final_batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        # Handle non-sequence fields (raise error)
        sequence_fields = {"input_ids", "labels", "attention_mask", "position_ids"}
        for f in features:
            for key in f.keys():
                if key not in sequence_fields:
                    raise NotImplementedError(
                        f"Non-sequence field {key} not handled yet"
                    )

        # Convert to requested tensor type
        if return_tensors is None or return_tensors == "np":
            result = {}
            for k, v in final_batch.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.numpy().astype(np.long)
                else:
                    result[k] = v
            return result

        if return_tensors == "pt":
            return final_batch

        raise ValueError(f"Unsupported return_tensors='{return_tensors}'")

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """
        Convert a list of token IDs to a list of tokens.

        Args:
            ids: The list of token IDs to convert.

        Returns:
            The list of tokens.
        """
        return [
            self._mistral.instruct_tokenizer.tokenizer.id_to_piece(id) for id in ids
        ]
