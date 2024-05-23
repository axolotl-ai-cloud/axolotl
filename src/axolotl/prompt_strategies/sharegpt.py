"""Module containing the SimpleShareGPTPromptTokenizingStrategy class"""

import logging
from typing import Any, Dict, Optional, Type

from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

from axolotl.prompt_tokenizers import ShareGPTPromptTokenizingStrategy
from axolotl.prompters import ShareGPTPrompterV2
from axolotl.utils.tokenization import (
    chatml_to_conversation,
    merge_consecutive_messages,
)

LOG = logging.getLogger("axolotl")


def register_chatml_template(system_message=None):
    system_message = system_message or "You are a helpful assistant."
    register_conv_template(
        Conversation(
            name="chatml",
            system_template="<|im_start|>system\n{system_message}",
            system_message=system_message,
            roles=("<|im_start|>user", "<|im_start|>assistant"),
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
        )
    )
    register_conv_template(
        Conversation(
            name="chatml_glaive",
            system_template="<|im_start|>system\n{system_message}",
            system_message=system_message,
            roles=("<|im_start|>user", "<|im_start|>assistant", "<|im_start|>tool"),
            sep_style=SeparatorStyle.CHATML,
            sep="<|im_end|>",
        )
    )


def register_llama3_template(system_message=None):
    system_message = system_message or "You are a helpful assistant."
    register_conv_template(
        Conversation(
            name="llama3",
            system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
            system_message=system_message,
            roles=("user", "assistant"),
            sep_style=SeparatorStyle.LLAMA3,
            sep="",
            stop_str="<|eot_id|>",
            stop_token_ids=[128001, 128009],
        )
    )


def build_loader(
    tokenization_strategy_cls: Type["ShareGPTPromptTokenizingStrategy"],
    prompter_cls: Type["ShareGPTPrompterV2"],
    default_conversation: Optional[str] = None,
):
    def _load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
        conversation = (
            ds_cfg["conversation"]
            if ds_cfg and "conversation" in ds_cfg
            else default_conversation
        )
        field_human = (
            ds_cfg["field_human"] if ds_cfg and "field_human" in ds_cfg else None
        )
        field_model = (
            ds_cfg["field_model"] if ds_cfg and "field_model" in ds_cfg else None
        )
        roles = ds_cfg["roles"].to_dict() if ds_cfg and "roles" in ds_cfg else None
        strategy = tokenization_strategy_cls(
            prompter_cls(
                conversation=conversation,
                role_key_model=field_model,
                role_key_human=field_human,
                roles=roles,
            ),
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
        if ds_cfg and "strict" in ds_cfg and hasattr(strategy, "strict"):
            strategy.strict = ds_cfg["strict"]
        if ds_cfg and "field_messages" in ds_cfg and hasattr(strategy, "messages"):
            strategy.messages = ds_cfg["field_messages"]
        return strategy

    return _load


class SimpleShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
    """
    basic sharegpt strategy to grab conversations from the sample row
    """

    _strict = False
    _messages = "conversations"

    @property
    def strict(self):
        return self._strict

    @strict.setter
    def strict(self, strict):
        self._strict = strict

    @property
    def messages(self):
        return self._messages

    @messages.setter
    def messages(self, messages):
        self._messages = messages

    def get_conversation_thread(self, prompt):
        conversations = prompt[self.messages]
        if self.strict:
            return conversations
        role_key = "from"
        if "role" in conversations[0].keys():
            role_key = "role"
        value_key = "value"
        if "text" in conversations[0].keys():
            value_key = "text"
        elif "content" in conversations[0].keys():
            value_key = "content"
        # remap roles - allow for assistant turn"
        role_map = {
            "user": "human",
            "human": "human",
            "assistant": "gpt",
            "gpt": "gpt",
            "system": "system",
        }
        turns = [
            {
                "from": (
                    role_map[t[role_key]] if t[role_key] in role_map else t[role_key]
                ),
                "value": t[value_key],
            }
            for t in conversations
        ]
        return turns


class SimpleRoleShareGPTPromptTokenizingStrategy(
    SimpleShareGPTPromptTokenizingStrategy
):
    """
    basic sharegpt strategy to grab conversations from the sample row, but uses role instead of from
    """

    def get_conversation_thread(self, prompt):
        conversations = prompt["conversations"]
        # remap role: prompter/assistant, text: ... => from: human/gpt, value: ...
        turns = [{"from": t["role"], "value": t["value"]} for t in conversations]
        return turns


class GuanacoShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
    """
    sharegpt strategy that remaps oasst data to sharegpt format
    """

    def get_conversation_thread(self, prompt):
        conversations = prompt["conversations"]
        # remap role: prompter/assistant, text: ... => from: human/gpt, value: ...
        role_map = {"prompter": "human", "assistant": "gpt"}
        turns = [
            {"from": role_map[t["role"]], "value": t["text"]} for t in conversations
        ]
        return turns


class UltrachatShareGPTPromptTokenizingStrategy(SimpleShareGPTPromptTokenizingStrategy):
    """
    sharegpt strategy that remaps ultrachat data to sharegpt format
    """

    def get_conversation_thread(self, prompt):
        conversations = prompt["messages"]
        role_map = {"user": "human", "assistant": "gpt"}
        turns = [
            {"from": role_map[t["role"]], "value": t["content"]} for t in conversations
        ]
        return turns


class GlaiveShareGPTPromptTokenizingStrategy(SimpleShareGPTPromptTokenizingStrategy):
    """
    sharegpt strategy that remaps glaive data to sharegpt format
    """

    def get_conversation_thread(self, prompt):
        conversation = chatml_to_conversation(prompt)
        conversation = merge_consecutive_messages(conversation)

        return conversation


load = build_loader(SimpleShareGPTPromptTokenizingStrategy, ShareGPTPrompterV2)
load_role = build_loader(SimpleRoleShareGPTPromptTokenizingStrategy, ShareGPTPrompterV2)
load_ultrachat = build_loader(
    UltrachatShareGPTPromptTokenizingStrategy, ShareGPTPrompterV2
)
load_guanaco = build_loader(GuanacoShareGPTPromptTokenizingStrategy, ShareGPTPrompterV2)
load_glaive = build_loader(
    GlaiveShareGPTPromptTokenizingStrategy,
    ShareGPTPrompterV2,
    default_conversation="chatml_glaive",
)
