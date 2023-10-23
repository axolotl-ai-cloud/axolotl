"""Module containing the SimpleShareGPTPromptTokenizingStrategy class"""
from typing import Any, Dict, Optional

from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

from axolotl.prompt_tokenizers import ShareGPTPromptTokenizingStrategy
from axolotl.prompters import ShareGPTPrompterV2

register_conv_template(
    Conversation(
        name="chatml",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=["<|im_start|>user", "<|im_start|>assistant"],
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>\n",
    )
)


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    conversation = (
        ds_cfg["conversation"] if ds_cfg and "conversation" in ds_cfg else None
    )
    field_human = ds_cfg["field_human"] if ds_cfg and "field_human" in ds_cfg else None
    field_model = ds_cfg["field_model"] if ds_cfg and "field_model" in ds_cfg else None
    strategy = SimpleShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2(
            conversation=conversation,
            role_key_model=field_model,
            role_key_human=field_human,
        ),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    if ds_cfg and "strict" in ds_cfg:
        strategy.strict = ds_cfg["strict"]
    return strategy


def load_role(tokenizer, cfg):
    return SimpleRoleShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_guanaco(tokenizer, cfg):
    return GuanacoShareGPTPromptTokenizingStrategy(
        ShareGPTPrompterV2(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


class SimpleShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
    """
    basic sharegpt strategy to grab conversations from the sample row
    """

    _strict = True

    @property
    def strict(self):
        return self._strict

    @strict.setter
    def strict(self, strict):
        self._strict = strict

    def get_conversation_thread(self, prompt):
        conversations = prompt["conversations"]
        if self.strict:
            return conversations
        # remap roles - allow for assistant turn
        role_map = {"human": "human", "assistant": "gpt", "gpt": "gpt"}
        turns = [
            {"from": role_map[t["from"]], "value": t["value"]} for t in conversations
        ]
        return turns


class SimpleRoleShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
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
