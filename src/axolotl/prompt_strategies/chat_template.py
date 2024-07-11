"""
HF Chat Templates prompt strategy
"""

import logging
from typing import Any, Dict, List, Optional

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import Prompter
from axolotl.utils.chat_templates import chat_templates

LOG = logging.getLogger("axolotl")


class ChatTemplatePrompter(Prompter):
    """prompter for HF chat templates"""

    def __init__(
        self,
        tokenizer,
        chat_template=None,
        max_length=2048,
        message_field_role: str = "from",
        message_field_content: str = "value",
        roles: Optional[Dict[str, List[str]]] = None,
        drop_system_message: bool = False,
    ):
        if roles:
            self.roles = {s: t for t, sources in roles.items() for s in sources}
        else:
            self.roles = {
                "human": "user",
                "user": "user",
                "assistant": "assistant",
                "gpt": "assistant",
                "system": "system",
            }
        self.message_field_role = message_field_role
        self.message_field_content = message_field_content
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length
        self.drop_system_message = drop_system_message

    def build_prompt(self, conversation, add_generation_prompt=False):
        turns = [
            {
                "role": self.roles[t[self.message_field_role]],
                "content": t[self.message_field_content],
            }
            for t in conversation
        ]

        if self.drop_system_message and turns[0]["role"] == "system":
            turns = turns[1:]

        return self.tokenizer.apply_chat_template(
            turns,
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=add_generation_prompt,
            chat_template=self.chat_template,
        )


class ChatTemplateStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for instruction-based prompts.
    """

    _messages = "conversations"

    @property
    def messages(self):
        return self._messages

    @messages.setter
    def messages(self, messages):
        self._messages = messages

    def tokenize_prompt(self, prompt):
        turns = self.get_conversation_thread(prompt)
        prompt_ids = self.prompter.build_prompt(turns[:-1], add_generation_prompt=True)
        input_ids = self.prompter.build_prompt(turns)

        if not self.train_on_inputs:
            user_prompt_len = len(prompt_ids)
            labels = [-100] * user_prompt_len + input_ids[user_prompt_len:]
        else:
            labels = input_ids

        tokenized_prompt = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

        return tokenized_prompt

    def get_conversation_thread(self, prompt):
        return prompt[self.messages]


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    chat_template = (
        ds_cfg["chat_template"] if ds_cfg and "chat_template" in ds_cfg else "chatml"
    )
    message_field_role = (
        ds_cfg["message_field_role"]
        if ds_cfg and "message_field_role" in ds_cfg
        else "from"
    )
    message_field_content = (
        ds_cfg["message_field_content"]
        if ds_cfg and "message_field_content" in ds_cfg
        else "value"
    )
    roles = ds_cfg["roles"] if ds_cfg and "roles" in ds_cfg else None
    drop_system_message = (
        ds_cfg["drop_system_message"]
        if ds_cfg and "drop_system_message" in ds_cfg
        else False
    )

    strategy = ChatTemplateStrategy(
        ChatTemplatePrompter(
            tokenizer,
            chat_templates(chat_template),
            message_field_role=message_field_role,
            message_field_content=message_field_content,
            roles=roles,
            drop_system_message=drop_system_message,
        ),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    if ds_cfg and "field_messages" in ds_cfg and hasattr(strategy, "messages"):
        strategy.messages = ds_cfg["field_messages"]
    return strategy
