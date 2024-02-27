"""
HF Chat Templates prompt strategy
"""
from typing import Any, Dict, Optional

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import Prompter
from axolotl.utils.chat_templates import chat_templates


class ChatTemplatePrompter(Prompter):
    """prompter for HF chat templates"""

    def __init__(self, tokenizer, chat_template=None, max_length=2048):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.max_length = max_length

    def build_prompt(self, conversation, add_generation_prompt=False):
        return self.tokenizer.apply_chat_template(
            conversation,
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=add_generation_prompt,
            chat_template=self.chat_template,
        )


class ChatTemplateStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for instruction-based prompts.
    """

    def tokenize_prompt(self, prompt):
        turns = self.get_conversation_thread(prompt)
        prompt_ids = self.prompter.build_prompt([turns[0]], add_generation_prompt=True)
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
        conversations = prompt["conversations"]
        # remap roles - allow for assistant turn
        role_map = {
            "human": "user",
            "user": "user",
            "assistant": "assistant",
            "gpt": "assistant",
        }
        turns = [
            {"role": role_map[t["from"]], "content": t["value"]} for t in conversations
        ]
        return turns


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    chat_template = (
        ds_cfg["chat_template"] if ds_cfg and "chat_template" in ds_cfg else "chatml"
    )
    strategy = ChatTemplateStrategy(
        ChatTemplatePrompter(tokenizer, chat_templates(chat_template)),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    return strategy
