"""
Prompt Strategy for finetuning Vicuna v1.1+ models
see also https://huggingface.co/psmathur/orca_mini_v2_7b for more information

Use dataset type: "vicuna" in conig.yml to use this prompt style.

E.g. in the config.yml:
```
datasets:
  - path: vicuna_finetune_train.jsonl
    type: vicuna
```

The dataset itself should look like this:
```
{'conversation':[{"from": "human", "value": "Who are you?"}, {"from": "gpt", "value": "I am Vicuna"},...]}
```
in a jsonl file. The first message should be from the human, the second from gpt.
For a custom system message, the first "from" can be "system" (followed by "human" and "gpt" turns).
"""

import logging
from dataclasses import dataclass, field
from typing import Generator, List, Sequence

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import IGNORE_TOKEN_ID


@dataclass
class VicunaConversation:
    """A class that manages prompt templates and keeps all conversation history.
    copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py"""

    name: str = "vicuna_v1.1"
    # The system prompt
    system: str = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    roles: Sequence[str] = ("USER", "ASSISTANT")
    messages: List[List[str]] = field(default_factory=list)
    offset: int = 0
    sep: str = " "
    sep2: str = "</s>"

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if (i == len(self.messages) - 1) and (role == self.roles[0]):
                # last message is from user (due to length),
                #  return prompt without it
                return ret
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])


class VicunaTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for ShareGPT prompts.
    adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py
    """

    def tokenize_prompt(self, prompt):
        conv = next(self.prompter.build_prompt(prompt))
        conversation_str = conv.get_prompt()

        # Tokenize conversations
        input_ids = self.tokenizer(
            conversation_str,
            return_tensors="pt",
            padding="max_length",
            max_length=self.sequence_len,
            truncation=True,
        ).input_ids[0]
        target = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "

        total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

        turns = conversation_str.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for turn in turns:
            if turn == "":
                break
            turn_len = len(self.tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < self.sequence_len:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                logging.warning(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

        return {
            "input_ids": input_ids.tolist(),
            "labels": target.tolist(),
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id).tolist(),
        }


class VicunaPrompter:  # pylint: disable=too-few-public-methods
    """
    A prompter that generates prompts for the ShareGPT
    """

    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def build_prompt(self, source) -> Generator[VicunaConversation, None, None]:
        # see https://github.com/lm-sys/FastChat/blob/da0641e567cf93756b0978ab5a6b092e96f06240/fastchat/train/train.py#L78
        source = source["conversation"]  # fix data structure for datasets

        # if system prompt provided, use it
        if source[0]["from"] == "system":
            system = source[0]["value"]
            source = source[1:]
        else:
            system = self.system_prompt

        conv = VicunaConversation(system=system)

        if len(source) < 2:
            # If there isn't a back and forth conversation, ignore it
            # also happens on the data splitting leaving empty conversations
            raise IndexError

        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []  # pylint: disable=R0801
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        yield conv


def load(tokenizer, cfg) -> VicunaTokenizingStrategy:
    return VicunaTokenizingStrategy(
        VicunaPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
