"""
Prompt Strategy for finetuning Llama2 chat models
see also https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L213 for ma reference implementation.

This implementation is based on the Vicuna PR and the fastchat repo, see also:
https://github.com/lm-sys/FastChat/blob/cdd7730686cb1bf9ae2b768ee171bdf7d1ff04f3/fastchat/conversation.py#L847

Use dataset type: "llama2_chat" in conig.yml to use this prompt style.

E.g. in the config.yml:
```
datasets:
  - path: llama_finetune_train.jsonl
    type: llama2_chat
```

The dataset itself should look like this:
```
{'conversations':[{"from": "human", "value": "Who are you?"}, {"from": "gpt", "value": "I am Vicuna"},...]}
```
in a jsonl file. The first message should be from the human, the second from gpt.
For a custom system message, the first "from" can be "system" (followed by alternating "human" and "gpt" turns).

Important: Don't use "special_tokens:" in your config.yml if you are not sure what you are doing!
"""

import logging
from dataclasses import dataclass, field
from typing import Generator, List, Sequence

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import IGNORE_TOKEN_ID, SHAREGPT_ASSERTION_FAILED_ROLE


@dataclass
class Llama2ChatConversation:
    """A class that manages prompt templates and keeps all conversation history.
    copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py"""

    name: str = "llama2"
    # The system prompt
    system: str = (
        "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
    )
    roles: Sequence[str] = ("[INST]", "[/INST]")
    messages: List[List[str]] = field(default_factory=list)
    offset: int = 0
    sep = " "
    sep2 = " </s><s>"
    stop_token_ids = [2]

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        seps = [self.sep, self.sep2]
        ret = ""
        for i, (role, message) in enumerate(self.messages):
            if (i == len(self.messages) - 1) and (role == self.roles[0]):
                # last message is from user (due to length),
                #  return prompt without it for training
                return ret
            if i == 0:
                ret += self.system + message.strip()
            else:
                ret += role + " " + message.strip() + seps[i % 2]
        return ret

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])


class LLama2ChatTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for ShareGPT prompts.
    adapted from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_len = 4096
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/added_tokens.json

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
        sep = conv.roles[1]

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
            # "-1" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 1

            # Ignore the user instructions
            target[cur_len - 1 : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len + 2  # due to length of role token

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < self.sequence_len:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                logging.warning(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).tolist()
        input_ids = input_ids.tolist()
        target = target.tolist()
        # this is a fix for the tokenizer which tokenizes [ differently with eos tokens and
        # follows the original llama implementation
        for i in range(2, total_len - 2):
            if input_ids[i] == 29961:
                input_ids[i] = 518
            if target[i] == 29961:
                target[i] = 518
        return {
            "input_ids": input_ids,
            "labels": target,
            "attention_mask": attention_mask,
        }


class Llama2ChatPrompter:  # pylint: disable=too-few-public-methods
    """
    A prompter that generates prompts for Llama2 models.
    """

    system_prompt = (
        "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
    )

    def build_prompt(self, source) -> Generator[Llama2ChatConversation, None, None]:
        # see https://github.com/lm-sys/FastChat/blob/da0641e567cf93756b0978ab5a6b092e96f06240/fastchat/train/train.py#L78
        source = source["conversations"]  # fix data structure for datasets

        # if system prompt provided, use it
        if source[0]["from"] == "system":
            system = f"[INST] <<SYS>>\n{source[0]['value']}\n<</SYS>>\n\n"
            source = source[1:]
        else:
            system = self.system_prompt

        conv = Llama2ChatConversation(system=system)

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
            assert role == conv.roles[j % 2], SHAREGPT_ASSERTION_FAILED_ROLE
            if sentence["value"]:
                conv.append_message(role, sentence["value"])
        yield conv


def load(tokenizer, cfg) -> LLama2ChatTokenizingStrategy:
    return LLama2ChatTokenizingStrategy(
        Llama2ChatPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
