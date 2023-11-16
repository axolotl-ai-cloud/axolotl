"""Module containing the AptChatPromptTokenizingStrategy and AptChatPrompter class"""

import copy
import logging
from collections import defaultdict
from typing import Generator, List, Tuple

from axolotl.prompt_tokenizers import (
    PromptTokenizingStrategy,
    parse_tokenized_to_result,
    tokenize_prompt_default,
)

LOG = logging.getLogger("axolotl")

IGNORE_TOKEN_ID = -100


class AptChatPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for AptChat.
    """

    bot_prefix_token_ids: List[int] = []

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        super().__init__(prompter, tokenizer, *args, **kwargs)
        res = self._tokenize("<|assistant|>\n", add_eos_token=False, strip_bos_token=True)
        self.bot_prefix_token_ids = res["input_ids"]

    def tokenize_prompt(self, prompt):
        result, current_len = tokenize_prompt_default()
        for _, part in enumerate(self.prompter.build_prompt(prompt["conversations"])):
            role, message = part
            if role == "system":
                prefix = "<|system|>\n"
                res = self._tokenize(
                    prefix + message.strip(),
                    add_eos_token=True,
                    strip_bos_token=False,
                )
                # everything from this is masked out from the labels
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            elif role == "user" or role == "human":
                prefix = "<|user|>\n"
                res = self._tokenize(
                    prefix + message.strip(),
                    add_eos_token=True,
                    strip_bos_token=True,
                )
                # everything from this is masked out from the labels
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            elif role == "assistant" or role == "bot" or role == "gpt":
                prefix = "<|assistant|>\n"
                res = self._tokenize(
                    prefix + message.strip(),
                    add_eos_token=True,
                    strip_bos_token=True,
                )
                # mask out the prefix token, rest is not masked out from labels
                # make sure we create the labels first, otherwise we get incorrect lengths
                labels = [IGNORE_TOKEN_ID] * len(self.bot_prefix_token_ids) + [
                    *copy.deepcopy(res["input_ids"])
                ][len(self.bot_prefix_token_ids) :]
            else:
                LOG.warning(f"unknown role in conversation: {role}")
                res = defaultdict(lambda: [])

            # pylint: disable=duplicate-code
            result, current_len = parse_tokenized_to_result(
                result,
                current_len,
                res,
                labels,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return result


class AptChatPrompter:
    """
    Prompter for AptChat.
    """

    def __init__(self, *args, **kwargs):
        pass

    def build_prompt(
        self, source, *args, **kwargs  # pylint: disable=unused-argument
    ) -> Generator[Tuple[str, str], None, None]:
        for msg in source:
            yield msg["role"], msg["content"]


def load(tokenizer, cfg):
    return AptChatPromptTokenizingStrategy(
        AptChatPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
