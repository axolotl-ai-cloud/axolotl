import copy
import logging
from collections import defaultdict
from typing import Generator

from axolotl.prompt_tokenizers import PromptTokenizingStrategy

IGNORE_TOKEN_ID = -100


class PygmalionPromptTokenizingStrategy(PromptTokenizingStrategy):
    bot_prefix_token_ids = []

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        super().__init__(prompter, tokenizer)
        res = self._tokenize("<|model|>", add_eos_token=False, strip_bos_token=True)
        self.bot_prefix_token_ids = res["input_ids"]

    def tokenize_prompt(self, prompt):
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        current_len = 0
        for i, part in enumerate(self.prompter.build_prompt(prompt["conversations"])):
            role, message = part
            if role == "system":
                prefix = "<|system|>"
                # this should include a bos token, no eos token, strip trailing "\n<START>"
                if message.endswith("\n<START>"):
                    message = message[:-8]
                res = self._tokenize(
                    prefix + "Persona: " + message.strip(),
                    add_eos_token=False,
                    strip_bos_token=False,
                )
                # everything from this is masked out from the labels
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            elif role == "human":
                prefix = "<|user|>"
                res = self._tokenize(
                    prefix + " " + message.strip(),
                    add_eos_token=False,
                    strip_bos_token=True,
                )
                # everything from this is masked out from the labels
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
            elif role == "bot":
                prefix = "<|model|>"
                res = self._tokenize(
                    prefix + " " + message.strip(),
                    add_eos_token=True,
                    strip_bos_token=True,
                )
                # mask out the prefix token, rest is not masked out from labels
                # make sure we create the labels first, otherwise we get incorrect lengths
                labels = [IGNORE_TOKEN_ID] * len(self.bot_prefix_token_ids) + [
                    *copy.deepcopy(res["input_ids"])
                ][len(self.bot_prefix_token_ids) :]
            else:
                logging.warning(f"unknown role in conversation: {role}")
                res = defaultdict(lambda: [])
            input_ids = res["input_ids"]
            input_len = len(input_ids)
            result["input_ids"][current_len : current_len + input_len] = input_ids
            result["attention_mask"][current_len : current_len + input_len] = [
                1 if x != self.tokenizer.pad_token_id else 0 for x in input_ids
            ]
            result["labels"][current_len : current_len + input_len] = labels
            current_len += input_len
        return result

    def _tokenize(self, prompt, add_eos_token=True, strip_bos_token=False):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.sequence_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.sequence_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if result["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()
        return result


class PygmalionPrompter:
    def __init__(self, *args, **kwargs):
        pass

    def build_prompt(self, source, *args, **kwargs) -> Generator[str, None, None]:
        for msg in source:
            yield msg["role"], msg["value"]


def load(tokenizer, cfg):
    return PygmalionPromptTokenizingStrategy(
        PygmalionPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
