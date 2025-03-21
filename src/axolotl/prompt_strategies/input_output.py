"""Module for plain input/output prompt pairs"""

from typing import Generator, Tuple

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import IGNORE_TOKEN_ID, Prompter


class RawInputOutputStrategy(PromptTokenizingStrategy):
    """Prompt Strategy class for input/output pairs"""

    def __init__(self, *args, eos_token=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eos_token = eos_token
        if not eos_token:
            self.eos_token = self.tokenizer.eos_token

    def tokenize_prompt(self, prompt):
        # pylint: disable=duplicate-code
        input_ids = []
        labels = []
        for label, text in self.prompter.build_prompt(prompt["segments"]):
            tokenized_output = self.tokenizer(
                text, add_special_tokens=False, return_tensors=None
            )["input_ids"]
            input_ids += tokenized_output
            if label or self.train_on_inputs:
                labels += tokenized_output
            else:
                labels += [IGNORE_TOKEN_ID] * len(tokenized_output)

        tokenized_prompt = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

        return tokenized_prompt


class RawInputOutputPrompter(Prompter):
    """prompter for raw i/o data"""

    def build_prompt(self, source) -> Generator[Tuple[bool, str], None, None]:
        for segment in source:
            yield segment["label"], segment["text"]


def load(tokenizer, cfg):
    return RawInputOutputStrategy(
        RawInputOutputPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
