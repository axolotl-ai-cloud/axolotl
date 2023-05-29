"""Module containing the AlpacaQAPromptTokenizingStrategy class"""

from typing import Tuple

from axolotl.prompt_tokenizers import (
    AlpacaPromptTokenizingStrategy,
    InstructionPromptTokenizingStrategy,
)
from axolotl.prompters import AlpacaPrompter, PromptStyle


def load(tokenizer, cfg):
    return AlpacaPromptTokenizingStrategy(
        AlpacaPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


class AlpacaQAPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for AlpacaQA
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["question"],
            "",
            prompt["answer"],
        )


def load_qa(tokenizer, cfg):
    return AlpacaQAPromptTokenizingStrategy(
        AlpacaPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
