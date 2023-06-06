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


class AlpacaConcisePrompter(AlpacaPrompter):
    """
    Alpaca Prompter extending the system prompt to ask for concise answers
    """

    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that concisely and appropriately completes the request.\n\n"
    system_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately and concisely completes the request.\n\n"


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


class CamelAIPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for CamelAI datasets
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["message_1"],
            "",
            prompt["message_1"],
        )


def load_concise(tokenizer, cfg):
    return AlpacaPromptTokenizingStrategy(
        AlpacaConcisePrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_qa(tokenizer, cfg):
    return AlpacaQAPromptTokenizingStrategy(
        AlpacaPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_camel_ai(tokenizer, cfg):
    return CamelAIPromptTokenizingStrategy(
        AlpacaPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
