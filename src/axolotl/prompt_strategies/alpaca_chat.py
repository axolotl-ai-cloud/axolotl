"""Module for Alpaca prompt strategy classes"""

from typing import Any, Dict, Optional, Tuple

from axolotl.prompt_tokenizers import (
    AlpacaPromptTokenizingStrategy,
    InstructionPromptTokenizingStrategy,
)
from axolotl.prompters import AlpacaPrompter, PromptStyle, UnpromptedPrompter


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    prompt_style = PromptStyle.CHAT.value
    if ds_cfg and "conversation" in ds_cfg:
        prompt_style = ds_cfg["conversation"]

    return AlpacaPromptTokenizingStrategy(
        AlpacaPrompter(prompt_style),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


class AlpacaConcisePrompter(AlpacaPrompter):
    """
    Alpaca Prompter extending the system prompt to ask for concise chat-instruct answers
    """

    system_prompt = "Below is an instruction from a USER that describes a task, paired with an input that provides further context. The ASSISTANT writes a response that concisely and appropriately completes the request.\n\n"
    system_no_input_prompt = "Below is an instruction from a USER that describes a task. The ASSISTANT writes a response that appropriately and concisely completes the request.\n\n"


class AlpacaChatPrompter(AlpacaPrompter):
    """
    Alpaca Chat Prompter extending the system prompt to for chat-instruct answers
    """

    system_prompt = "Below is an instruction from a USER that describes a task, paired with an input that provides further context. The ASSISTANT writes a response that concisely and appropriately completes the request.\n\n"
    system_no_input_prompt = "Below is an instruction from a USER that describes a task. The ASSISTANT writes a response that appropriately and concisely completes the request.\n\n"

    def __init__(self):  # pylint: disable=super-init-not-called
        self.prompt_style = PromptStyle.CHAT.value
        self.match_prompt_style()


class NoSystemPrompter(AlpacaPrompter):
    """
    Null Prompter with no system prompts
    """

    system_prompt = ""
    system_no_input_prompt = ""
    turn_format = "{instruction} {input} "
    turn_no_input_format = "{instruction} "

    def __init__(self):  # pylint: disable=super-init-not-called
        pass


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
            prompt["message_2"],
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
        AlpacaChatPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_camel_ai(tokenizer, cfg):
    return CamelAIPromptTokenizingStrategy(
        AlpacaChatPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_no_prompt(tokenizer, cfg):
    return AlpacaPromptTokenizingStrategy(
        UnpromptedPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
