"""Module loading the AlpacaInstructPromptTokenizingStrategy class"""

from axolotl.prompt_tokenizers import AlpacaPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, PromptStyle, UnpromptedPrompter


def load(tokenizer, cfg):
    return AlpacaPromptTokenizingStrategy(
        AlpacaPrompter(PromptStyle.INSTRUCT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_no_prompt(tokenizer, cfg):
    return AlpacaPromptTokenizingStrategy(
        UnpromptedPrompter(PromptStyle.INSTRUCT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
