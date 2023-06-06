"""Module containing the classes for Context QA Prompt Tokenization Strategies"""
from typing import Tuple

from axolotl.prompt_tokenizers import InstructionPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, PromptStyle


# article, unanswerable_question, question, answer
def load_404(tokenizer, cfg):
    return AlpacaMissingInfoContextPromptTokenizingStrategy(
        AlpacaContextPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load(tokenizer, cfg):
    return AlpacaContextPromptTokenizingStrategy(
        AlpacaContextPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


class AlpacaContextPrompter(AlpacaPrompter):
    """
    Customized system prompted for concise QA
    """

    system_prompt = (
        "Use the following contextual information to concisely answer the question.\n"
    )
    system_no_input_prompt = (
        "Use the following contextual information to concisely answer the question.\n"
    )


class AlpacaContextPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenization Strategy to combine in-context article with a question and answer
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["article"] + "\n===\n" + prompt["question"],
            "",
            prompt["answer"],
        )


class AlpacaMissingInfoContextPromptTokenizingStrategy(
    InstructionPromptTokenizingStrategy
):
    """
    Tokenization Strategy to combine in-context article with a question that can't be answered
    from the context and a default response to that effect
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["article"] + "\n===\n" + prompt["unanswerable_question"],
            "",
            "The context provided does not contain any information about your inquiry. "
            "Therefore, I'm unable to answer your question based on the given context.",
        )
