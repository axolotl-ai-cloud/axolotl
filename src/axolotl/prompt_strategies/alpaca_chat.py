from axolotl.prompt_tokenizers import AlpacaPromptTokenizingStrategy, InstructionPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, PromptStyle


def load(tokenizer, cfg):
    return AlpacaPromptTokenizingStrategy(
        AlpacaPrompter(PromptStyle.chat), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )


class AlpacaQAPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> (str, str, str):
        return (
            prompt["question"],
            "",
            prompt["answer"],
        )


def load_qa(tokenizer, cfg):
    return AlpacaQAPromptTokenizingStrategy(
        AlpacaPrompter(PromptStyle.chat), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
