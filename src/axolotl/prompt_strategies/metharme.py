"""Module containing the MetharmenPromptTokenizingStrategy and MetharmePrompter class"""

import logging
from typing import Tuple

from axolotl.prompt_tokenizers import InstructionPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter

LOG = logging.getLogger("axolotl")

IGNORE_TOKEN_ID = -100

# pylint: disable=duplicate-code


class MetharmePromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for the Metharme models
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (prompt["prompt"], "", prompt["generation"])

    def _tokenize(
        self,
        prompt: str,
        add_eos_token: bool = True,
        strip_bos_token: bool = False,
        num_eos_tokens: int = 3,
    ):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.sequence_len,
            padding=False,
            return_tensors=None,
        )
        if len(result["input_ids"]) == 0:
            LOG.warning("Tokenizer result is empty. You may want to audit your dataset")
        # If there's already an EOS token there, subtract from the number added
        if result["input_ids"][-1] == self.tokenizer.eos_token_id:
            num_eos_tokens -= 1

        if num_eos_tokens > 0 and add_eos_token and len(result["input_ids"]) > 0:
            for _ in range(num_eos_tokens):
                if len(result["input_ids"]) < self.sequence_len:
                    result["input_ids"].append(self.tokenizer.eos_token_id)
                    result["attention_mask"].append(1)

        if result["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()
        return result


class MetharmePrompter(AlpacaPrompter):
    """
    Prompter for the Metharme models.
    """

    system_prompt = ""
    system_no_input_prompt = ""
    system_format = ""
    turn_format = "{instruction}"
    turn_no_input_format = "{instruction}"

    def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
        pass


def load(tokenizer, cfg):
    return MetharmePromptTokenizingStrategy(
        MetharmePrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
