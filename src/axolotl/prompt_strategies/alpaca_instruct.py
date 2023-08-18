"""Module loading the AlpacaInstructPromptTokenizingStrategy class"""
import logging

from axolotl.prompt_tokenizers import AlpacaPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, PromptStyle, UnpromptedPrompter

LOG = logging.getLogger("axolotl.prompt_strategies.alpaca_instruct")


class LatentSpaceAlpacaPromptTokenizingStrategy(AlpacaPromptTokenizingStrategy):
    """
    Overrides the tokenization to include additional padding tokens as
    latent space on the inputs
    """

    def _tokenize(self, prompt: str, add_eos_token=True, strip_bos_token=False):
        # pylint: disable=duplicate-code
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.sequence_len,
            padding=False,
            return_tensors=None,
        )
        if len(result["input_ids"]) == 0:
            LOG.warning("Tokenizer result is empty. You may want to audit your dataset")
        if (
            len(result["input_ids"]) > 0
            and result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.sequence_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if result["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        # latent space
        if add_eos_token and not strip_bos_token:
            result["input_ids"].extend([self.tokenizer.pad_token_id] * 100)

        result["labels"] = result["input_ids"].copy()
        return result


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


def load_latent_space(tokenizer, cfg):
    return LatentSpaceAlpacaPromptTokenizingStrategy(
        AlpacaPrompter(PromptStyle.INSTRUCT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
