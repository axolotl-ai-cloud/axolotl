"""Module containing the InstructShareGPTPromptTokenizingStrategy class"""
from typing import Any, Dict, Optional

from axolotl.prompt_tokenizers import ShareGPTPromptTokenizingStrategy
from axolotl.prompters import ShareGPTPrompterV2


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    conversation = (
        ds_cfg["conversation"] if ds_cfg and "conversation" in ds_cfg else None
    )
    strategy = InstructShareGPTPromptTokenizingStrategy(
        # pylint: disable=duplicate-code
        ShareGPTPrompterV2(
            conversation=conversation,
        ),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
    return strategy


class InstructShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
    """
    basic sharegpt strategy to grab conversations from the sample row
    """

    def get_conversation_thread(self, prompt):
        return [
            {"from": "human", "value": prompt["instruction"]},
            {"from": "gpt", "value": prompt["output"]},
        ]
