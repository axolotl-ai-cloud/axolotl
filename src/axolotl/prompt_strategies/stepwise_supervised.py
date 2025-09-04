"""
Module for stepwise datasets, typically including a prompt and reasoning traces,
and (optionally) per-step, or per-prompt-trace labels for reward modelling.
"""

from itertools import chain
from typing import Dict, List, Optional, Union

from transformers import BatchEncoding, PreTrainedTokenizer

from axolotl.prompt_tokenizers import IGNORE_INDEX
from axolotl.utils.dict import DictDefault


class StepwiseSupervisedPromptTokenizingStrategy:
    """
    Tokenizing strategy for supervised stepwise datasets, typically used for COT-reasoning.
    These datasets should include the following columns:
    - prompt: the prompt text
    - completions: a list of `n` completion steps
    - labels: a list of `n` labels indicating the "correctness" of each step
    """

    def __init__(
        self,
        tokenizer,
        sequence_len: int = 2048,
        step_separator: str = "\n",
        max_completion_length: Optional[int] = None,
        train_on_last_step_only: bool = False,
    ):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.step_separator = step_separator
        self.max_completion_length = max_completion_length
        self.train_on_last_step_only = train_on_last_step_only

    def tokenize_prompt(
        self, prompt: Dict[str, Union[str, List[str]]]
    ) -> BatchEncoding:
        # Inspired by TRL's PRMTRainer
        # https://github.com/huggingface/trl/blob/ed7de87dc766478c024b68f12530d1b0e7c3ff23/trl/trainer/prm_trainer.py#L206
        prompt_ids = self.tokenizer(prompt["prompt"], add_special_tokens=False)[
            "input_ids"
        ]

        completions_ids = [
            self.tokenizer(completion, add_special_tokens=False)["input_ids"]
            for completion in prompt["completions"]
        ]

        # Handle labels
        if self.train_on_last_step_only:
            labels = [IGNORE_INDEX] * (len(prompt["labels"]) - 1) + [
                int(prompt["labels"][-1])
            ]
        else:
            labels = [int(label) for label in prompt["labels"]]

        # Add step separators
        separator_ids = self.tokenizer.encode(
            self.step_separator, add_special_tokens=False
        )
        completions_ids = [completion + separator_ids for completion in completions_ids]

        # Create step-wise labels
        labels = [
            [IGNORE_INDEX] * (len(completion) - 1) + [label]  # type: ignore
            for completion, label in zip(completions_ids, labels)
        ]

        # Join all steps
        completion_ids = list(chain(*completions_ids))
        labels = list(chain(*labels))  # type: ignore

        # Handle max lengths
        if self.max_completion_length:
            completion_ids = completion_ids[: self.max_completion_length]
            labels = labels[: self.max_completion_length]

        # Add BOS token if model has one
        if self.tokenizer.bos_token_id is not None:
            prompt_ids = [self.tokenizer.bos_token_id] + prompt_ids

        # Combine prompt and completion
        input_ids = prompt_ids + completion_ids

        full_labels = [IGNORE_INDEX] * len(prompt_ids) + labels
        # Apply max sequence length
        if self.sequence_len:
            input_ids = input_ids[: self.sequence_len]
            full_labels = full_labels[: self.sequence_len]

        return {
            "input_ids": input_ids,
            "labels": full_labels,
            "attention_mask": [1] * len(input_ids),
        }

    @property
    def supports_batched(self):
        return False


def load(
    tokenizer: PreTrainedTokenizer,
    cfg: DictDefault,
    ds_cfg: DictDefault,
) -> StepwiseSupervisedPromptTokenizingStrategy:
    return StepwiseSupervisedPromptTokenizingStrategy(
        tokenizer,
        cfg.sequence_len,
        step_separator=ds_cfg.get("step_separator", "\n"),
        max_completion_length=ds_cfg.max_completion_length,
        train_on_last_step_only=ds_cfg.get("train_on_last_step_only", False),
    )
