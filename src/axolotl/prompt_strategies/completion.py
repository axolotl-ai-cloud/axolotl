"""
Basic completion text
"""

from collections import defaultdict
from typing import Any, Dict, Generator, Optional, Tuple

from axolotl.prompt_tokenizers import InstructionPromptTokenizingStrategy


class CompletionPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for Completion prompts.
    """

    _field: str = "text"

    def __init__(self, *args, max_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_length is not None:
            self.max_length = max_length

    @property
    def supports_batched(self):
        return True

    @property
    def field(self) -> str:
        return self._field

    @field.setter
    def field(self, new_field: str):
        self._field = new_field

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt[self.field],
            "",
            "",
        )

    def tokenize_prompt(self, prompt):
        res = defaultdict(lambda: [])
        feature_names = list(prompt.keys())
        for row in zip(*prompt.values()):
            prompt_row = dict(zip(feature_names, row))
            (
                instruction,
                _,
                _,
            ) = self.parse_instruction_fields(prompt_row)

            full_prompt = self._build_full_prompt(instruction, None, None)
            tokenized_full_prompt = self._tokenize(full_prompt)

            for key, val in tokenized_full_prompt.items():
                for i in range(0, len(val), self.sequence_len):
                    res[key].append(val[i : i + self.sequence_len])

        return dict(res)

    def _build_full_prompt(
        self, instruction, input, response
    ):  # pylint: disable=redefined-builtin
        return next(iter(self.prompter.build_prompt(instruction, input, response)))


class CompletionPrompter:
    """
    Prompter for completion
    """

    def build_prompt(
        self,
        instruction: str,
        input=None,  # pylint: disable=redefined-builtin, unused-argument
        output=None,  # pylint: disable=unused-argument
    ) -> Generator[str, None, None]:
        yield instruction


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    strat = CompletionPromptTokenizingStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        max_length=cfg.sequence_len * 64,
    )
    if ds_cfg and "field" in ds_cfg:
        strat.field = ds_cfg["field"]

    return strat
