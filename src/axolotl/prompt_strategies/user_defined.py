"""
User Defined prompts with configuration from the YML config
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

from axolotl.prompt_strategies.alpaca_w_system import (
    InstructionWSystemPromptTokenizingStrategy,
    SystemDataPrompter,
)


@dataclass
class UserDefinedDatasetConfig:
    """
    dataclass configuration representing a userdefined dataset type
    """

    system_prompt: str = ""
    field_system: str = "system"
    field_instruction: str = "instruction"
    field_input: str = "input"
    field_output: str = "output"
    format: str = "{instruction} {input} "
    no_input_format: str = "{instruction} "
    system_format: str = "{system}"

    def __getitem__(self, item):
        return getattr(self, item)


class UserDefinedPromptTokenizationStrategy(InstructionWSystemPromptTokenizingStrategy):
    """
    Prompt Tokenization Strategy for user defined prompts
    """


def load(tokenizer, cfg, ds_cfg: Optional[UserDefinedDatasetConfig] = None):
    if not ds_cfg:
        raise ValueError("Missing dataset prompt configuration")

    system_prompt = ""
    if ds_cfg.system_prompt:
        system_prompt = ds_cfg.system_prompt

    def parse_instruction_fields(
        field_instruction,
        field_input,
        field_output,
        field_system,
        system_prompt,
        prompt,
    ) -> Tuple[str, str, str, str]:
        return (
            prompt[field_instruction],
            prompt[field_input] if field_input in prompt else "",
            prompt[field_output] if field_output in prompt else "",
            prompt[field_system] if field_system in prompt else system_prompt,
        )

    turn_format = ds_cfg.format
    turn_no_input_format = ds_cfg.no_input_format
    system_format = ds_cfg.system_format

    class UserDefinedPrompter(SystemDataPrompter):
        """
        Prompter for user defined prompts
        """

        def match_prompt_style(self):
            self.turn_format = turn_format
            self.turn_no_input_format = turn_no_input_format
            self.system_format = system_format

    prompter = UserDefinedPrompter()

    strat = UserDefinedPromptTokenizationStrategy(
        prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )

    setattr(
        strat,
        "parse_instruction_fields",
        partial(
            parse_instruction_fields,
            ds_cfg.field_instruction,
            ds_cfg.field_input,
            ds_cfg.field_output,
            ds_cfg.field_system,
            system_prompt,
        ),
    )
    return strat
