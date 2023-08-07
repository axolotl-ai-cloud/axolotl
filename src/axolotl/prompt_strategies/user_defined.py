"""
User Defined prompts with configuration from the YML config
"""

from typing import Tuple

from axolotl.prompt_strategies.alpaca_w_system import (
    InstructionWSystemPromptTokenizingStrategy,
    SystemDataPrompter,
)


class UserDefinedPromptTokenizationStrategy(InstructionWSystemPromptTokenizingStrategy):
    """
    Prompt Tokenization Strategy for user defined prompts
    """


class UserDefinedPrompter(SystemDataPrompter):
    """
    Prompter for user defined prompts
    """


def load(tokenizer, cfg, ds_cfg=None):
    if not ds_cfg:
        raise ValueError("Missing dataset prompt configuration")

    system_prompt = ""
    if ds_cfg["system_prompt"] and not ds_cfg["field_system"]:
        system_prompt = ds_cfg["system_prompt"]

    def parse_instruction_fields(
        self, prompt  # pylint: disable=unused-argument
    ) -> Tuple[str, str, str, str]:
        return (
            prompt[ds_cfg["field_instruction"]],
            prompt[ds_cfg["field_input"]]
            if ds_cfg["field_input"] and ds_cfg["field_input"] in prompt
            else "",
            prompt[ds_cfg["field_output"]],
            prompt[ds_cfg["field_system"]]
            if ds_cfg["field_system"] and ds_cfg["field_system"] in prompt
            else system_prompt,
        )

    def match_prompt_style(self):
        self.turn_format = ds_cfg["format"]
        self.turn_no_input_format = (
            ds_cfg["no_input_format"]
            if "no_input_format" in ds_cfg
            else ds_cfg["format"]
        )
        self.system_format = ds_cfg["system_format"]

    prompter = UserDefinedPrompter()
    prompter.match_prompt_style = match_prompt_style

    strat = UserDefinedPromptTokenizationStrategy(
        prompter,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )

    strat.parse_instruction_fields = parse_instruction_fields
    return strat
