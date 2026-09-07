"""
Prompt Strategy for finetuning Orca Mini (v2) models
see also https://huggingface.co/psmathur/orca_mini_v2_7b for more information

Use dataset type: orcamini in conig.yml to use this prompt style.

Compared to the alpaca_w_system.open_orca dataset type,
this one specifies the system prompt with "### System:".

Not suited/tested for multiple-turn conversations without further adjustments.
"""

from typing import Generator, Union

from axolotl.prompt_strategies.alpaca_w_system import OpenOrcaPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter


class OrcaMiniPrompter(AlpacaPrompter):
    """Adjusted Prompter for Orca Mini (v2) datasets"""

    def match_prompt_style(self):
        self.turn_no_input_format = (
            "### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
        )

    def build_prompt_w_system(
        self,
        system: str,
        instruction: str,
        output: Union[None, str] = None,
    ) -> Generator[str, None, None]:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        res = self.turn_no_input_format.format(system=system, instruction=instruction)
        if output:
            res = f"{res}{output}"
        yield res


def load(tokenizer, cfg):
    return OpenOrcaPromptTokenizingStrategy(
        OrcaMiniPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
