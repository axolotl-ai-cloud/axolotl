"""
Prompt strategies loader for alpaca instruction datasets with system prompts
"""
from typing import Generator, Tuple, Union

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, PromptStyle


class InstructionWSystemPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for instruction-based prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str, str]:
        return (
            prompt["instruction"],
            prompt["input"] if "input" in prompt else "",
            prompt["output"],
            prompt["system"],
        )

    def tokenize_prompt(self, prompt):
        # pylint: disable=duplicate-code
        (
            instruction,
            input,  # pylint: disable=redefined-builtin
            response,
            system,
        ) = self.parse_instruction_fields(prompt)
        user_prompt = next(
            iter(
                self.prompter.build_prompt_w_system(
                    system,
                    instruction,
                    input,
                )
            )
        )
        tokenized_prompt = self._tokenize(user_prompt, add_eos_token=False)
        if not self.train_on_inputs:
            user_prompt_len = len(tokenized_prompt["input_ids"])
            # TODO this could be sped up using numpy array slicing
            tokenized_prompt["labels"] = [-100] * user_prompt_len
        tokenized_res_prompt = self._tokenize(
            response, strip_bos_token=True, add_eos_token=True
        )
        tokenized_prompt["input_ids"] += tokenized_res_prompt["input_ids"]
        tokenized_prompt["attention_mask"] += tokenized_res_prompt["attention_mask"]
        tokenized_prompt["labels"] += tokenized_res_prompt["input_ids"]

        return tokenized_prompt


class SystemDataPrompter(AlpacaPrompter):
    """
    Alpaca Style Prompter that uses system prompts from the dataset
    """

    def build_prompt_w_system(
        self,
        system: str,
        instruction: str,
        input: Union[None, str] = None,  # pylint: disable=redefined-builtin
        output: Union[None, str] = None,
    ) -> Generator[str, None, None]:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        formatted_sys_prompt = (
            self.system_format.format(system=system)
            if system and self.system_format
            else ""
        )
        if input:
            res = formatted_sys_prompt + self.turn_format.format(
                instruction=instruction, input=input
            )
        else:
            res = formatted_sys_prompt + self.turn_no_input_format.format(
                instruction=instruction
            )
        if output:
            res = f"{res}{output}"
        yield res


class OpenOrcaSystemDataPrompter(SystemDataPrompter):
    """
    Alpaca Style Prompter that uses system prompts from the dataset, with OpenOrca prompts
    """

    def match_prompt_style(self):
        # pylint: disable=duplicate-code
        if self.prompt_style == PromptStyle.INSTRUCT.value:
            self.turn_format = "### Human:\n{instruction}\n### Additional Context:\n{input}\n### Assistant:\n"
            self.turn_no_input_format = "### Human:\n{instruction}\n### Assistant:\n"
            self.system_format = "### System:\n{system}\n"
        if self.prompt_style == PromptStyle.CHAT.value:
            self.turn_format = "USER: {instruction}\n{input}\nASSISTANT:"
            self.turn_no_input_format = "USER: {instruction}\nASSISTANT:"
            self.system_format = "SYSTEM: {system}\n"
        if self.prompt_style == PromptStyle.CHATML.value:
            self.turn_format = "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n"
            self.turn_no_input_format = (
                "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            )
            self.system_format = "<|im_start|>system\n{system}<|im_end|>\n"


class OpenOrcaPromptTokenizingStrategy(InstructionWSystemPromptTokenizingStrategy):
    """
    Tokenizing strategy for OpenOrca datasets
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str, str]:
        return (
            prompt["question"],
            "",
            prompt["response"],
            prompt["system_prompt"],
        )


def load(tokenizer, cfg):
    return load_chat(tokenizer, cfg)


def load_instruct(tokenizer, cfg):
    return InstructionWSystemPromptTokenizingStrategy(
        SystemDataPrompter(PromptStyle.INSTRUCT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_chat(tokenizer, cfg):
    return InstructionWSystemPromptTokenizingStrategy(
        SystemDataPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_open_orca(tokenizer, cfg):
    return OpenOrcaPromptTokenizingStrategy(
        OpenOrcaSystemDataPrompter(PromptStyle.INSTRUCT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


def load_open_orca_chatml(tokenizer, cfg):
    return OpenOrcaPromptTokenizingStrategy(
        OpenOrcaSystemDataPrompter(PromptStyle.CHATML.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
