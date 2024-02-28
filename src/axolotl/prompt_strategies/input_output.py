"""Module for plain input/output prompt pairs"""

from axolotl.prompt_tokenizers import PromptTokenizingStrategy


class InputOutputStrategy(PromptTokenizingStrategy):
    """Prompt Strategy class for input/output pairs"""

    def __init__(self, *args, eos_token=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eos_token = eos_token
        if not eos_token:
            self.eos_token = self.tokenizer.eos_token

    def tokenize_prompt(self, prompt):
        # pylint: disable=duplicate-code
        input_: str = prompt["input"]
        output: str = prompt["output"] + self.eos_token
        if not input_.endswith(" ") and not input_.endswith("\n"):
            input_ += " "
        input_ids_prompt = self.tokenizer(input_, return_tensors=None)["input_ids"]
        input_ids = self.tokenizer(input_ + output, return_tensors=None)["input_ids"]

        if not self.train_on_inputs:
            user_prompt_len = len(input_ids_prompt)
            labels = [-100] * user_prompt_len + input_ids[user_prompt_len:]
        else:
            labels = input_ids

        tokenized_prompt = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

        return tokenized_prompt


def load(tokenizer, cfg):
    return InputOutputStrategy(
        None,
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )
