"""Module containing PromptTokenizingStrategy and Prompter classes"""

import abc
import copy
import functools
import logging
from typing import Dict, List, Tuple, Union

from transformers import PreTrainedTokenizer

from axolotl.prompters import IGNORE_TOKEN_ID

LOG = logging.getLogger("axolotl")

IGNORE_INDEX = -100
LLAMA_DEFAULT_PAD_TOKEN = "[PAD]"  # nosec
LLAMA_DEFAULT_EOS_TOKEN = "</s>"  # nosec
LLAMA_DEFAULT_BOS_TOKEN = "<s>"  # nosec
LLAMA_DEFAULT_UNK_TOKEN = "<unk>"  # nosec


class InvalidDataException(Exception):
    """
    Exception raised when the data is invalid
    """


class PromptTokenizingStrategy(abc.ABC):
    """
    Abstract class for tokenizing strategies
    """

    def __init__(
        self,
        prompter,
        tokenizer,
        train_on_inputs: bool = False,
        sequence_len: int = 2048,
    ):
        self.prompter = prompter
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.sequence_len = sequence_len

    @abc.abstractmethod
    def tokenize_prompt(self, prompt):
        pass

    @functools.lru_cache(maxsize=128)
    def _get_user_token(self):
        try:
            id_or_ids = self.tokenizer.convert_tokens_to_ids("<|USER|>")
            if isinstance(id_or_ids, (int,)):
                return id_or_ids
        except KeyError:
            pass
        return False

    @functools.lru_cache(maxsize=128)
    def _get_assistant_token(self):
        try:
            id_or_ids = self.tokenizer.convert_tokens_to_ids("<|ASSISTANT|>")
            if isinstance(id_or_ids, (int,)):
                return id_or_ids
        except KeyError:
            pass
        return False

    def _tokenize(self, prompt: str, add_eos_token=True, strip_bos_token=False):
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

        if (
            len(result["input_ids"]) > 0
            and result["input_ids"][0] == self.tokenizer.bos_token_id
            and strip_bos_token
        ):
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()
        return result


class InstructionPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for instruction-based prompts.
    """

    def parse_instruction_fields(
        self, prompt
    ) -> Union[Tuple[str, str, str], Tuple[str, str, str, str]]:
        raise NotImplementedError

    def tokenize_prompt(self, prompt):
        (
            instruction,
            input,  # pylint: disable=redefined-builtin
            response,
        ) = self.parse_instruction_fields(prompt)
        user_prompt = next(
            iter(
                self.prompter.build_prompt(
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

    def _build_full_prompt(
        self, instruction, input, response  # pylint: disable=redefined-builtin
    ):
        return next(
            iter(
                self.prompter.build_prompt(
                    instruction,
                    input,
                    response,
                )
            )
        )


class AlpacaPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for Alpaca prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["instruction"],
            prompt["input"] if "input" in prompt else "",
            prompt["output"],
        )


class AlpacaMultipleChoicePromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for Alpaca Multiple Choice prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["question"],
            "\n".join(f'- "{choice}"' for choice in prompt["choices"]),
            prompt["solution"] if "solution" in prompt else prompt["explanation"],
        )


class JeopardyPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for Jeopardy prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["question"],
            prompt["category"],
            "what is " + prompt["answer"],
        )


class OpenAssistantPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for OpenAssistant prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["INSTRUCTION"],
            "",
            prompt["RESPONSE"],
        )


class SummarizeTLDRPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for SummarizeTLDR prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["article"],
            "",
            prompt["summary"],
        )


class GPTeacherPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for GPTeacher prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["instruction"],
            prompt["input"] if "input" in prompt else "",
            prompt["response"],
        )


class NomicGPT4AllPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for NomicGPT4All prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt["prompt"],
            "",
            prompt["response"],
        )


class CompletionPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for Completion prompts.
    """

    def tokenize_prompt(self, prompt):
        full_prompt = self._build_full_prompt(prompt["text"], None, None)
        tokenized_full_prompt = self._tokenize(full_prompt)

        return tokenized_full_prompt

    def _build_full_prompt(
        self, instruction, input, response
    ):  # pylint: disable=redefined-builtin
        return next(iter(self.prompter.build_prompt(instruction, input, response)))


class ReflectionPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for Reflection prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str, str, str]:
        raise NotImplementedError

    def tokenize_prompt(self, prompt):
        (
            instruction,
            input,  # pylint: disable=redefined-builtin
            output,
            reflection,
            corrected,
        ) = self.parse_instruction_fields(prompt)
        full_prompt = self._build_full_prompt(
            instruction, input, output, reflection, corrected
        )
        tokenized_full_prompt = self._tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = next(
                iter(
                    self.prompter.build_prompt(
                        instruction,
                        input,
                    )
                )
            )
            tokenized_user_prompt = self._tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # TODO this could be sped up using numpy array slicing
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    def _build_full_prompt(
        self, instruction, input, output, reflection, corrected
    ):  # pylint: disable=redefined-builtin
        return next(
            iter(
                self.prompter.build_prompt(
                    instruction,
                    input,
                    output,
                    reflection,
                    corrected,
                )
            )
        )

    def _tokenize(self, prompt, add_eos_token=True, strip_bos_token=False):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.sequence_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.sequence_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result


class AlpacaReflectionPTStrategy(ReflectionPromptTokenizingStrategy):
    """
    Tokenizing strategy for Alpaca Reflection prompts.
    """

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str, str, str]:
        return (
            prompt["instruction"],
            prompt["input"] if "input" in prompt else "",
            prompt["output"],
            prompt["reflection"],
            prompt["corrected"],
        )


class ShareGPTPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for ShareGPT prompts.
    """

    def get_conversation_thread(self, prompt):
        return prompt["conversations"]

    def tokenize_prompt(self, prompt):
        result, current_len = tokenize_prompt_default()
        user_token = self._get_user_token()
        assistant_token = self._get_assistant_token()
        try:
            for _, part in enumerate(
                self.prompter.build_prompt(self.get_conversation_thread(prompt))
            ):
                if isinstance(part, tuple):
                    if part[0] == "USER:":
                        part = part[0] + part[1] if not user_token else part[1]
                        # this is still the user query, we should
                        res = self._tokenize(
                            part.strip(),
                            add_eos_token=False,
                            strip_bos_token=True,
                        )
                        if user_token:
                            res["input_ids"] = [user_token, *res["input_ids"]]
                        # everything from this is masked out from the labels
                        labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
                    elif part[0] == "ASSISTANT:":
                        # TODO label assistant token/tokens w/ IGNORE_TOKEN_ID
                        part = part[0] + part[1] if not assistant_token else part[1]
                        # this should be the assistent response, should end with an eos token
                        res = self._tokenize(
                            part.strip(),
                            add_eos_token=True,
                            strip_bos_token=True,
                        )
                        if assistant_token:
                            res["input_ids"] = [
                                assistant_token,
                                *res["input_ids"],
                            ]
                        # not masked out from labels
                        labels = copy.deepcopy(res["input_ids"])
                    elif part[0] == "SYSTEM:":
                        part = part[1]  # Ignore the system role from preamble
                        # this is only ever the first part, should include the bos token and the user query
                        res = self._tokenize(
                            part.strip(), add_eos_token=False, strip_bos_token=False
                        )
                        # everything from this is masked out from the labels
                        labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
                    else:
                        LOG.warning(f"unhandled role: {part[0]}")

                # pylint: disable=duplicate-code
                result, current_len = parse_tokenized_to_result(
                    result,
                    current_len,
                    res,
                    labels,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            return result
        except (KeyError, AssertionError, IndexError) as err:
            raise InvalidDataException(str(err)) from err

    def _tokenize(self, prompt, add_eos_token=True, strip_bos_token=False):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.sequence_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.sequence_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if result["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()
        return result


def tokenize_prompt_default() -> Tuple[Dict[str, List[int]], int]:
    """
    Returns the default values for the tokenize prompt function
    """

    result: Dict[str, List[int]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    current_len = 0
    return result, current_len


def parse_tokenized_to_result(
    result: Dict[str, List[int]],
    current_len: int,
    res: Dict[str, List[int]],
    labels: List[int],
    pad_token_id: Union[int, None] = None,
) -> Tuple[Dict[str, List[int]], int]:
    """
    Parses the tokenized prompt and append the tokenized input_ids, attention_mask and labels to the result
    """

    input_ids = res["input_ids"]
    input_len = len(input_ids)
    result["input_ids"][current_len : current_len + input_len] = input_ids
    result["attention_mask"][current_len : current_len + input_len] = [
        1 if x != pad_token_id else 0 for x in input_ids
    ]
    result["labels"][current_len : current_len + input_len] = labels
    current_len += input_len

    return result, current_len
