import abc
import copy

from transformers import PreTrainedTokenizer

from axolotl.prompters import IGNORE_TOKEN_ID

IGNORE_INDEX = -100
LLAMA_DEFAULT_PAD_TOKEN = "[PAD]"
LLAMA_DEFAULT_EOS_TOKEN = "</s>"
LLAMA_DEFAULT_BOS_TOKEN = "<s>"
LLAMA_DEFAULT_UNK_TOKEN = "<unk>"


class InvalidDataException(Exception):
    pass


class PromptTokenizingStrategy(abc.ABC):
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


class InstructionPromptTokenizingStrategy(PromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> (str, str, str):
        raise NotImplementedError

    def tokenize_prompt(self, prompt):
        instruction, input, response = self.parse_instruction_fields(prompt)
        full_prompt = self._build_full_prompt(instruction, input, response)
        tokenized_full_prompt = self._tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = next(iter(self.prompter.build_prompt(
                instruction,
                input,
            )))
            tokenized_user_prompt = self._tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # TODO this could be sped up using numpy array slicing
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    def _build_full_prompt(self, instruction, input, response):
        return next(iter(self.prompter.build_prompt(
            instruction,
            input,
            response,
        )))

    def _tokenize(self, prompt, add_eos_token=True):
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


class AlpacaPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> (str, str, str):
        return (
            prompt["instruction"],
            prompt["input"] if "input" in prompt else "",
            prompt["output"],
        )


class JeopardyPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> (str, str, str):
        return (
            prompt["question"],
            prompt["category"],
            "what is " + prompt["answer"],
        )


class OpenAssistantPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> (str, str, str):
        return (
            prompt["INSTRUCTION"],
            "",
            prompt["RESPONSE"],
        )


class GPTeacherPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> (str, str, str):
        return (
            prompt["instruction"],
            prompt["input"] if "input" in prompt else "",
            prompt["response"],
        )


class NomicGPT4AllPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> (str, str, str):
        return (
            prompt["prompt"],
            "",
            prompt["response"],
        )


class CompletionPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> str:
        return prompt["text"]

    def tokenize_prompt(self, prompt):
        instruction = self.parse_instruction_fields(prompt)
        full_prompt = self._build_full_prompt(instruction, None, None)
        tokenized_full_prompt = self._tokenize(full_prompt)

        return tokenized_full_prompt

    def _build_full_prompt(self, instruction, input, response):
        return next(iter(self.prompter.build_prompt(instruction)))


class ReflectionPromptTokenizingStrategy(PromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> (str, str, str, str, str):
        raise NotImplementedError

    def tokenize_prompt(self, prompt):
        (
            instruction,
            input,
            output,
            reflection,
            corrected,
        ) = self.parse_instruction_fields(prompt)
        full_prompt = self._build_full_prompt(
            instruction, input, output, reflection, corrected
        )
        tokenized_full_prompt = self._tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = next(iter(self.prompter.build_prompt(
                instruction,
                input,
            )))
            tokenized_user_prompt = self._tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # TODO this could be sped up using numpy array slicing
            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    def _build_full_prompt(self, instruction, input, output, reflection, corrected):
        return next(iter(self.prompter.build_prompt(
            instruction,
            input,
            output,
            reflection,
            corrected,
        )))

    def _tokenize(self, prompt, add_eos_token=True):
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
    def parse_instruction_fields(self, prompt) -> (str, str, str, str, str):
        return (
            prompt["instruction"],
            prompt["input"] if "input" in prompt else "",
            prompt["output"],
            prompt["reflection"],
            prompt["corrected"],
        )


class ShareGPTPromptTokenizingStrategy(PromptTokenizingStrategy):
    def tokenize_prompt(self, prompt):
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        current_len = 0
        try:
            for i, part in enumerate(self.prompter.build_prompt(prompt["conversations"], self.tokenizer)):
                if i == 0:
                    # this is only ever the first part, should include the bos token and the user query
                    res = self._tokenize(part.strip(), add_eos_token=False, strip_bos_token=False)
                    # everything from this is masked out from the labels
                    labels = [ IGNORE_TOKEN_ID ] * len(res["input_ids"])
                elif i % 2 == 0:
                    # this is still the user query, we should
                    res = self._tokenize(part.strip(), add_eos_token=False, strip_bos_token=True)
                    # everything from this is masked out from the labels
                    labels = [ IGNORE_TOKEN_ID ] * len(res["input_ids"])
                else:
                    # this should be the assistent response, should end with an eos token
                    res = self._tokenize(part.strip(), add_eos_token=True, strip_bos_token=True)
                    # not masked out from labels
                    labels = copy.deepcopy(res["input_ids"])
                input_ids = res["input_ids"]
                input_len = len(input_ids)
                result["input_ids"][current_len : current_len + input_len] = input_ids
                result["attention_mask"][current_len : current_len + input_len] = [
                    1 if x != self.tokenizer.pad_token_id else 0
                    for x in input_ids
                ]
                result["labels"][current_len : current_len + input_len] = labels
                current_len += input_len
            return result
        except (KeyError, AssertionError, IndexError) as e:
            raise InvalidDataException(str(e))

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

        if (
            result["input_ids"][0] == self.tokenizer.bos_token_id
            and strip_bos_token
        ):
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()
        return result
