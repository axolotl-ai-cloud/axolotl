"""Module containing prompters"""

import dataclasses
import logging
from enum import Enum, auto
from typing import Generator, List, Optional, Tuple, Union

LOG = logging.getLogger("axolotl")
IGNORE_TOKEN_ID = -100


class PromptStyle(Enum):
    """
    Enum for prompt styles
    """

    INSTRUCT = "instruct"
    CHAT = "chat"


class AlpacaPrompter:
    """
    Base class for alpaca prompters
    """

    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    system_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    turn_format: str
    turn_no_input_format: str
    prompt_style: Optional[PromptStyle] = None

    def __init__(self, prompt_style=PromptStyle.INSTRUCT.value):
        self.prompt_style = prompt_style if prompt_style else PromptStyle.INSTRUCT.value
        self.match_prompt_style()

    def match_prompt_style(self):
        if self.prompt_style == PromptStyle.INSTRUCT.value:
            self.turn_format = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            self.turn_no_input_format = (
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )
        if self.prompt_style == PromptStyle.CHAT.value:
            self.turn_format = "USER: {instruction}\n{input}\nASSISTANT:"
            self.turn_no_input_format = "USER: {instruction}\nASSISTANT:"

    def build_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,  # pylint: disable=redefined-builtin
        output: Union[None, str] = None,
    ) -> Generator[str, None, None]:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.system_prompt + self.turn_format.format(
                instruction=instruction, input=input
            )
        else:
            res = self.system_no_input_prompt + self.turn_no_input_format.format(
                instruction=instruction
            )
        if output:
            res = f"{res}{output}"
        yield res


class UnpromptedPrompter(AlpacaPrompter):
    """
    Prompter for alpaca no system prompt
    """

    system_prompt = ""
    system_no_input_prompt = ""


class JeopardyPrompter(AlpacaPrompter):
    """
    Prompter for Jeopardy
    """

    prompt_input = "Below is a Jeopardy clue paired with input providing the category of the clue. Write a concise response that best answers tbe clue given the category.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


class MultipleChoiceExplainPrompter(AlpacaPrompter):
    """
    Prompter for multiple choice explain
    """

    system_prompt = (
        "Choose the answer that best answers the question. Explain your reasoning.\n"
    )
    system_no_input_prompt = (
        "Choose the answer that best answers the question. Explain your reasoning.\n"
    )


class MultipleChoiceConcisePrompter(AlpacaPrompter):
    """
    Prompter for multiple choice concise
    """

    system_prompt = "Choose the answer that best answers the question. Be concise in your response.\n\n"
    system_no_input_prompt = "Choose the answer that best answers the question. Be concise in your response.\n\n"

    def match_prompt_style(self):
        self.turn_format = "USER: {instruction}\n{input}\nASSISTANT:"
        self.turn_no_input_format = "USER: {instruction}\nASSISTANT:"


class SummarizeTLDRPrompter(AlpacaPrompter):
    """
    Prompter for summarize TLDR
    """

    system_prompt = ""
    system_no_input_prompt = ""

    def match_prompt_style(self):
        self.turn_format = "USER: Summarize the following article as a TL;DR.\n{instruction}\n{input}\nASSISTANT:"
        self.turn_no_input_format = "USER: Summarize the following article as a TL;DR.\n{instruction}\nASSISTANT:"


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


class GPTeacherPrompter(AlpacaPrompter):
    """
    Prompter for GPTeacher
    """


class NomicGPT4AllPrompter(AlpacaPrompter):
    """
    Prompter for NomicGPT4All
    """


class ReflectAlpacaPrompter:
    """
    Prompter for ReflectAlpaca
    """

    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. You, the Assistant, should generate a response as if it were an abstract for an academic or technical paper on the query along with a methodology. Then generate an Agent Reflection where you create a long form response as if from subject matter expert, be verbose, diligent, and creative in your application of knowledge, apply it through the lens of the response generated by the assistant. Look for flawed reasoning, faulty logic, or other mistakes in the method. Finally, generate a final response and method for the user with the Assistant abstract and Reflection analysis as augmentations to the generation\n\n"
    system_no_input_prompt = "Below is an instruction that describes a task. You, the Assistant, should generate a response as if it were an abstract for an academic or technical paper on the query along with a methodology. Then generate an Agent Reflection where you create a long form response as if from subject matter expert, be verbose, diligent, and creative in your application of knowledge, apply it through the lens of the response generated by the assistant. Look for flawed reasoning, faulty logic, or other mistakes in the method. Finally, generate a final response and method for the user with the Assistant abstract and Reflection analysis as augmentations to the generation\n\n"

    prompt_input = (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
    prompt_no_input = "### Instruction:\n{instruction}\n\n### Response:\n"
    agent_label = "### Thought:\n{output}\n\n### Agent Reflection:\n{reflection}\n\n### Final Response:\n{corrected}"
    response_split = "### Response:"

    def __init__(self, prompt_style="instruct"):
        self.prompt_style = prompt_style
        self.match_prompt_style()

    def match_prompt_style(self):
        if self.prompt_style == PromptStyle.INSTRUCT.value:
            self.prompt_input = (
                self.system_prompt
                + "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            )
            self.prompt_no_input = (
                self.system_no_input_prompt
                + "### Instruction:\n{instruction}\n\n### Response:\n"
            )
            self.agent_label = "### Thought:\n{output}\n\n### Agent Reflection:\n{reflection}\n\n### Final Response:\n{corrected}"
            self.response_split = "### Final Response:"
        if self.prompt_style == PromptStyle.CHAT.value:
            self.prompt_input = (
                self.system_prompt + "USER: {instruction}\n{input}\nASSISTANT:"
            )
            self.prompt_no_input = (
                self.system_no_input_prompt + "USER: {instruction}\nASSISTANT:"
            )
            self.agent_label = (
                "\nTHOUGHT: {output}\nASSISTANT REFLECTION: {reflection}\nASSISTANT:"
            )
            self.response_split = "ASSISTANT:"

    def build_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,  # pylint: disable=redefined-builtin
        output: Union[None, str] = None,
        reflection: Union[None, str] = None,
        corrected: Union[None, str] = None,
    ) -> Generator[str, None, None]:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.prompt_input.format(instruction=instruction, input=input)
        else:
            res = self.prompt_no_input.format(instruction=instruction)
        if output and reflection and corrected:
            label = self.agent_label.format(
                output=output,
                reflection=reflection,
                corrected=corrected,
            )
            res = f"{res}{label}"
        yield res


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    DOLLY = auto()


# TODO clean this 💩 up
@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: Optional[str] = None

    def get_prompt(self) -> Generator[Tuple[str, str], None, None]:
        # seps = [self.sep, self.sep2]
        preamble = self.system + self.sep
        yield ("SYSTEM:", preamble)
        for _, (role, message) in enumerate(self.messages):
            if message:
                yield (role + ":", " " + message)
            else:
                LOG.warning(f"role with empty message: {role}")
                yield (role + ":", "")

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
        )

    def append_message(self, role, message):
        self.messages.append([role, message])


SHAREGPT_ASSERTION_FAILED_ROLE = (
    "Role did not alternate between turns (gpt and human). Please check your data."
)


class ShareGPTPrompter:  # pylint: disable=too-few-public-methods
    """
    A prompter that generates prompts for the ShareGPT
    """

    def __init__(self, prompt_style=None, system_prompt: Optional[str] = None):
        if prompt_style != PromptStyle.CHAT.value:
            raise ValueError(
                f"unsupported prompt_style for ShareGPTPrompter({prompt_style})"
            )
        system: str = (
            system_prompt
            if system_prompt
            else (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            )
        )
        self._conversation = Conversation(
            system=system,
            roles=["USER", "ASSISTANT"],
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2=" ",
        )

    def build_prompt(self, source) -> Generator[str, None, None]:
        # ignore the system prompt if provided
        if source[0]["from"] == "system":
            source.pop(0)

        if len(source) < 2:
            # If there isn't a back and forth conversation, ignore it
            # also happens on the data splitting leaving empty conversations
            raise IndexError

        conv = self._conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        try:
            # Apply prompt templates
            if (
                source[0]["from"] not in roles
                or roles[source[0]["from"]] != conv.roles[0]
            ):
                # Skip the first one if it is not from human
                source = source[1:]
        except IndexError as err:
            # sometimes there is a bing or system chat
            raise err

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], SHAREGPT_ASSERTION_FAILED_ROLE
            conv.append_message(role, sentence["value"])

        for part in conv.get_prompt():
            yield part
