"""Module containing prompters"""

import logging
from enum import Enum
from typing import Generator, Optional, Union

from colorama import Fore
from fastchat.conversation import Conversation, get_conv_template

LOG = logging.getLogger("axolotl")
IGNORE_TOKEN_ID = -100
REPR_TEMPLATE = "\n<start>\n" + Fore.CYAN + "{full_prompt}" + Fore.RESET + "\n<end>\n"


class PromptStyle(Enum):
    """
    Enum for prompt styles
    """

    INSTRUCT = "instruct"
    CHAT = "chat"
    CHATML = "chatml"
    PHI = "phi"


class Prompter:
    """
    Base prompter class for all prompters
    """


class AlpacaPrompter(Prompter):
    """
    Base class for alpaca prompters
    """

    system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    system_no_input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    system_format: str = "{system}"
    turn_format: str
    turn_no_input_format: str
    prompt_style: Optional[str] = None

    def __init__(self, prompt_style: Optional[str] = PromptStyle.INSTRUCT.value):
        self.prompt_style = prompt_style if prompt_style else PromptStyle.INSTRUCT.value
        self.match_prompt_style()

    def match_prompt_style(self):
        # pylint: disable=duplicate-code
        if self.prompt_style == PromptStyle.INSTRUCT.value:
            self.turn_format = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            self.turn_no_input_format = (
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )
            self.system_format = "{system}\n\n"
        elif self.prompt_style == PromptStyle.CHAT.value:
            self.turn_format = "USER: {instruction}\n{input}\nASSISTANT:"
            self.turn_no_input_format = "USER: {instruction}\nASSISTANT:"
            self.system_format = "SYSTEM: {system}\n"
        elif self.prompt_style == PromptStyle.CHATML.value:
            self.turn_format = "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n<|im_start|>assistant\n"
            self.turn_no_input_format = (
                "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            )
            self.system_format = "<|im_start|>system\n{system}<|im_end|>\n"
        elif self.prompt_style == PromptStyle.PHI.value:
            self.turn_format = "<|user|>\n{instruction}<|end|>{input}<|assistant|>"
            self.turn_no_input_format = "<|user|>\n{instruction}<|end|><|assistant|>"
            self.system_format = "<|system|>{system}\n"

    def _build_result(self, instruction, input_text, output):
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input_text:
            res = (
                self.system_format.format(system=self.system_prompt)
                if self.system_prompt
                else ""
            ) + self.turn_format.format(instruction=instruction, input=input_text)
        else:
            res = (
                self.system_format.format(system=self.system_no_input_prompt)
                if self.system_no_input_prompt
                else ""
            ) + self.turn_no_input_format.format(instruction=instruction)
        if output:
            res = f"{res}{output}"

        return res

    def build_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,  # pylint: disable=redefined-builtin
        output: Union[None, str] = None,
    ) -> Generator[str, None, None]:
        yield self._build_result(instruction, input, output)

    def __repr__(self) -> str:
        return REPR_TEMPLATE.format(
            full_prompt=self._build_result("{instruction}", "{input}", "{output}")
        )


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


class GPTeacherPrompter(AlpacaPrompter):
    """
    Prompter for GPTeacher
    """


class NomicGPT4AllPrompter(AlpacaPrompter):
    """
    Prompter for NomicGPT4All
    """


class ReflectAlpacaPrompter(Prompter):
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

    def _build_result(
        self,
        instruction: str,
        input: Union[None, str] = None,  # pylint: disable=redefined-builtin
        output: Union[None, str] = None,
        reflection: Union[None, str] = None,
        corrected: Union[None, str] = None,
    ):
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

        return res

    def build_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,  # pylint: disable=redefined-builtin
        output: Union[None, str] = None,
        reflection: Union[None, str] = None,
        corrected: Union[None, str] = None,
    ) -> Generator[str, None, None]:
        # pylint: disable=duplicate-code
        yield self._build_result(
            instruction,
            input,
            output,
            reflection,
            corrected,
        )

    def __repr__(self) -> str:
        return REPR_TEMPLATE.format(
            full_prompt=self._build_result("{instruction}", "{input}", "{output}")
        )


SHAREGPT_ASSERTION_FAILED_ROLE = (
    "Role did not alternate between turns (gpt and human). Please check your data."
)

CONVERSATION_ROLE_FORMAT = {
    "chatml": "<|im_start|>{ROLE}",
    "zephyr": "<|{ROLE}|>",
    "vicuna_v1.1": "{ROLE}",
    "llama3": "<|start_header_id|>{ROLE}<|end_header_id|>",
}


class ShareGPTPrompter(Prompter):  # pylint: disable=too-few-public-methods
    """
    A prompter that generates prompts for the ShareGPT
    """

    role_key_human = "human"
    role_key_model = "gpt"
    # Optional, only used for tool usage datasets.
    role_key_tool: Optional[str] = None
    # Optional, role input/output mapping
    roles: Optional[dict] = None

    def __init__(
        self,
        prompt_style=None,  # pylint: disable=unused-argument
        conversation: Optional[Union[str, Conversation]] = None,
        role_key_human: Optional[str] = None,
        role_key_model: Optional[str] = None,
        role_key_tool: Optional[str] = None,
        roles: Optional[dict] = None,
    ):
        if conversation:
            if isinstance(conversation, Conversation):
                self._conversation = conversation
            else:
                self._conversation = get_conv_template(conversation)
        else:
            self._conversation = get_conv_template("vicuna_v1.1")
        if role_key_human:
            self.role_key_human = role_key_human
        if role_key_model:
            self.role_key_model = role_key_model
        if role_key_tool:
            self.role_key_tool = role_key_tool
        if roles:
            self.roles = roles

    def _build_result(self, source):
        if len(source) < 2:
            # If there isn't a back and forth conversation, ignore it
            # also happens on the data splitting leaving empty conversations
            raise IndexError(
                f"A conversation entry has less than 2 messages :\n{source}"
            )

        conv = self._conversation.copy()

        original_source = source.copy()
        # Add the conversation system prompt if provided, otherwise use the default one
        if source[0]["from"] == "system":
            conv.set_system_message(source[0]["value"])
            source.pop(0)

        roles = {self.role_key_human: conv.roles[0], self.role_key_model: conv.roles[1]}
        if self.role_key_tool:
            roles[self.role_key_tool] = conv.roles[2]

        try:
            # Apply prompt templates
            if source[0]["from"] not in roles:
                # Skip the first one if it is not from human
                source = source[1:]
        except IndexError as err:
            # sometimes there is a bing or system chat
            raise err

        conv.messages = []
        for _, sentence in enumerate(source):
            from_role = sentence["from"]
            if from_role in roles:
                role = roles[from_role]
            else:
                if self._conversation.name not in CONVERSATION_ROLE_FORMAT:
                    raise NotImplementedError(
                        f"Role ({role}) not in default roles, and {self._conversation.name} does not support role remapping yet."
                        "Please help us by creating an Issue to add support for this conversation type."
                    )

                role = CONVERSATION_ROLE_FORMAT[self._conversation.name].format(
                    ROLE=from_role
                )

            if len(conv.messages) > 0 and ((role == conv.messages[-1][0])):
                if (
                    role != "assistant"
                ):  # back to back assistant calls may be okay for tool calls
                    LOG.warning(f"{SHAREGPT_ASSERTION_FAILED_ROLE}: {sentence}")

            conv.append_message(role, sentence["value"])
        turns = list(conv.get_turns())
        original_source_length = len(original_source)
        assert len(turns) in [
            original_source_length - 1,
            original_source_length,
            original_source_length + 1,
        ]
        if len(turns) == original_source_length + 1:
            original_source = [{"weight": None}] + original_source
        elif len(turns) == original_source_length - 1:
            original_source = original_source[1:]
        return [
            (*turn, weight)
            for turn, weight in zip(
                turns,
                [
                    1 if "weight" not in e or e["weight"] is None else e["weight"]
                    for e in original_source
                ],
            )
        ]

    def build_prompt(self, source) -> Generator[str, None, None]:
        turns = self._build_result(source)

        for part in turns:
            if part[0] and not part[1]:
                LOG.warning(f"role with empty message: {part[0]}")
            yield part

    def __repr__(self) -> str:
        turns = self._build_result([{"from": "{from}", "value": "{value}"}])
        return "\n".join([REPR_TEMPLATE.format(full_prompt=part) for part in turns])


class ShareGPTPrompterV2(ShareGPTPrompter):
    """
    A V2 prompter that generates prompts for the ShareGPT
    """

    def __init__(
        self,
        conversation: Optional[Union[str, Conversation]] = None,
        role_key_human: Optional[str] = None,
        role_key_model: Optional[str] = None,
        role_key_tool: Optional[str] = None,
        roles: Optional[dict] = None,
    ):
        super().__init__(
            conversation=conversation,
            role_key_human=role_key_human,
            role_key_model=role_key_model,
            role_key_tool=role_key_tool,
            roles=roles,
        )


class UnsupportedPrompter(Prompter):
    """
    A dummy class for custom prompters
    """

    def __init__(self) -> None:
        pass

    def __repr__(self):
        return "Pre-tokenized or custom dataset types are unsupported for logging"
