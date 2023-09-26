"""
monkeypatch to add a get_turns method
"""

import logging
from typing import Generator, Tuple

from fastchat.conversation import SeparatorStyle

LOG = logging.getLogger("axolotl.monkeypatch.fastchat_conversation_turns")


def get_prompt(self) -> str:
    ret = ""
    for role, msg in self.get_turns():
        ret += role + msg
    return ret


def get_turns(  # pylint: disable=too-many-return-statements
    self,
) -> Generator[Tuple[str, str], None, None]:
    """Get the prompt for generation."""
    system_prompt = self.system_template.format(system_message=self.system_message)
    if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
        yield "", system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + ": ", message + self.sep
            else:
                yield role + ":", ""
        return
    if self.sep_style == SeparatorStyle.ADD_COLON_TWO:
        seps = [self.sep, self.sep2]
        yield "", system_prompt + seps[0]
        for i, (role, message) in enumerate(self.messages):
            if message:
                yield role + ": ", message + seps[i % 2]
            else:
                yield role + ":", ""
        return
    if self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
        yield "", system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + ": ", message + self.sep
            else:
                yield role + ": ", ""  # must be end with a space
        return
    if self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
        yield "", "" if system_prompt == "" else system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + "\n", message + self.sep
            else:
                yield role + "\n", ""
        return
    if self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
        yield "", system_prompt
        for role, message in self.messages:
            if message:
                yield role, message + self.sep
            else:
                yield role, ""
        return
    if self.sep_style == SeparatorStyle.NO_COLON_TWO:
        seps = [self.sep, self.sep2]
        yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                yield role, message + seps[i % 2]
            else:
                yield role, ""
        return
    if self.sep_style == SeparatorStyle.RWKV:
        yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                yield role + ": ", message.replace("\r\n", "\n").replace(
                    "\n\n", "\n"
                ) + "\n\n"
            else:
                yield role + ":", ""
        return
    if self.sep_style == SeparatorStyle.LLAMA2:
        seps = [self.sep, self.sep2]
        if self.system_message:
            yield "", system_prompt
        else:
            yield "", "[INST] "
        for i, (role, message) in enumerate(self.messages[1:]):
            if message:
                yield role + " ", message + seps[i % 2]
            else:
                yield role, ""
        return
    if self.sep_style == SeparatorStyle.CHATGLM:
        # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
        # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
        round_add_n = 1 if self.name == "chatglm2" else 0
        if system_prompt:
            yield "", system_prompt + self.sep

        for i, (role, message) in enumerate(self.messages):
            if i % 2 == 0:
                yield "", f"[Round {i//2 + round_add_n}]{self.sep}"

            if message:
                yield f"{role}：", f"{message}{self.sep}"
            else:
                yield f"{role}：", ""
        return
    if self.sep_style == SeparatorStyle.CHATML:
        yield "", "" if system_prompt == "" else system_prompt + self.sep + "\n"
        for role, message in self.messages:
            if message:
                yield role + "\n", message + self.sep + "\n"
            else:
                yield role + "\n", ""
        return
    if self.sep_style == SeparatorStyle.CHATINTERN:
        # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
        seps = [self.sep, self.sep2]
        yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            prefix = "<s>" if i % 2 == 0 else ""
            if message:
                yield prefix + role + ":", message + seps[i % 2] + "\n"
            else:
                yield role + ":", ""
        return
    if self.sep_style == SeparatorStyle.DOLLY:
        seps = [self.sep, self.sep2]
        yield "", system_prompt
        for i, (role, message) in enumerate(self.messages):
            if message:
                suffix = "\n\n" if i % 2 == 1 else ""
                yield role + ":\n", message + seps[i % 2] + suffix
            else:
                yield role + ":\n", ""
        return
    if self.sep_style == SeparatorStyle.PHOENIX:
        yield "", system_prompt
        for role, message in self.messages:
            if message:
                yield role + ": ", "<s>" + message + "</s>"
            else:
                yield role + ": " + "<s>", ""
        return
    if self.sep_style == SeparatorStyle.ROBIN:
        yield "", system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + ":\n", message + self.sep
            else:
                yield role + ":\n", ""
        return
    if self.sep_style == SeparatorStyle.FALCON_CHAT:
        if self.system_message:
            yield "", system_prompt + self.sep
        for role, message in self.messages:
            if message:
                yield role + ": ", message + self.sep
            else:
                yield role + ":", ""
    else:
        raise ValueError(f"Invalid style: {self.sep_style}")


def add_get_turns_to_conversation():
    import fastchat.conversation

    fastchat.conversation.Conversation.get_turns = get_turns
    fastchat.conversation.Conversation.get_prompt = get_prompt
