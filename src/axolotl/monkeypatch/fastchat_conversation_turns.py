"""
monkeypatch to add a get_turns method
"""

import logging
from typing import Generator, Tuple

LOG = logging.getLogger("axolotl.monkeypatch.fastchat_conversation_turns")


def get_turns(self) -> Generator[Tuple[str, str], None, None]:
    # seps = [self.sep, self.sep2]
    preamble = self.system_message + self.sep
    yield ("SYSTEM:", preamble)
    for _, (role, message) in enumerate(self.messages):
        if message:
            yield (role + ":", " " + message)
        else:
            LOG.warning(f"role with empty message: {role}")
            yield (role + ":", "")


def add_get_turns_to_conversation():
    import fastchat.conversation

    fastchat.conversation.Conversation.get_turns = get_turns
