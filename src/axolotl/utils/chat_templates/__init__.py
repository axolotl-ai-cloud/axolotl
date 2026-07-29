"""
This module provides functionality for selecting chat templates based on user choices.
These templates are used for formatting messages in a conversation.
"""

from .base import (
    _CHAT_TEMPLATES,
    extract_chat_template_args,
    get_chat_template,
    get_chat_template_from_config,
    register_chat_template,
)
from .kwargs_mix import (
    ChatTemplateKwargsMix,
    ChatTemplateKwargsMixEntry,
    build_chat_template_kwargs_mix,
)

__all__ = [
    "get_chat_template",
    "extract_chat_template_args",
    "get_chat_template_from_config",
    "register_chat_template",
    "ChatTemplateKwargsMix",
    "ChatTemplateKwargsMixEntry",
    "build_chat_template_kwargs_mix",
    "_CHAT_TEMPLATES",
]
