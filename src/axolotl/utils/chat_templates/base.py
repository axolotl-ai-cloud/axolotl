"""
utility functions for chat templates
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

LOG = get_logger("axolotl.utils.chat_templates")

_JINJA_TEMPLATE_CHOICE = "jinja"
_DEFAULT_TEMPLATE_CHOICE = "tokenizer_default"
_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX = "tokenizer_default_fallback_"

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_CHAT_TEMPLATES: dict[str, str] = {}
for filename in [f for f in os.listdir(TEMPLATE_DIR) if f.endswith(".jinja")]:
    with open(os.path.join(TEMPLATE_DIR, filename), "r", encoding="utf-8") as f:
        _CHAT_TEMPLATES[filename[:-6]] = f.read()


def get_chat_template(
    user_choice: str,
    jinja_template: str | None = None,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> str:
    """
    Finds the correct chat_template based on the user's choice, jinja_template, and tokenizer.

    Args:
        user_choice (str): The user's choice of template.
        jinja_template (str, optional): The jinja template string or Path to a valid jinja template file. Defaults to None.
        tokenizer (PreTrainedTokenizerBase, optional): The tokenizer. Defaults to None.

    Returns:
        str: The chosen template string.

    Raises:
        ValueError: If the user_choice is not found in the templates.
    """
    if user_choice == _JINJA_TEMPLATE_CHOICE:
        if not jinja_template:
            raise ValueError(
                f"`jinja_template` cannot be None when `chat_template` choice is {_JINJA_TEMPLATE_CHOICE}"
            )
        if os.path.exists(jinja_template) and os.path.isfile(jinja_template):
            with open(jinja_template, "r", encoding="utf-8") as file:
                jinja_template = file.read()
        return jinja_template

    if user_choice == _DEFAULT_TEMPLATE_CHOICE:
        if not tokenizer:
            raise ValueError(
                f"`tokenizer` cannot be None when chat_template choice is {_DEFAULT_TEMPLATE_CHOICE}"
            )
        if not tokenizer.chat_template:
            raise ValueError(
                f"`chat_template choice is {_DEFAULT_TEMPLATE_CHOICE} but tokenizer's chat_template is null. "
                f"Please add a chat_template in tokenizer config"
            )
        return tokenizer.chat_template  # type: ignore

    if user_choice.startswith(_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX):
        if not tokenizer:
            raise ValueError(
                f"`tokenizer` cannot be None when chat_template choice starts with {_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX}"
            )
        if tokenizer.chat_template:
            return tokenizer.chat_template  # type: ignore

        user_choice = user_choice[
            len(_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX) :
        ]
        LOG.warning(
            f"No chat template found on tokenizer, falling back to {user_choice}. It is recommended to set --train_on_inputs to True for the model to learn this chat template."
        )

    if user_choice in _CHAT_TEMPLATES:
        return _CHAT_TEMPLATES[user_choice]

    raise ValueError(f"Template '{user_choice}' not found.")


def extract_chat_template_args(cfg, ds_cfg: Dict[str, Any] | None = None):
    if ds_cfg and ds_cfg.get("chat_template"):
        chat_template_choice = ds_cfg.get("chat_template") or _DEFAULT_TEMPLATE_CHOICE
        chat_template_jinja = ds_cfg.get("chat_template_jinja")
    else:
        chat_template_choice = cfg.get("chat_template") or _DEFAULT_TEMPLATE_CHOICE
        chat_template_jinja = cfg.get("chat_template_jinja")
    return chat_template_choice, chat_template_jinja


def get_chat_template_from_config(
    cfg,
    ds_cfg: Dict[str, Any] | None = None,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> str:
    chat_template_choice, chat_template_jinja = extract_chat_template_args(
        cfg=cfg, ds_cfg=ds_cfg
    )
    return get_chat_template(
        user_choice=chat_template_choice,
        jinja_template=chat_template_jinja,
        tokenizer=tokenizer,
    )


def register_chat_template(template_name: str, chat_template: str):
    """
    Registers chat templates.

    Args:
        template_name (str): The name of the template.
        chat_template (str): The template string.
    """

    if template_name in _CHAT_TEMPLATES:
        raise ValueError(f"Template '{template_name}' already exists.")

    _CHAT_TEMPLATES[template_name] = chat_template
