"""
This module provides functionality for selecting chat templates based on user choices.
These templates are used for formatting messages in a conversation.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

LOG = logging.getLogger("axolotl.utils.chat_templates")

_JINJA_TEMPALTE_CHOICE = "jinja"
_DEFAULT_TEMPLATE_CHOICE = "tokenizer_default"
_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX = "tokenizer_default_fallback_"

_TEMPLATES = {
    "alpaca": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Instruction: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Response: ' + message['content'] + eos_token}}{% endif %}{% endfor %}",
    "inst": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",  # I don't know what this one is called. Used by Mistral/Mixtral.
    "mistral": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] | trim + eos_token }}{% endif %}{% endfor %}",
    "chatml": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
    "gemma": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}",
    "cohere": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true %}{% set loop_messages = messages %}{% set system_message = 'You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% if system_message != false %}{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% elif message['role'] == 'assistant' %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}{% endif %}",
    "llama3": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
    "phi_3": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
    "deepseek_v2": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<｜User｜>' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<｜Assistant｜>' }}{% endif %}",
}


def get_chat_template(
    user_choice: str,
    jinja_template: Optional[str] = None,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
):
    """
    Finds the correct chat_template for the tokenizer_config.

    Args:
        user_choice (str): The user's choice of template.

    Returns:
        str: The chosen template string.

    Raises:
        ValueError: If the user_choice is not found in the templates.
    """
    if user_choice == _JINJA_TEMPALTE_CHOICE:
        if not jinja_template:
            raise ValueError(
                f"`jinja_template` cannot be None when `chat_template` choice is {_JINJA_TEMPALTE_CHOICE}"
            )
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
        return tokenizer.chat_template

    if user_choice.startswith(_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX):
        if not tokenizer:
            raise ValueError(
                f"`tokenizer` cannot be None when chat_template choice starts with {_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX}"
            )
        if tokenizer.chat_template:
            return tokenizer.chat_template

        user_choice = user_choice[
            len(_DEFAULT_FALLBACK_CHATML_TEMPLATE_CHOICE_PREFIX) :
        ]
        LOG.warning(
            f"No chat template found on tokenizer, falling back to {user_choice}. It is recommended to set --train_on_inputs to True for the model to learn this chat template."
        )

    if user_choice in _TEMPLATES:
        return _TEMPLATES[user_choice]

    raise ValueError(f"Template '{user_choice}' not found.")


def extract_chat_template_args(cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    if ds_cfg and ds_cfg.get("chat_template"):
        chat_template_choice = ds_cfg.get("chat_template") or "chatml"
        chat_template_jinja = ds_cfg.get("chat_template_jinja")
    else:
        chat_template_choice = cfg.get("chat_template") or "chatml"
        chat_template_jinja = cfg.get("chat_template_jinja")
    return chat_template_choice, chat_template_jinja


def get_chat_template_from_config(
    cfg,
    ds_cfg: Optional[Dict[str, Any]] = None,
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
) -> str:
    ds_cfg = ds_cfg or {}
    chat_template_choice, chat_template_jinja = extract_chat_template_args(
        cfg=cfg, ds_cfg=ds_cfg
    )
    return get_chat_template(
        user_choice=chat_template_choice,
        jinja_template=chat_template_jinja,
        tokenizer=tokenizer,
    )
