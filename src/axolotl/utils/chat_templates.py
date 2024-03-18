"""
This module provides functionality for selecting chat templates based on user choices.
These templates are used for formatting messages in a conversation.
"""


def chat_templates(user_choice: str):
    """
    Finds the correct chat_template for the tokenizer_config.

    Args:
        user_choice (str): The user's choice of template.

    Returns:
        str: The chosen template string.

    Raises:
        ValueError: If the user_choice is not found in the templates.
    """

    templates = {
        "alpaca": "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Instruction: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Response: ' + message['content'] + eos_token}}{% endif %}{% endfor %}",
        "inst": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",  # I don't know what this one is called. Used by Mistral/Mixtral.
        "chatml": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "gemma": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}",
    }

    if user_choice in templates:
        return templates[user_choice]

    raise ValueError(f"Template '{user_choice}' not found.")
