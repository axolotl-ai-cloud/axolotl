# constants.py
"""
This module contains constants and configuration dictionaries used for
datasets and other utilities in the Axolotl project, specifically for testing.
"""
# Configuration for Alpaca Messages Dataset
ALPACA_MESSAGES_CONFIG_OG = {
    "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
    "type": "chat_template.default",
    "chat_template": "llama3",
    "field_messages": "conversation",
    "field_chosen": "chosen",
    "field_rejected": "rejected",
    "message_field_role": "role",
    "message_field_content": "content",
    "roles": {
        "system": ["system"],
        "user": ["user"],
        "assistant": ["assistant"],
    },
}

# Revision configuration extending the original
ALPACA_MESSAGES_CONFIG_REVISION = ALPACA_MESSAGES_CONFIG_OG.copy()
ALPACA_MESSAGES_CONFIG_REVISION["revision"] = "ea82cff"


SPECIAL_TOKENS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
}
