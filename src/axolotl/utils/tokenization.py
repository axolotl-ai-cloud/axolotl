"""Module for tokenization utilities"""

import logging
import re
from typing import Dict, List

from termcolor import colored

LOG = logging.getLogger("axolotl")


def check_dataset_labels(
    dataset,
    tokenizer,
    num_examples=5,
    text_only=False,
    rl_mode=False,
):
    # the dataset is already shuffled, so let's just check the first 5 elements
    for idx in range(num_examples):
        if not rl_mode:
            check_example_labels(dataset[idx], tokenizer, text_only=text_only)
        else:
            check_rl_example_labels(dataset[idx], tokenizer, text_only=text_only)


def check_example_labels(example, tokenizer, text_only=False):
    # Get the input_ids, labels, and attention_mask from the dataset
    input_ids = example["input_ids"]
    labels = example["labels"]

    # You can compare the input_ids and labels element-wise
    # Remember to ignore positions with IGNORE_TOKEN_ID (if you use it) or attention_mask equal to 0
    colored_tokens = []
    for _, (input_id, label_id) in enumerate(zip(input_ids, labels)):
        decoded_input_token = tokenizer.decode(input_id)
        # Choose the color based on whether the label has the ignore value or not
        color = "red" if label_id == -100 else ("yellow" if label_id == 0 else "green")
        colored_token = colored(decoded_input_token, color) + (
            not text_only and colored(f"({label_id}, {input_id})", "white") or ""
        )
        colored_tokens.append(colored_token)

    delimiter = "" if text_only else " "
    LOG.info(delimiter.join(colored_tokens))
    LOG.info("\n\n\n")

    return " ".join(colored_tokens)


def color_token_for_rl_debug(decoded_token, encoded_token, color, text_only):
    """Helper function to color tokens based on their type."""
    colored_text = colored(decoded_token, color)
    return (
        colored_text
        if text_only
        else f"{colored_text}{colored(f'({encoded_token})', 'white')}"
    )


def process_tokens_for_rl_debug(tokens, color, tokenizer, text_only):
    """Helper function to process and color tokens."""
    colored_tokens = [
        color_token_for_rl_debug(tokenizer.decode(token), token, color, text_only)
        for token in tokenizer.encode(tokens)
    ]
    return colored_tokens


def check_rl_example_labels(example, tokenizer, text_only=False):
    field_prompt, field_chosen, field_rejected = "prompt", "chosen", "rejected"

    input_tokens = example[field_prompt]
    labels_chosen, labels_rejected = example[field_chosen], example[field_rejected]

    # Process and color each type of token
    colored_tokens = process_tokens_for_rl_debug(
        input_tokens, "yellow", tokenizer, text_only
    )
    colored_chosens = process_tokens_for_rl_debug(
        labels_chosen, "green", tokenizer, text_only
    )
    colored_rejecteds = process_tokens_for_rl_debug(
        labels_rejected, "red", tokenizer, text_only
    )

    # Create a delimiter based on text_only flag
    delimiter = "" if text_only else " "

    # Logging information
    LOG.info(f"INPUT PROMPT: {delimiter.join(colored_tokens)}\n\n")
    LOG.info(f"CHOSEN RESPONSE: {delimiter.join(colored_chosens)}\n\n")
    LOG.info(f"REJECTED RESPONSE: {delimiter.join(colored_rejecteds)}\n\n\n")

    return delimiter.join(colored_tokens)


GLAIVE_ROLES = ["USER", "ASSISTANT", "FUNCTION RESPONSE"]
GLAIVE_TO_SHAREGPT_ROLE = {
    "SYSTEM": "system",
    "USER": "human",
    "ASSISTANT": "gpt",
    "FUNCTION RESPONSE": "tool",
}

GLAIVE_MSG_REGEX = re.compile(rf"({'|'.join(GLAIVE_ROLES)}): ")


def chatml_to_conversation(row: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Converts a ChatML formatted row to a list of messages in ShareGPT format.
    Initially based off https://github.com/lilacai/lilac/blob/main/notebooks/GlaiveToShareGPT.ipynb.
    """

    system_prompt = row.get("system")
    if system_prompt:
        system_prompt = system_prompt.removeprefix("SYSTEM: ")

    chat_str = row["chat"]
    chat_msgs = [s.strip() for s in GLAIVE_MSG_REGEX.split(chat_str) if s]

    chat_msg_dicts = [
        {"from": GLAIVE_TO_SHAREGPT_ROLE[role], "value": value}
        for role, value in zip(chat_msgs[::2], chat_msgs[1::2])
    ]

    if system_prompt:
        chat_msg_dicts = [
            {"from": GLAIVE_TO_SHAREGPT_ROLE["SYSTEM"], "value": system_prompt}
        ] + chat_msg_dicts

    return chat_msg_dicts


def merge_consecutive_messages(messages):
    """
    Merge consecutive messages from the same sender into a single message.
    This can be useful with datasets that contain multiple consecutive tool calls.
    """

    merged_messages = []
    current_from = None
    current_message = ""

    for msg in messages:
        if current_from == msg["from"]:
            current_message += msg["value"]
        else:
            if current_from is not None:
                merged_messages.append({"from": current_from, "value": current_message})
            current_from = msg["from"]
            current_message = msg["value"]

    if current_from is not None:
        merged_messages.append({"from": current_from, "value": current_message})

    return merged_messages
