"""Module for tokenization utilities"""


import logging
import re
from typing import Dict, List

from termcolor import colored

LOG = logging.getLogger("axolotl")


def check_dataset_labels(dataset, tokenizer, num_examples=5, text_only=False):
    # the dataset is already shuffled, so let's just check the first 5 elements
    for idx in range(num_examples):
        check_example_labels(dataset[idx], tokenizer, text_only=text_only)


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
