"""Module for tokenization utilities"""

import logging

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
    target_mask = example.pop("target_mask", None)

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
    target_labels_count = sum(label_id != -100 for label_id in labels)
    total_len = len(input_ids)
    LOG.info(f"Total input len: {total_len}")
    LOG.info(f"Count of labels: {target_labels_count}")
    if target_mask:
        target_mask_positions = sum(m[0] for m in target_mask)
        LOG.info(f"Number of positions in target_mask: {target_mask_positions}")

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
        for token in tokenizer.encode(tokens, add_special_tokens=False)
    ]
    return colored_tokens


def check_rl_example_labels(example, tokenizer, text_only=False):
    field_prompt, field_chosen, field_rejected, field_completion = (
        "prompt",
        "chosen",
        "rejected",
        "completion",
    )

    input_tokens = example[field_prompt]

    labels_chosen = example.get(field_chosen)
    labels_rejected = example.get(field_rejected)
    labels_completion = example.get(field_completion)

    # Create a delimiter based on text_only flag
    delimiter = "" if text_only else " "

    # Process and color each type of token
    colored_tokens = process_tokens_for_rl_debug(
        input_tokens, "yellow", tokenizer, text_only
    )

    # Process tokens
    if labels_completion is None:
        colored_chosens = process_tokens_for_rl_debug(
            labels_chosen, "green", tokenizer, text_only
        )
        colored_rejecteds = process_tokens_for_rl_debug(
            labels_rejected, "red", tokenizer, text_only
        )
    else:
        colored_completion = process_tokens_for_rl_debug(
            labels_completion, "green", tokenizer, text_only
        )

    # Logging information
    LOG.info(f"INPUT PROMPT: {delimiter.join(colored_tokens)}\n\n")

    if labels_completion is None:
        LOG.info(f"CHOSEN RESPONSE: {delimiter.join(colored_chosens)}\n\n")
        LOG.info(f"REJECTED RESPONSE: {delimiter.join(colored_rejecteds)}\n\n\n")
    else:
        LOG.info(f"COMPLETION RESPONSE: {delimiter.join(colored_completion)}\n\n\n")

    return delimiter.join(colored_tokens)
