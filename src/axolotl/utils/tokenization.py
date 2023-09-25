"""Module for tokenization utilities"""


import logging

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

    LOG.info(" ".join(colored_tokens))
    LOG.info("\n\n\n")
    print(" ".join(colored_tokens))

    return " ".join(colored_tokens)
