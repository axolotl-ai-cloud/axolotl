from termcolor import colored
import logging


def check_dataset_labels(dataset, tokenizer):
    # the dataset is already shuffled, so let's just check the first 5 elements
    for idx in range(5):
        check_example_labels(dataset[idx], tokenizer)


def check_example_labels(example, tokenizer):
    # Get the input_ids, labels, and attention_mask from the dataset
    input_ids = example["input_ids"]
    labels = example["labels"]
    attention_mask = example["attention_mask"]

    # You can compare the input_ids and labels element-wise
    # Remember to ignore positions with IGNORE_TOKEN_ID (if you use it) or attention_mask equal to 0
    colored_tokens = []
    for i, (input_id, label_id, mask) in enumerate(
        zip(input_ids, labels, attention_mask)
    ):
        decoded_input_token = tokenizer.decode(input_id)
        # Choose the color based on whether the label has the ignore value or not
        color = "red" if label_id == -100 else ("yellow" if label_id == 0 else "green")
        colored_token = colored(decoded_input_token, color) + colored(
            f"({label_id}, {mask}, {input_id})", "white"
        )
        colored_tokens.append(colored_token)

    logging.info(" ".join(colored_tokens))
    logging.info("\n\n\n")
