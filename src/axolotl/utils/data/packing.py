"""Sample-packing dataset transforms, light enough for streaming DataLoader workers."""

from functools import partial


def add_position_ids(sample):
    """
    Handle both single-example and batched data.
    - single example: sample['input_ids'] is a list[int]
    - batched data: sample['input_ids'] is a list[list[int]]
    """
    # Return sample unchanged if "input_ids" is not present, or is empty
    if "input_ids" not in sample or not sample["input_ids"]:
        return sample

    input_ids = sample["input_ids"]

    # If first element is an int, it’s a single example
    # If first element is a list, it’s a batch
    if isinstance(input_ids[0], int):
        # ---- SINGLE EXAMPLE ----
        seq_len = len(input_ids)
        # Position IDs for a single example
        # As a list
        sample["position_ids"] = list(range(seq_len))
        sample["length"] = seq_len

    else:
        # ---- BATCHED EXAMPLES ----
        # input_ids is a list of lists
        position_ids_batch = []
        lengths_batch = []
        for seq in input_ids:
            seq_len = len(seq)
            position_ids_batch.append(list(range(seq_len)))
            lengths_batch.append(seq_len)

        # Now store them back
        sample["position_ids"] = position_ids_batch
        sample["length"] = lengths_batch

    return sample


def filter_sequences_by_length(
    sample, sequence_len=2048, min_sequence_len=2, raise_on_drop=False
):
    """
    Filter sequences outside valid length range [min_sequence_len, sequence_len].

    Drops samples that are either too short (< min_sequence_len) or too long (> sequence_len).

    Works for both single-example (list[int]) or batched (list[list[int]]).

    If raise_on_drop is set, the code raises a ValueError if a sample is
    encountered that is too long and would have been dropped.
    """
    min_sequence_len = min_sequence_len or 2

    input_ids = sample["input_ids"]

    # Edge case: if input_ids is empty
    if not input_ids:
        # Decide if you want to drop or keep empty. Let's drop.
        return False

    # Check if single example or batched by looking at the first element
    if isinstance(input_ids[0], int):
        # Single example (input_ids is a list of int)
        length = len(input_ids)
        if raise_on_drop and length > sequence_len:
            raise ValueError(
                f"Sequence encountered with {length} tokens, which exceeds the maximum {sequence_len}."
            )
        return min_sequence_len <= length <= sequence_len

    # Batched (input_ids is a list of lists)
    results = []
    for seq in input_ids:
        length = len(seq)
        if raise_on_drop and length > sequence_len:
            raise ValueError(
                f"Sequence encountered with {length} tokens, which exceeds the maximum {sequence_len}."
            )
        results.append(min_sequence_len <= length <= sequence_len)
    return results


def process_pretraining_datasets_for_packing(
    train_dataset, sequence_len, skip_position_ids=True, drop_attention_mask=False
):
    drop_outside_range = partial(filter_sequences_by_length, sequence_len=sequence_len)

    train_dataset = train_dataset.filter(
        drop_outside_range,
        desc="Dropping Long Sequences",
        load_from_cache_file=False,
    )
    if not skip_position_ids:
        train_dataset = train_dataset.map(
            add_position_ids,
            batched=True,
            desc="Add position_id column (Pretraining Sample Packing)",
        )
    if drop_attention_mask:
        train_dataset = train_dataset.remove_columns("attention_mask")

    return train_dataset
