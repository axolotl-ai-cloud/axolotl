"""
helper functions for fixing the embeddings/tokenizer
"""

# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# GNU LESSER GENERAL PUBLIC LICENSE
# Version 3, 29 June 2007
#
# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.

import gc
import itertools
import logging
from collections import Counter

import datasets
import numpy as np
import torch

LOG = logging.getLogger("axolotl.core.tokenizer_utils")


@torch.inference_mode()
def fix_untrained_tokens(  # pylint: disable=too-many-return-statements
    model, tokenizer, train_dataset, ignored_tokenizer_names=None, eps=1e-16
):
    """
    Llama-3 for eg has untrained vectors in the base model.
    These include <|eot_id|>, <|start_header_id|>, <|end_header_id|>
    We reset them to the mean of the rest of the tokens
    """
    # Code licensed under LGPL
    embedding_matrix = model.get_input_embeddings().weight
    lm_head_matrix = model.get_output_embeddings().weight
    chat_template = getattr(tokenizer, "chat_template", None)
    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    # Ignore some model checks for now
    if not ignored_tokenizer_names:
        ignored_tokenizer_names = []
    if (
        model.config._name_or_path  # pylint: disable=protected-access
        in ignored_tokenizer_names
    ):
        return

    # Sometimes the sizes can be different like in vision models
    # Ie <image> is in input, but not in output
    min_size = min(embedding_matrix.shape[1], lm_head_matrix.shape[1])
    embedding_matrix = embedding_matrix[:, :min_size]
    lm_head_matrix = lm_head_matrix[:, :min_size]

    # Get untrained tokens
    indicator_untrained1 = torch.amax(embedding_matrix, axis=1) <= eps
    # Check lm_head as well

    # Does NOT work for Llama 3.1!!
    indicator_untrained2 = torch.amax(lm_head_matrix, axis=1) <= eps

    # We instead check for repeated vectors
    lm_head_where = torch.where(indicator_untrained1)[0]
    lm_head_bad = lm_head_matrix[lm_head_where]
    lm_head_bad = lm_head_bad.cpu().float().numpy().round(3)
    counter = Counter()
    for row in lm_head_bad:
        counter[hash(row.data.tobytes())] += 1
    counter = Counter({k: c for k, c in counter.items() if c >= 2})

    lm_head_where = lm_head_where.cpu().numpy()
    final_bad_lm_head = []
    for j, row in enumerate(lm_head_bad):
        if hash(row.data.tobytes()) in counter:
            final_bad_lm_head.append(lm_head_where[j])
    indicator_untrained2 = indicator_untrained2 | torch.zeros_like(indicator_untrained2)
    indicator_untrained2[final_bad_lm_head] = True

    # Combine both checks
    indicator_untrained = indicator_untrained1 & indicator_untrained2

    # Remove pad token possibility
    if hasattr(tokenizer, "pad_token_id"):
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None and pad_token_id < indicator_untrained.shape[0]:
            indicator_untrained[pad_token_id] = False

    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained

    # Get set and actual tokens
    where_untrained = where_untrained.tolist()
    if len(where_untrained) == 0:
        return

    # Remove untrained indices where it's longer
    where_untrained_set = frozenset(where_untrained)
    actual_bad_tokens = tokenizer.convert_ids_to_tokens(where_untrained)
    # Remove None items in actual_bad_tokens
    actual_bad_tokens = [x for x in actual_bad_tokens if x is not None]

    # Check if tokenizer and training datasets have bad tokens
    if_bad_first = False
    if_bad_second = False
    # Check tokenizer's chat template for any untrained tokens
    if chat_template is not None:
        if_bad_first = any(x in chat_template for x in actual_bad_tokens)

    if isinstance(train_dataset, datasets.IterableDataset):
        # Skip the check, since the code below assumes
        # an indexable dataset
        return

    # Check the first 250, last 250 input_ids
    size_dataset = len(train_dataset)
    size = min(size_dataset, 250)
    for j in range(size):
        input_ids = train_dataset[j]
        if "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]
            if_bad = any(item in where_untrained_set for item in input_ids)
            if if_bad:
                if_bad_second = True
                break

    # Check last 250
    if not if_bad_second:
        left = max(size_dataset - 250, 0)
        for j in range(left, size_dataset):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                if_bad = any(item in where_untrained_set for item in input_ids)
                if if_bad:
                    if_bad_second = True
                    break

    # Check if bad tokens exists!
    if not if_bad_first and not if_bad_second:
        return

    # Check if lm_head / embed_token are trainable!
    bad_not_trainable = False
    if not embedding_matrix.requires_grad:
        bad_not_trainable = True
    if not lm_head_matrix.requires_grad:
        bad_not_trainable = True

    if bad_not_trainable:  # pylint: disable=too-many-nested-blocks
        final_bad_items = []

        # Re-check the first 250, last 250 input_ids
        size_dataset = len(train_dataset)
        size = min(size_dataset, 250)
        for j in range(size):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                for item in input_ids:
                    if item in where_untrained_set:
                        final_bad_items.append(item)

        # Re-check last 250
        left = max(size_dataset - 250, 0)
        for j in range(left, size_dataset):
            input_ids = train_dataset[j]
            if "input_ids" in input_ids:
                input_ids = input_ids["input_ids"]
                for item in input_ids:
                    if item in where_untrained_set:
                        final_bad_items.append(item)

        # If no bad tokens, possibly chat template itself has issues?
        if len(final_bad_items) == 0:
            # Recheck 2000 and last 2000 items
            size_dataset = len(train_dataset)
            size = min(size_dataset, 2000)
            for j in range(size):
                input_ids = train_dataset[j]
                if "input_ids" in input_ids:
                    input_ids = input_ids["input_ids"]
                    for item in input_ids:
                        if item in where_untrained_set:
                            final_bad_items.append(item)

            # Re-check last 2000
            left = max(size_dataset - 2000, 0)
            for j in range(left, size_dataset):
                input_ids = train_dataset[j]
                if "input_ids" in input_ids:
                    input_ids = input_ids["input_ids"]
                    for item in input_ids:
                        if item in where_untrained_set:
                            final_bad_items.append(item)

            # Most likely false signal!
            if len(final_bad_items) == 0:
                return

        raise ValueError(
            f"Untrained tokens of [{list(set(final_bad_items))}] found, but embed_tokens & lm_head not trainable, causing NaNs. "
        )

    # Count all the possible bad tokens
    final_counts = np.zeros(
        max(len(tokenizer), embedding_matrix.shape[0]), dtype=np.int64
    )

    def mapping(examples):
        input_ids = examples["input_ids"]
        counter = np.fromiter(itertools.chain.from_iterable(input_ids), dtype=np.int32)
        np.add.at(final_counts, counter, 1)

    train_dataset.map(mapping, batched=True, desc="Counting untrained tokens")

    # Get counts for untrained tokens
    counts_untrained = final_counts[where_untrained]
    # Identify untrained tokens seen in train_dataset
    indices_seen_in_train = np.where(counts_untrained > 0)[0]
    tokens_to_update = [where_untrained[i] for i in indices_seen_in_train]

    if len(tokens_to_update) == 0:
        LOG.info(
            "No untrained tokens found in train_dataset. No embeddings were modified."
        )
        return

    # Log the token IDs that are being rescaled
    LOG.info(
        f"Rescaling embeddings for tokens seen in train_dataset: {tokens_to_update}"
    )

    # Get sum of all items
    sum_embedding = torch.sum(embedding_matrix, dtype=torch.float32, axis=0)
    sum_lm_head = torch.sum(lm_head_matrix, dtype=torch.float32, axis=0)

    # Remove bad tokens
    sum_embedding -= torch.sum(
        embedding_matrix[where_untrained], dtype=torch.float32, axis=0
    )
    sum_lm_head -= torch.sum(
        lm_head_matrix[where_untrained], dtype=torch.float32, axis=0
    )

    # Find correct average by dividing by sum of trained tokens
    mean_embedding = sum_embedding / n_trained
    mean_lm_head = sum_lm_head / n_trained

    # Compute scaling for tokens to update
    scaling = counts_untrained[indices_seen_in_train] / max(final_counts.max(), 1)
    scaling = torch.tensor(scaling, device=mean_embedding.device).unsqueeze(1)

    # Prepare mean embeddings for tokens to update
    mean_embedding_repeated = (
        mean_embedding.unsqueeze(0).repeat(len(tokens_to_update), 1) * scaling
    )
    mean_lm_head_repeated = (
        mean_lm_head.unsqueeze(0).repeat(len(tokens_to_update), 1) * scaling
    )

    # Update embeddings only for tokens seen in train_dataset
    embedding_matrix[tokens_to_update] = mean_embedding_repeated.to(
        embedding_matrix.dtype
    )
    lm_head_matrix[tokens_to_update] = mean_lm_head_repeated.to(lm_head_matrix.dtype)

    # Clean up
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    return
