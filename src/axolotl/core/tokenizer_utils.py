"""
helper functions for fixing the embeddings/tokenizer
"""

# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import itertools

import numpy as np
import torch


@torch.inference_mode
def fix_untrained_tokens(model, tokenizer, train_dataset, eps=1e-16):
    """
    Many of the newer models have reserved tokens that are not trained.
    """
    embedding_matrix = model.get_input_embeddings().weight
    lm_head_matrix = model.get_output_embeddings().weight

    # Get untrained tokens
    indicator_untrained = torch.amax(embedding_matrix, axis=1) <= eps
    where_untrained = torch.where(indicator_untrained)[0]
    n_untrained = where_untrained.shape[0]
    n_trained = embedding_matrix.shape[0] - n_untrained

    # Get set and actual tokens
    where_untrained = where_untrained.tolist()
    if len(where_untrained) == 0:
        return False

    # Remove untrained indices where it's longer

    where_untrained_set = frozenset(where_untrained)
    actual_bad_tokens = tokenizer.convert_ids_to_tokens(where_untrained)
    # Remove None items in actual_bad_tokens
    actual_bad_tokens = [x for x in actual_bad_tokens if x is not None]

    # Check if tokenizer and training datasets have bad tokens
    if_bad_first = False
    if_bad_second = False
    # Check tokenizer's chat template for any untrained tokens
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is not None:
        if_bad_first = any(x in chat_template for x in actual_bad_tokens)

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
        return False

    # Count all the possible bad tokens
    final_counts = np.zeros(
        max(len(tokenizer), embedding_matrix.shape[0]), dtype=np.int64
    )

    def mapping(examples):
        input_ids = examples["input_ids"]
        counter = np.fromiter(itertools.chain.from_iterable(input_ids), dtype=np.int32)
        np.add.at(final_counts, counter, 1)

    train_dataset.map(mapping, batched=True, desc="Counting untrained tokens")

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

    # Scale each to be equal to 1/max_frequency. Also set some to 0 if none seen
    scaling = final_counts[where_untrained] / max(final_counts.max(), 1)
    scaling = torch.tensor(scaling, device=mean_embedding.device).unsqueeze(1)
    mean_embedding = (
        mean_embedding.repeat(
            (
                n_untrained,
                1,
            )
        )
        * scaling
    )
    mean_lm_head = (
        mean_lm_head.repeat(
            (
                n_untrained,
                1,
            )
        )
        * scaling
    )
    where_null = scaling.ravel() == 0
    mean_embedding[where_null] = 0
    mean_lm_head[where_null] = 0

    # Set them to the mean
    embedding_matrix[where_untrained] = mean_embedding.to(embedding_matrix.dtype)
    lm_head_matrix[where_untrained] = mean_lm_head.to(lm_head_matrix.dtype)

    # Clean up
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

    return True
