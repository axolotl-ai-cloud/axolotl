"""
helper util to calculate dataset lengths
"""

import numpy as np


def get_dataset_lengths(dataset, from_arrow=False):
    if "length" in dataset.column_names:
        lengths = np.array(dataset["length"])
    elif "position_ids" in dataset.column_names:
        position_ids = dataset["position_ids"]
        lengths = np.array([x[-1] + 1 for x in position_ids])
    else:
        if from_arrow:
            input_ids = dataset.data.column("input_ids")
            lengths = np.vectorize(len)(np.array(input_ids, dtype=object))
        else:
            input_ids = dataset["input_ids"]
            lengths = np.array([len(seq) for seq in input_ids])
    return lengths


def get_dataset_label_counts(dataset, shift_labels=True):
    """Read loss-bearing label counts and causal row-start corrections in chunks."""
    column = "shift_labels" if "shift_labels" in dataset.column_names else "labels"
    if column not in dataset.column_names:
        raise ValueError("Label-balanced packing requires tokenized labels")
    counts = np.empty(len(dataset), dtype=np.int64)
    starts = np.zeros(len(dataset), dtype=np.int64)
    offset = 0
    for batch in dataset.select_columns([column]).iter(batch_size=1024):
        for labels in batch[column]:
            labels = np.asarray(labels)
            if labels.ndim != 1:
                raise ValueError(
                    "Label-balanced packing requires one-dimensional labels"
                )
            counts[offset] = np.count_nonzero(labels != -100)
            if shift_labels and column == "labels" and labels.size:
                starts[offset] = int(labels[0] != -100)
            offset += 1
    return counts, starts
