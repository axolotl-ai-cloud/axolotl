"""
helper util to calculate dataset lengths
"""

import numpy as np


def get_dataset_lengths(dataset, from_arrow=False):
    column_names = dataset.column_names or []
    if "length" in column_names:
        lengths = np.array(dataset["length"])
    elif "position_ids" in column_names:
        position_ids = dataset["position_ids"]
        lengths = np.array([x[-1] + 1 for x in position_ids])
    elif "input_ids" not in column_names:
        raise ValueError(
            "Cannot compute sample lengths: dataset has none of `length`, "
            "`position_ids`, or `input_ids`. This usually means an unprepared "
            "(raw) dataset reached a sample-packing sampler, e.g. "
            "`skip_prepare_dataset`/`streaming` with `eval_sample_packing`."
        )
    else:
        if from_arrow:
            input_ids = dataset.data.column("input_ids")
            lengths = np.vectorize(len)(np.array(input_ids, dtype=object))
        else:
            input_ids = dataset["input_ids"]
            lengths = np.array([len(seq) for seq in input_ids])
    return lengths
