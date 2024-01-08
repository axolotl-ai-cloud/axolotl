"""
helper util to calculate dataset lengths
"""
import numpy as np


def get_dataset_lengths(dataset):
    if "length" in dataset.data.column_names:
        lengths = np.array(dataset.data.column("length"))
    else:
        lengths = (
            dataset.data.column("position_ids")
            .to_pandas()
            .apply(lambda x: x[-1] + 1)
            .values
        )
    return lengths
