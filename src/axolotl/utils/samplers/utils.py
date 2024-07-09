"""
helper util to calculate dataset lengths
"""
import numpy as np


def get_dataset_lengths(dataset):
    if "length" in dataset.data.column_names:
        lengths = np.array(dataset.data.column("length"))
    elif "position_ids" in dataset.data.column_names:
        position_ids = dataset.data.column("position_ids")
        lengths = np.array([x[-1] + 1 for x in position_ids])
    else:
        input_ids = dataset.data.column("input_ids")
        lengths = np.vectorize(len)(np.array(input_ids, dtype=object))
        return lengths
    return lengths


def plot_ascii_lengths_histogram(data, title, logger):
    max_value = max(data)
    bucket_width = 512
    bins = np.arange(0, max_value + bucket_width, bucket_width)
    histogram, _ = np.histogram(data, bins=bins)
    top = " ".join(("-" * 10, title, "-" * 10))
    bottom = "-" * len(top)
    logger.info(top)
    scale_factor = 40 / max(histogram)
    for i, value in enumerate(histogram):
        lower_bound = i * bucket_width
        upper_bound = (i + 1) * bucket_width - 1
        if value:
            hist_bar = "□" * int(value * scale_factor)
        else:
            hist_bar = "x"
        logger.info(f"{hist_bar} ({lower_bound}-{upper_bound} tokens, Count: {value})")
    logger.info(bottom)
    logger.info("\n")
