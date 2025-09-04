"""
DPO prompt strategies passthrough/zero-processing strategy
"""


def default(
    cfg, dataset_idx=0, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(
        sample, tokenizer=None
    ):  # pylint: disable=possibly-unused-variable,unused-argument
        return sample

    return transform_fn
