"""
DPO prompt strategies passthrough/zero-processing strategy
"""


def default(cfg, dataset_idx=0, **kwargs):
    def transform_fn(sample, tokenizer=None):
        return sample

    return transform_fn
