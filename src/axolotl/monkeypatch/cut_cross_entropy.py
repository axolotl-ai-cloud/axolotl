"""
Modeling module for apply cut_cross_entropy

https://github.com/apple/ml-cross-entropy
"""

import importlib


def check_cce_installed():
    cce_spec = importlib.util.find_spec("cut_cross_entropy")

    if cce_spec is None:
        raise ImportError(
            "Please install cut_cross_entropy with `pip install axolotl[cce]`"
        )

    cce_spec_transformers = importlib.util.find_spec("cut_cross_entropy.transformers")
    if cce_spec_transformers is None:
        raise ImportError(
            "Please install cut_cross_entropy with transformers with `pip install axolotl[cce]`"
        )


def patch_cut_cross_entropy(model_type: str):
    from cut_cross_entropy.transformers import cce_patch

    # The patch checks model_type internally
    cce_patch(model_type)


check_cce_installed()
