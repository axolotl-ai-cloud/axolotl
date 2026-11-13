"""
axolotl samplers module
"""

from axolotl.utils import make_lazy_getattr

__all__ = [
    "LabelBalancedRandomSampler",
    "MultipackBatchSampler",
    "get_dataset_lengths",
]

# lazy so packing pool children import samplers.bin_packing without torch/accelerate
_LAZY_IMPORTS = {
    "LabelBalancedRandomSampler": ".label_balanced",
    "MultipackBatchSampler": ".multipack",
    "get_dataset_lengths": ".utils",
}

__getattr__ = make_lazy_getattr(_LAZY_IMPORTS, __name__, globals())
