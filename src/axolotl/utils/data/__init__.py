"""Init for `axolotl.utils.data` module."""

from axolotl.utils import make_lazy_getattr

__all__ = [
    "encode_streaming",
    "wrap_streaming_dataset",
    "prepare_preference_datasets",
    "get_dataset_wrapper",
    "prepare_datasets",
    "md5",
]

# lazy so streaming DataLoader workers importing a submodule skip the sft/rl loaders
_LAZY_IMPORTS = {
    "encode_streaming": ".streaming",
    "wrap_streaming_dataset": ".streaming",
    "prepare_preference_datasets": ".rl",
    "get_dataset_wrapper": ".wrappers",
    "prepare_datasets": ".sft",
    "md5": ".utils",
}

__getattr__ = make_lazy_getattr(_LAZY_IMPORTS, __name__, globals())
