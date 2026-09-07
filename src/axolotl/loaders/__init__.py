"""Init for axolotl.loaders module"""

# flake8: noqa

from axolotl.utils import make_lazy_getattr

from .constants import MULTIMODAL_AUTO_MODEL_MAPPING

__all__ = [
    "MULTIMODAL_AUTO_MODEL_MAPPING",
    "ModelLoader",
    "load_adapter",
    "load_lora",
    "load_processor",
    "load_tokenizer",
]

_LAZY_IMPORTS = {
    "ModelLoader": ".model",
    "load_adapter": ".adapter",
    "load_lora": ".adapter",
    "load_processor": ".processor",
    "load_tokenizer": ".tokenizer",
}

__getattr__ = make_lazy_getattr(_LAZY_IMPORTS, __name__, globals())
