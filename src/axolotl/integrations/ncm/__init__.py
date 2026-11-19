"""Next-Concept Modeling: coarse-to-fine parallel text generation over a learned
multiscale residual VQ tokenizer (arXiv:2610.14162)."""

from .args import NCMArgs, NCMConfig
from .configuration import NCMGeneratorConfig, NCMTokenizerConfig
from .modeling import NCMGeneratorModel, NCMTokenizerModel
from .plugin import NCMPlugin

__all__ = [
    "NCMArgs",
    "NCMConfig",
    "NCMGeneratorConfig",
    "NCMGeneratorModel",
    "NCMPlugin",
    "NCMTokenizerConfig",
    "NCMTokenizerModel",
]
