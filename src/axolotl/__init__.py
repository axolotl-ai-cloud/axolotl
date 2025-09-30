"""Axolotl - Train and fine-tune large language models."""

import pkgutil

from ._version import __version__

__path__ = pkgutil.extend_path(__path__, __name__)
__all__ = ["__version__"]
