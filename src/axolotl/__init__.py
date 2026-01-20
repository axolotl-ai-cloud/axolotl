"""Axolotl - Train and fine-tune large language models"""

import pkgutil
from importlib.metadata import PackageNotFoundError, version

__path__ = pkgutil.extend_path(__path__, __name__)  # Make this a namespace package

try:
    __version__ = version("axolotl")
except PackageNotFoundError:
    __version__ = "unknown"
