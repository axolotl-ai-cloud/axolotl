"""Axolotl - Train and fine-tune large language models."""

from __future__ import annotations

import pkgutil
from importlib import metadata

try:
    from ._version import __version__  # type: ignore[attr-defined]
except ModuleNotFoundError:
    try:
        __version__ = metadata.version("axolotl")
    except metadata.PackageNotFoundError:  # pragma: no cover
        __version__ = "0+unknown"

__path__ = pkgutil.extend_path(__path__, __name__)
__all__ = ["__version__"]
