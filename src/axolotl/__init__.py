"""Axolotl - Train and fine-tune large language models"""

try:
    from importlib.metadata import version

    __version__ = version("axolotl")
except ImportError:
    __version__ = "unknown"
