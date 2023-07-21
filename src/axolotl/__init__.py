"""The Axolotl module"""

from axolotl import cli
from axolotl.utils.dict import DictDefault
from axolotl.version import VERSION

__version__ = ".".join(map(str, VERSION))

cfg: DictDefault = DictDefault()

__all__ = ["cli", "cfg"]
