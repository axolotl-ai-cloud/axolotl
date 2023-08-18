"""Module containing the DictDefault class"""

from addict import Dict


class DictDefault(Dict):
    """
    A Dict that returns None instead of returning empty Dict for missing keys.
    """

    def __missing__(self, key):
        return None

    def __or__(self, other):
        return DictDefault(super().__or__(other))
