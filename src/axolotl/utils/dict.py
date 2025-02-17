"""Module containing the DictDefault class"""

from addict import Dict


class DictDefault(Dict):
    """
    A Dict that returns None instead of returning empty Dict for missing keys.
    """

    def __missing__(self, key):
        return None

    def __or__(self, other):
        return DictDefault(super().__ror__(other))

    def __setitem__(self, name, value):
        # workaround for pickle/unpickle issues and __frozen not being available
        try:
            isFrozen = hasattr(  # pylint: disable=invalid-name
                self, "__frozen"
            ) and object.__getattribute__(self, "__frozen")
        except AttributeError:
            isFrozen = False  # pylint: disable=invalid-name

        if isFrozen and name not in super().keys():
            raise KeyError(name)
        super(Dict, self).__setitem__(name, value)  # pylint: disable=bad-super-call
        try:
            p = object.__getattribute__(self, "__parent")
            key = object.__getattribute__(self, "__key")
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, "__parent")
            object.__delattr__(self, "__key")
