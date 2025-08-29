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
            isFrozen = hasattr(self, "__frozen") and object.__getattribute__(
                self, "__frozen"
            )
        except AttributeError:
            isFrozen = False

        if isFrozen and name not in super().keys():
            raise KeyError(name)
        super(Dict, self).__setitem__(name, value)
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


def remove_none_values(obj):
    """
    Remove null from a dictionary-like obj or list.
    These can appear due to Dataset loading causing schema merge.
    See https://github.com/axolotl-ai-cloud/axolotl/pull/2909
    """
    if hasattr(obj, "items"):
        return {k: remove_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [remove_none_values(elem) for elem in obj]
    return obj
