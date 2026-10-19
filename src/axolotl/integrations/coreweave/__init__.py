"""CoreWeave checkpoint publishing and rollout integration."""


def __getattr__(name):
    if name == "CoreWeavePlugin":
        from .plugin import CoreWeavePlugin

        return CoreWeavePlugin
    raise AttributeError(name)
