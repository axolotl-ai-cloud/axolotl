"""Offline projection sampling for supervised finetuning."""

__all__ = ["ProjectionSamplingPlugin"]


def __getattr__(name):
    if name == "ProjectionSamplingPlugin":
        from .plugin import ProjectionSamplingPlugin

        return ProjectionSamplingPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
