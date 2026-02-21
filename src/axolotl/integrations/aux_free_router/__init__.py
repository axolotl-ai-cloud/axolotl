"""Aux-loss-free (AFB) MoE router integration package."""

from .args import AuxFreeRouterArgs
from .plugin import AuxFreeMoEPlugin

__all__ = [
    "AuxFreeRouterArgs",
    "AuxFreeMoEPlugin",
]
