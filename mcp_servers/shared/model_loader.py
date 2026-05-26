"""Shared model loading helpers for MCP servers.

This module starts as a placeholder in Phase 0 and will be implemented in Phase 1.
"""

from __future__ import annotations


class ModelLoader:
    """Placeholder model loader used during scaffold stage."""

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized
