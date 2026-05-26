"""Shared utility helpers for MCP servers.

This module starts as a placeholder in Phase 0 and will be expanded in later phases.
"""

from __future__ import annotations


def health() -> dict[str, str]:
    """Return a lightweight health payload for basic checks."""
    return {"status": "ok"}
