"""Init for the context-parallel group registry"""

# flake8: noqa

from .patch import get_ring_attn_group, set_ring_attn_group

__all__ = (
    "get_ring_attn_group",
    "set_ring_attn_group",
)
