"""Init for ring attention monkeypatch module"""

# flake8: noqa

from .patch import (
    get_ring_attn_group,
    register_ring_attn_from_device_mesh,
    set_ring_attn_group,
    update_ring_attn_params,
)

__all__ = (
    "get_ring_attn_group",
    "register_ring_attn_from_device_mesh",
    "set_ring_attn_group",
    "update_ring_attn_params",
)
