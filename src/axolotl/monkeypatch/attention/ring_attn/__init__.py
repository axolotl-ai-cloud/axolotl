"""Init for ring attention monkeypatch module"""

# pylint: disable=unused-import
# flake8: noqa

from .patch import (
    RingAttnFunc,
    get_ring_attn_group,
    register_ring_attn,
    set_ring_attn_group,
    update_ring_attn_params,
)
