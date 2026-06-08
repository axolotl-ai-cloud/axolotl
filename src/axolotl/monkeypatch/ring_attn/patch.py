"""Context-parallel group registry.

Tracks the process group used for context parallelism so the rest of axolotl
(GRPO trainer, mamba/SSM CP correction, ``is_cp_active``) can find it. The
ringmaster ContextParallelPlugin sets this group; the actual ring/ulysses
attention is owned by ringmaster (the legacy ``ring_flash_attn`` kernel path has
been removed).
"""

import torch.distributed as dist

RING_ATTN_GROUP = None


def get_ring_attn_group() -> dist.ProcessGroup:
    """Getter for the context-parallel group on this rank."""
    if RING_ATTN_GROUP is None:
        raise RuntimeError("context-parallel group not registered")
    return RING_ATTN_GROUP


def set_ring_attn_group(ring_attn_group: dist.ProcessGroup | None):
    """Setter for the context-parallel group on this rank."""
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = ring_attn_group
