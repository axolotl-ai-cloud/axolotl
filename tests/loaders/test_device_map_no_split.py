"""`infer_auto_device_map` must receive `_no_split_modules` as a list.

transformers exposes `_no_split_modules` as a `set`. Accelerate wraps any
non-list/tuple in a list, so a set arrives as ``[{"Block"}]`` and never matches a
class name -- indivisible blocks then get split across devices and the model
raises on the first cross-device op.
"""

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from torch import nn


class Block(nn.Module):
    """Stands in for a decoder layer: must not be split across devices."""

    def __init__(self, dim: int):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias=False)
        self.b = nn.Linear(dim, dim, bias=False)


class Tiny(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(depth))


def _device_map(no_split_module_classes):
    with init_empty_weights():
        model = Tiny()
    # tight enough that placement must span both devices
    per_block = 256 * 256 * 4 * 2  # two fp32 [dim, dim] weights
    budget = int(per_block * 1.5)  # an int is raw bytes
    return infer_auto_device_map(
        model,
        max_memory={0: budget, 1: budget, "cpu": int(1e9)},
        dtype=torch.float32,
        no_split_module_classes=no_split_module_classes,
    )


def _split_blocks(device_map) -> list[str]:
    """Keys naming something inside a Block, i.e. a Block that got split."""
    return [key for key in device_map if key.count(".") > 1]


def test_list_form_keeps_blocks_intact():
    assert not _split_blocks(_device_map(["Block"]))


def test_set_form_is_why_the_loader_must_cast():
    """Guards the reason for the ``list(...)`` cast, so removing it fails loudly."""
    assert _split_blocks(_device_map({"Block"}))
