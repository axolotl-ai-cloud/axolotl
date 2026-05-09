"""Unit tests for ProTrain + bitsandbytes quantization composability.

The M2 + M3 milestones (collapsed per the M0 spike report) drop the
``args.py`` validators that rejected ``load_in_8bit`` / ``load_in_4bit``
when the ProTrain plugin is active. The M0 spike showed both bnb param
types compose cleanly with the chunk manager in Mode A (all-persistent)
because their ``.data`` is a packed-byte tensor (``torch.int8`` for
``Int8Params``, ``torch.uint8`` for ``Params4bit``) that ``_param_bytes``
sizes correctly via ``numel * element_size``.

These tests pin two invariants:

1. Validator drop — ``ProTrainArgs.model_validate`` accepts both
   ``load_in_8bit: true`` and ``load_in_4bit: true`` when the ProTrain
   plugin is registered (the previous behavior raised
   ``ValidationError``; the new behavior must NOT).
2. ``_param_bytes`` correctness for synthetic int8/uint8 tensors that
   stand in for the storage layout bnb produces — the chunk layout's
   byte math must equal ``numel * element_size`` regardless of dtype.

Bnb itself is not imported here so the tests run in any env (the bnb
storage layout is reproduced with stock ``torch.uint8`` / ``torch.int8``
tensors of matching shapes).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from axolotl.integrations.protrain.args import ProTrainArgs
from axolotl.integrations.protrain.chunk.layout import _param_bytes
from axolotl.integrations.protrain.types import ParamId


def _minimal_active_cfg(**overrides) -> dict:
    cfg: dict = {
        "protrain_auto_memory": True,
        "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
        "base_model": "HuggingFaceTB/SmolLM2-135M",
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------
# Validator drop — load_in_8bit / load_in_4bit must be accepted when
# ProTrain is active. Mirrors the positive-control test in
# ``test_plugin_args_validators.py`` but kept here so the quant
# milestone owns its own regression surface.
# ---------------------------------------------------------------------


def test_load_in_8bit_passes_with_protrain_active() -> None:
    cfg = _minimal_active_cfg(load_in_8bit=True)
    # Must NOT raise.
    ProTrainArgs.model_validate(cfg)


def test_load_in_4bit_passes_with_protrain_active() -> None:
    cfg = _minimal_active_cfg(load_in_4bit=True)
    # Must NOT raise.
    ProTrainArgs.model_validate(cfg)


def test_load_in_4bit_passes_with_qlora_adapter() -> None:
    """QLoRA = ``load_in_4bit: true`` + ``adapter: qlora``; the canonical config."""
    cfg = _minimal_active_cfg(load_in_4bit=True, adapter="qlora")
    ProTrainArgs.model_validate(cfg)


# ---------------------------------------------------------------------
# Chunk layout — _param_bytes must size packed-byte storage correctly.
# Synthetic models stand in for bnb's Int8Params / Params4bit because:
#   * Int8Params post-.cuda() with has_fp16_weights=False is a
#     torch.int8 tensor of shape (out, in), element_size=1.
#   * Params4bit storage is a torch.uint8 tensor of shape
#     (ceil(in*out/2), 1), element_size=1.
# In both cases byte size = numel * 1 = packed bytes — the exact
# accounting the chunk packer relies on. Reproduce that shape with
# stock dtypes so the test runs without bnb installed.
# ---------------------------------------------------------------------


def test_param_bytes_int8_matches_packed_bytes() -> None:
    """Int8Params storage: numel == out*in, element_size == 1."""
    out, in_ = 32, 64
    model = nn.Module()
    # Bypass nn.Parameter's float-only constraint by registering a buffer-shaped
    # int8 storage as if it were a frozen weight (matches Int8Params stride).
    model.weight = nn.Parameter(
        torch.zeros(out, in_, dtype=torch.int8), requires_grad=False
    )
    sizes = _param_bytes(model)
    assert sizes[cast(ParamId, "weight")] == out * in_  # 1 byte per element


def test_param_bytes_uint8_matches_packed_bytes() -> None:
    """Params4bit storage: 2 weights packed per uint8 byte → numel == ceil(out*in/2)."""
    out, in_ = 32, 64
    packed = (out * in_ + 1) // 2  # 2-per-byte packing
    model = nn.Module()
    model.weight = nn.Parameter(
        torch.zeros(packed, 1, dtype=torch.uint8), requires_grad=False
    )
    sizes = _param_bytes(model)
    assert (
        sizes[cast(ParamId, "weight")] == packed
    )  # 1 byte per element, packed storage


def test_param_bytes_mixed_dtypes() -> None:
    """A frozen-int8 base + fp16 LoRA + fp32 norm scale — the realistic LoRA-on-8bit shape."""
    model = nn.Module()
    model.base_weight = nn.Parameter(
        torch.zeros(32, 64, dtype=torch.int8), requires_grad=False
    )
    model.lora_a = nn.Parameter(torch.zeros(16, 64, dtype=torch.float16))
    model.lora_b = nn.Parameter(torch.zeros(32, 16, dtype=torch.float16))
    model.norm = nn.Parameter(torch.zeros(64, dtype=torch.float32))
    sizes = _param_bytes(model)
    assert sizes[cast(ParamId, "base_weight")] == 32 * 64 * 1  # int8 packed
    assert sizes[cast(ParamId, "lora_a")] == 16 * 64 * 2  # fp16
    assert sizes[cast(ParamId, "lora_b")] == 32 * 16 * 2
    assert sizes[cast(ParamId, "norm")] == 64 * 4  # fp32
