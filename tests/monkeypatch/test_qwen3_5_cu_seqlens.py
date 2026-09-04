"""Tests for cu_seqlens derivation in the Qwen3.5 packing monkeypatch."""

import torch

from axolotl.monkeypatch.models.qwen3_5.modeling import get_cu_seqlens


def _packed_position_ids():
    """Two rows of [0, 1, 2, 3, 0, 1, 2, 3], i.e. two packed samples per row."""
    return torch.stack([torch.cat([torch.arange(4), torch.arange(4)])] * 2)


def test_get_cu_seqlens_accepts_non_contiguous_position_ids():
    """Sample packing hands in a sliced view of position_ids.

    `view` requires a contiguous layout, so the sliced tensor that packing
    produces raised ``RuntimeError: view size is not compatible with input
    tensor's size and stride`` before any boundary was computed.
    """
    sliced = _packed_position_ids()[:, :6]
    assert not sliced.is_contiguous()

    cu_seqlens = get_cu_seqlens(sliced)

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 4, 6, 10, 12]


def test_get_cu_seqlens_accepts_non_contiguous_mrope_position_ids():
    """MRoPE hands the same sliced tensor in as [axes, B, T]."""
    mrope = torch.stack([_packed_position_ids()] * 3)[:, :, :6]
    assert not mrope.is_contiguous()

    assert get_cu_seqlens(mrope).tolist() == [0, 4, 6, 10, 12]


def test_get_cu_seqlens_accepts_expanded_mrope_position_ids():
    """The shape transformers actually builds, and the one that crashed.

    Qwen3_5TextModel expands a 1-D arange into the MRoPE axes, so axis 0 is a
    stride-0 view rather than a slice:
    https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1177-L1182
    """
    batch, seq_len = 2, 4
    expanded = torch.arange(seq_len).view(1, 1, -1).expand(4, batch, -1)[1:]
    assert not expanded[0].is_contiguous()

    assert get_cu_seqlens(expanded).tolist() == [0, 4, 8]


def test_get_cu_seqlens_keys_off_smallest_position():
    """Sequences need not start at 0; boundaries follow the smallest position."""
    packed = _packed_position_ids() + 1

    assert get_cu_seqlens(packed).tolist() == [0, 4, 8, 12, 16]


def test_get_cu_seqlens_unchanged_for_contiguous_position_ids():
    """The contiguous path keeps the boundaries it produced before."""
    packed = _packed_position_ids()
    assert packed.is_contiguous()

    assert get_cu_seqlens(packed).tolist() == [0, 4, 8, 12, 16]
    assert get_cu_seqlens(torch.stack([packed] * 3)).tolist() == [0, 4, 8, 12, 16]
