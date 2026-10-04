"""Packed full-sequence attention visibility for native Nemotron diffusion."""

import pytest
import torch

from axolotl.integrations.diffusion.lm.attention import full_sequence_flex_block_mask


def test_full_sequence_flex_mask_is_bidirectional_within_valid_documents():
    document_ids = torch.tensor([[0, 0, 0, 1, 1] + [-1] * 123])
    validity = document_ids >= 0
    validity[0, 1] = False
    positions = torch.tensor([list(range(3)) + list(range(2)) + [0] * 123])

    block = full_sequence_flex_block_mask(document_ids, validity, positions)
    query = torch.arange(6)[:, None].expand(6, 6)
    key = torch.arange(6)[None, :].expand(6, 6)
    actual = block.mask_mod(
        torch.zeros_like(query), torch.zeros_like(query), query, key
    )
    expected = torch.tensor(
        [
            [True, False, True, False, False, False],
            [False] * 6,
            [True, False, True, False, False, False],
            [False, False, False, True, True, False],
            [False, False, False, True, True, False],
            [False] * 6,
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_full_sequence_flex_mask_rejects_sliding_attention():
    document_ids = torch.zeros((1, 128), dtype=torch.long)
    validity = torch.ones_like(document_ids, dtype=torch.bool)
    positions = torch.arange(128).unsqueeze(0)

    with pytest.raises(ValueError, match="does not support sliding attention"):
        full_sequence_flex_block_mask(
            document_ids, validity, positions, sliding_window=8
        )
