"""
Unit tests for the monkeypatch utils
"""

import unittest

import torch

from axolotl.monkeypatch.utils import (
    get_cu_seqlens,
    get_cu_seqlens_from_pos_ids,
    get_max_seqlen_in_batch,
    get_unpad_data,
)


class TestMonkeyPatchUtils(unittest.TestCase):
    """
    Unit test class for monkeypatch utils
    """

    def test_get_cu_seqlens_1d(self):
        attn_mask = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 0, 0]])
        target_res = torch.tensor([0, 4, 7, 12, 14, 16], dtype=torch.int32)
        self.assertTrue(torch.allclose(get_cu_seqlens(attn_mask)[0], target_res))

    def test_get_cu_seqlens_from_pos_ids_1d(self):
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 0, 0]])
        target_res = torch.tensor([0, 4, 7, 12, 14, 16], dtype=torch.int32)
        self.assertTrue(
            torch.allclose(get_cu_seqlens_from_pos_ids(position_ids)[0], target_res)
        )

    def test_get_cu_seqlens_from_pos_ids_2d(self):
        position_ids = torch.tensor(
            [
                [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 0, 0],
                [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 0],
            ]
        )
        target_res = torch.tensor(
            [[0, 4, 7, 12, 14, 16], [0, 5, 8, 15, 16, 16]], dtype=torch.int32
        )
        self.assertTrue(
            torch.allclose(get_cu_seqlens_from_pos_ids(position_ids)[0], target_res)
        )

    def test_get_max_seqlen_in_batch(self):
        attn_mask = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 0, 0]])
        target_res = torch.tensor([4, 3, 5, 2], dtype=torch.int32)
        self.assertTrue(torch.allclose(get_max_seqlen_in_batch(attn_mask), target_res))

    def test_get_unpad_data(self):
        attn_mask = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 0, 0]])
        target_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        target_cu_seqlen = torch.tensor([0, 4, 7, 12, 14], dtype=torch.int32)
        target_max_seqlen_in_batch = 5
        indices, cu_seqlen, max_seqlen_in_batch = get_unpad_data(attn_mask)
        self.assertTrue(torch.allclose(target_indices, indices))
        self.assertTrue(torch.allclose(target_cu_seqlen, cu_seqlen))
        self.assertEqual(target_max_seqlen_in_batch, max_seqlen_in_batch)

        attn_mask = torch.tensor(
            [
                [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 0, 0],
                [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
            ]
        )
        target_indices = torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
            ]
        )
        target_cu_seqlen = torch.tensor(
            [0, 4, 7, 12, 14, 17, 22, 24, 27, 30], dtype=torch.int32
        )
        target_max_seqlen_in_batch = 5
        indices, cu_seqlen, max_seqlen_in_batch = get_unpad_data(attn_mask)
        self.assertTrue(torch.allclose(target_indices, indices))
        self.assertTrue(torch.allclose(target_cu_seqlen, cu_seqlen))
        self.assertEqual(target_max_seqlen_in_batch, max_seqlen_in_batch)


if __name__ == "__main__":
    unittest.main()
