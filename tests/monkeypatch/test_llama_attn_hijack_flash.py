"""
Unit tests for the monkeypatch utils
"""
import unittest

import torch

from axolotl.monkeypatch.utils import get_cu_seqlens, get_cu_seqlens_from_pos_ids


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


if __name__ == "__main__":
    unittest.main()
