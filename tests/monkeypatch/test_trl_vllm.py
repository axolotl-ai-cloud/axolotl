"""Unit tests for TRL vLLM monkeypatches.

Tests:
- split_tensor_dict: scalar type preservation (int/float/bool)
- shuffle_sequence_dict: scalar type preservation
- extract_logprobs: NaN → 0.0 replacement
- VLLMClient.batch_update_named_params: method exists after patch
- VLLMGeneration: weight_sync_chunk_size attribute after patch
- Patch idempotency: applying patch twice doesn't break anything
"""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import torch


class TestSplitTensorDict(unittest.TestCase):
    """Tests for patched split_tensor_dict."""

    def setUp(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _patched_split_tensor_dict

        self.split = _patched_split_tensor_dict

    def test_scalar_int_preserved(self):
        d = {"a": torch.randn(4, 3), "count": 42}
        chunks = self.split(d, 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["count"], 42)
        self.assertEqual(chunks[1]["count"], 42)

    def test_scalar_float_preserved(self):
        d = {"a": torch.randn(6, 2), "lr": 1e-5}
        chunks = self.split(d, 3)
        for c in chunks:
            self.assertEqual(c["lr"], 1e-5)

    def test_scalar_bool_preserved(self):
        d = {"a": torch.randn(4, 2), "flag": True}
        chunks = self.split(d, 2)
        for c in chunks:
            self.assertTrue(c["flag"])

    def test_none_preserved(self):
        d = {"a": torch.randn(4, 2), "b": None}
        chunks = self.split(d, 2)
        for c in chunks:
            self.assertIsNone(c["b"])

    def test_tensor_split(self):
        t = torch.arange(8).reshape(4, 2)
        d = {"a": t, "n": 10}
        chunks = self.split(d, 2)
        self.assertEqual(chunks[0]["a"].shape, (2, 2))
        self.assertEqual(chunks[1]["a"].shape, (2, 2))
        torch.testing.assert_close(chunks[0]["a"], t[:2])
        torch.testing.assert_close(chunks[1]["a"], t[2:])

    def test_0d_tensor_preserved(self):
        d = {"a": torch.randn(4, 2), "scalar_t": torch.tensor(3.14)}
        chunks = self.split(d, 2)
        for c in chunks:
            self.assertAlmostEqual(c["scalar_t"].item(), 3.14, places=5)

    def test_list_split(self):
        d = {"a": torch.randn(4, 2), "names": ["a", "b", "c", "d"]}
        chunks = self.split(d, 2)
        self.assertEqual(chunks[0]["names"], ["a", "b"])
        self.assertEqual(chunks[1]["names"], ["c", "d"])


class TestShuffleSequenceDict(unittest.TestCase):
    """Tests for patched shuffle_sequence_dict."""

    def setUp(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _patched_shuffle_sequence_dict

        self.shuffle = _patched_shuffle_sequence_dict

    def test_scalar_int_preserved(self):
        d = {"a": torch.randn(4, 3), "count": 42}
        result = self.shuffle(d)
        self.assertEqual(result["count"], 42)

    def test_scalar_float_preserved(self):
        d = {"a": torch.randn(4, 3), "lr": 1e-5}
        result = self.shuffle(d)
        self.assertEqual(result["lr"], 1e-5)

    def test_scalar_bool_preserved(self):
        d = {"a": torch.randn(4, 3), "flag": False}
        result = self.shuffle(d)
        self.assertFalse(result["flag"])

    def test_none_preserved(self):
        d = {"a": torch.randn(4, 3), "b": None}
        result = self.shuffle(d)
        self.assertIsNone(result["b"])

    def test_tensor_permuted(self):
        torch.manual_seed(42)
        t = torch.arange(4).float()
        d = {"a": t}
        result = self.shuffle(d)
        # Same elements, possibly different order
        self.assertEqual(sorted(result["a"].tolist()), sorted(t.tolist()))
        self.assertEqual(result["a"].shape, t.shape)

    def test_list_permuted(self):
        torch.manual_seed(42)
        d = {"a": torch.randn(3, 2), "names": ["x", "y", "z"]}
        result = self.shuffle(d)
        self.assertEqual(sorted(result["names"]), ["x", "y", "z"])
        self.assertEqual(len(result["names"]), 3)

    def test_0d_tensor_preserved(self):
        d = {"a": torch.randn(4, 2), "scalar_t": torch.tensor(3.14)}
        result = self.shuffle(d)
        self.assertAlmostEqual(result["scalar_t"].item(), 3.14, places=5)


class TestExtractLogprobs(unittest.TestCase):
    """Tests for patched extract_logprobs (NaN → 0.0)."""

    def setUp(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _patched_extract_logprobs

        self.extract = _patched_extract_logprobs

    def _make_output(self, logprob_values):
        """Create a mock vLLM RequestOutput with given logprob values."""

        @dataclass
        class LogprobItem:
            logprob: float
            rank: int

        @dataclass
        class SeqOutput:
            logprobs: list[dict[int, LogprobItem]] | None

        @dataclass
        class RequestOutput:
            outputs: list[SeqOutput]

        logprobs_list = []
        for vals in logprob_values:
            lp_dict = {i: LogprobItem(logprob=v, rank=i) for i, v in enumerate(vals)}
            logprobs_list.append(lp_dict)

        return RequestOutput(outputs=[SeqOutput(logprobs=logprobs_list)])

    def test_nan_replaced_with_zero(self):
        output = self._make_output([[float("nan"), 0.5], [-0.3, float("nan")]])
        logprobs, token_ids = self.extract([output])
        self.assertEqual(logprobs[0][0][0], 0.0)  # NaN → 0.0
        self.assertEqual(logprobs[0][0][1], 0.5)
        self.assertEqual(logprobs[0][1][0], -0.3)
        self.assertEqual(logprobs[0][1][1], 0.0)  # NaN → 0.0

    def test_normal_values_preserved(self):
        output = self._make_output([[-0.5, -1.2], [-0.1, -2.0]])
        logprobs, token_ids = self.extract([output])
        self.assertAlmostEqual(logprobs[0][0][0], -0.5)
        self.assertAlmostEqual(logprobs[0][0][1], -1.2)

    def test_none_logprobs_returns_none(self):
        @dataclass
        class SeqOutput:
            logprobs: None = None

        @dataclass
        class RequestOutput:
            outputs: list

        output = RequestOutput(outputs=[SeqOutput()])
        logprobs, token_ids = self.extract([output])
        self.assertIsNone(logprobs)
        self.assertIsNone(token_ids)

    def test_token_ids_extracted(self):
        output = self._make_output([[-0.5]])
        logprobs, token_ids = self.extract([output])
        self.assertEqual(token_ids[0][0], [0])  # token_id=0 from enumerate


class TestPatchApplication(unittest.TestCase):
    """Tests for patch_trl_vllm() application."""

    def test_batch_update_added_to_client(self):
        from axolotl.monkeypatch.trainer.trl_vllm import patch_trl_vllm

        patch_trl_vllm()
        from trl.generation.vllm_client import VLLMClient

        self.assertTrue(hasattr(VLLMClient, "batch_update_named_params"))

    def test_extract_logprobs_patched(self):
        from axolotl.monkeypatch.trainer.trl_vllm import (
            _patched_extract_logprobs,
            patch_trl_vllm,
        )

        patch_trl_vllm()
        from trl.generation import vllm_generation

        self.assertIs(vllm_generation.extract_logprobs, _patched_extract_logprobs)

    def test_utils_patched(self):
        from axolotl.monkeypatch.trainer.trl_vllm import (
            _patched_shuffle_sequence_dict,
            _patched_split_tensor_dict,
            patch_trl_vllm,
        )

        patch_trl_vllm()
        import trl.trainer.utils

        self.assertIs(trl.trainer.utils.split_tensor_dict, _patched_split_tensor_dict)
        self.assertIs(
            trl.trainer.utils.shuffle_sequence_dict, _patched_shuffle_sequence_dict
        )

    def test_patch_idempotent(self):
        from axolotl.monkeypatch.trainer.trl_vllm import patch_trl_vllm

        patch_trl_vllm()
        patch_trl_vllm()  # second call should not error
        from trl.generation.vllm_client import VLLMClient

        self.assertTrue(hasattr(VLLMClient, "batch_update_named_params"))


class TestBatchUpdateChunking(unittest.TestCase):
    """Tests for batch_update_named_params chunking logic."""

    def test_no_chunk_single_batch(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        # Test that with chunk_size=None, all params go in one chunk
        client = MagicMock()
        client.base_url = "http://localhost:8000"
        client.session.post.return_value = MagicMock(status_code=200)
        client.communicator = MagicMock()
        client.communicator.group = MagicMock()
        client.rank = 0

        params = [
            ("layer.0.weight", torch.randn(10, 10)),
            ("layer.1.weight", torch.randn(10, 10)),
        ]
        _batch_update_named_params(client, params, chunk_size=None)

        # Should make exactly 1 HTTP call
        self.assertEqual(client.session.post.call_count, 1)

    def test_chunk_splits_params(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = MagicMock()
        client.base_url = "http://localhost:8000"
        client.session.post.return_value = MagicMock(status_code=200)
        client.communicator = MagicMock()
        client.communicator.group = MagicMock()
        client.rank = 0

        params = [
            ("a", torch.randn(100)),  # 100 elements
            ("b", torch.randn(100)),  # 100 elements
            ("c", torch.randn(100)),  # 100 elements
        ]
        _batch_update_named_params(client, params, chunk_size=150)

        # Should make 2 HTTP calls: [a,b] then [c] (100+100 > 150 triggers split)
        # Actually: a=100 < 150, a+b=200 > 150 → chunk [a], then b=100 < 150,
        # b+c=200 > 150 → chunk [b], then [c]. So 3 calls.
        # Wait: first a added (100 < 150), then b: 100+100=200 > 150, so chunk=[a],
        # new chunk starts with b (100 < 150), then c: 100+100=200 > 150, so chunk=[b],
        # final chunk=[c]. 3 HTTP calls.
        self.assertEqual(client.session.post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
