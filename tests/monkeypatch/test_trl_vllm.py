"""Unit tests for TRL vLLM monkeypatches.

Tests:
- split_tensor_dict: scalar type preservation (int/float/bool)
- shuffle_sequence_dict: scalar type preservation
- extract_logprobs: NaN → 0.0 replacement
- VLLMClient.batch_update_named_params: method exists after patch, native NCCL streaming
- Removed patches (update_model_params, VLLMGeneration init/sync_weights) stay absent
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

    def test_native_sync_paths_left_alone(self):
        from axolotl.monkeypatch.trainer import trl_vllm
        from axolotl.monkeypatch.trainer.trl_vllm import patch_trl_vllm

        patch_trl_vllm()
        from trl.generation.vllm_client import VLLMClient
        from trl.generation.vllm_generation import VLLMGeneration

        self.assertFalse(hasattr(trl_vllm, "_make_batched_sync_weights"))
        self.assertFalse(hasattr(trl_vllm, "_patch_sync_weights_batched"))
        self.assertFalse(hasattr(trl_vllm, "_update_model_params"))
        self.assertEqual(
            VLLMGeneration.sync_weights.__module__, VLLMGeneration.__module__
        )
        self.assertEqual(
            VLLMClient.update_model_params.__module__, VLLMClient.__module__
        )

    def test_patch_idempotent(self):
        from axolotl.monkeypatch.trainer.trl_vllm import patch_trl_vllm

        patch_trl_vllm()
        patch_trl_vllm()  # second call should not error
        from trl.generation.vllm_client import VLLMClient

        self.assertTrue(hasattr(VLLMClient, "batch_update_named_params"))


class TestBatchUpdateChunking(unittest.TestCase):
    """Tests for batch_update_named_params on the native client."""

    @staticmethod
    def _client(communicator="present"):
        client = MagicMock()
        client.communicator = communicator
        return client

    def test_no_chunk_single_call(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client()
        params = [
            ("layer.0.weight", torch.randn(10, 10)),
            ("layer.1.weight", torch.randn(10, 10)),
        ]
        _batch_update_named_params(client, params, chunk_size=None)

        self.assertEqual(client.update_named_params.call_count, 1)
        client.weight_update.assert_called_once()
        metadata = client.update_named_params.call_args[0][0]
        self.assertEqual([m[0] for m in metadata], ["layer.0.weight", "layer.1.weight"])

    def test_generation_paused_around_update(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client()
        client.base_url = "http://h:1"
        _batch_update_named_params(client, [("a", torch.randn(4))])

        urls = [c[0][0] for c in client._post.call_args_list]
        self.assertEqual(urls, ["http://h:1/pause", "http://h:1/resume"])
        self.assertEqual(client._post.call_args_list[0][1]["params"], {"mode": "keep"})

    def test_resumes_when_update_fails(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client()
        client.base_url = "http://h:1"
        client.update_named_params.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            _batch_update_named_params(client, [("a", torch.randn(4))])
        self.assertTrue(client._post.call_args[0][0].endswith("/resume"))

    def test_chunk_splits_params(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client()
        params = [(n, torch.randn(100)) for n in "abc"]
        _batch_update_named_params(client, params, chunk_size=150)

        self.assertEqual(client.update_named_params.call_count, 3)
        client.weight_update.assert_called_once()
        names = [
            [m[0] for m in call[0][0]]
            for call in client.update_named_params.call_args_list
        ]
        self.assertEqual(names, [["a"], ["b"], ["c"]])

    def test_chunk_groups_small_params(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client()
        params = [(n, torch.randn(50)) for n in "abc"]
        _batch_update_named_params(client, params, chunk_size=100)

        names = [
            [m[0] for m in call[0][0]]
            for call in client.update_named_params.call_args_list
        ]
        self.assertEqual(names, [["a", "b"], ["c"]])

    def test_lazy_init_when_no_communicator(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client(communicator=None)
        w = torch.randn(4)
        _batch_update_named_params(client, [("a", w)])

        client.init_communicator.assert_called_once_with(device=w.device)
        self.assertEqual(client.update_named_params.call_count, 1)

    def test_no_init_when_communicator_present(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client()
        _batch_update_named_params(client, [("a", torch.randn(4))])

        client.init_communicator.assert_not_called()

    def test_metadata_dtype_has_no_torch_prefix(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client()
        params = [
            ("a", torch.zeros(2, 3, dtype=torch.bfloat16)),
            ("b", torch.zeros(5, dtype=torch.float32)),
        ]
        _batch_update_named_params(client, params)

        metadata = client.update_named_params.call_args[0][0]
        self.assertEqual(metadata, [("a", "bfloat16", [2, 3]), ("b", "float32", [5])])

    def test_streamed_iterator_yields_chunk_tensors(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client()
        params = [("a", torch.randn(3)), ("b", torch.randn(3))]
        _batch_update_named_params(client, params)

        streamed = list(client.update_named_params.call_args[0][1])
        self.assertEqual([n for n, _ in streamed], ["a", "b"])

    def test_empty_params_is_noop(self):
        from axolotl.monkeypatch.trainer.trl_vllm import _batch_update_named_params

        client = self._client(communicator=None)
        _batch_update_named_params(client, [])

        client.init_communicator.assert_not_called()
        client.update_named_params.assert_not_called()


if __name__ == "__main__":
    unittest.main()
