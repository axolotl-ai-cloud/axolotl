"""Tests for HTTP weight sync serialization round-trip (bf16/fp16/fp32).

Exercises the encode/decode helpers in axolotl.utils.weight_serde that handle
the three-stage weight transfer: trainer → serve endpoint → vLLM worker.
"""

import pytest
import torch

from axolotl.utils.weight_serde import (
    decode_from_http,
    decode_from_ipc,
    encode_for_http,
    encode_for_ipc,
)

# ---------------------------------------------------------------------------
# Stage 1: trainer → serve endpoint (HTTP with base64)
# ---------------------------------------------------------------------------


class TestHttpEncodeRoundTrip:
    """Test encode_for_http / decode_from_http."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_round_trip_dtype(self, dtype):
        original = torch.randn(32, 64, dtype=dtype)
        entry = encode_for_http("layer.weight", original)
        name, decoded = decode_from_http(entry)

        assert name == "layer.weight"
        assert decoded.dtype == dtype
        assert decoded.shape == original.shape
        if dtype == torch.bfloat16:
            # bf16→fp16→bf16 loses some precision
            torch.testing.assert_close(decoded, original, atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(decoded, original, atol=0, rtol=0)

    def test_bfloat16_wire_format_is_fp16(self):
        """bf16 tensors should be sent as fp16 on the wire."""
        import base64

        original = torch.randn(8, 16, dtype=torch.bfloat16)
        entry = encode_for_http("w", original)
        raw = base64.b64decode(entry["data"])
        # 8*16 elements * 2 bytes/elem (fp16) = 256 bytes
        assert len(raw) == 8 * 16 * 2
        # dtype field should preserve original dtype for reconstruction
        assert entry["dtype"] == "torch.bfloat16"

    def test_multidimensional_shapes(self):
        for shape in [(128,), (4, 32), (2, 3, 16), (2, 2, 2, 8)]:
            original = torch.randn(*shape, dtype=torch.bfloat16)
            entry = encode_for_http("w", original)
            _, decoded = decode_from_http(entry)
            assert decoded.shape == original.shape
            assert decoded.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Stage 2: serve endpoint → vLLM worker (IPC with raw bytes)
# ---------------------------------------------------------------------------


class TestIpcEncodeRoundTrip:
    """Test encode_for_ipc / decode_from_ipc."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_round_trip_dtype(self, dtype):
        original = torch.randn(32, 64, dtype=dtype)
        entry = encode_for_ipc("layer.weight", original)
        name, decoded = decode_from_ipc(entry)

        assert name == "layer.weight"
        assert decoded.dtype == dtype
        assert decoded.shape == original.shape
        if dtype == torch.bfloat16:
            torch.testing.assert_close(decoded, original, atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(decoded, original, atol=0, rtol=0)

    def test_bfloat16_ipc_wire_is_fp16(self):
        """bf16 tensors should be serialized as fp16 bytes in IPC."""
        original = torch.randn(4, 8, dtype=torch.bfloat16)
        entry = encode_for_ipc("w", original)
        assert entry["dtype"] == "float16"
        assert entry["target_dtype"] == "bfloat16"
        assert len(entry["data"]) == 4 * 8 * 2  # fp16 bytes

    def test_fp32_has_no_target_dtype_mismatch(self):
        original = torch.randn(4, 8, dtype=torch.float32)
        entry = encode_for_ipc("w", original)
        assert entry["dtype"] == "float32"
        assert entry["target_dtype"] == "float32"

    def test_worker_handles_missing_target_dtype(self):
        """Backward compat: older serve code may not send target_dtype."""
        entry = {
            "name": "w",
            "data": torch.randn(4, 8, dtype=torch.float32).numpy().tobytes(),
            "dtype": "float32",
            "shape": [4, 8],
            # no target_dtype key
        }
        name, decoded = decode_from_ipc(entry)
        assert decoded.dtype == torch.float32
        assert decoded.shape == (4, 8)


# ---------------------------------------------------------------------------
# Full pipeline: trainer → serve → worker
# ---------------------------------------------------------------------------


class TestFullPipelineRoundTrip:
    """End-to-end: trainer → serve → worker."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_three_stage_round_trip(self, dtype):
        """Tensor survives trainer→serve→worker with correct dtype and values."""
        original = torch.randn(16, 32, dtype=dtype)

        # Stage 1: trainer encodes for HTTP
        http_entry = encode_for_http("model.layers.0.weight", original)

        # Stage 2: serve decodes HTTP, re-encodes for IPC
        name, at_serve = decode_from_http(http_entry)
        ipc_entry = encode_for_ipc(name, at_serve)

        # Stage 3: worker decodes IPC
        _, at_worker = decode_from_ipc(ipc_entry)

        assert at_worker.dtype == dtype
        assert at_worker.shape == original.shape
        if dtype == torch.bfloat16:
            # Two bf16→fp16→bf16 hops compound precision loss slightly
            torch.testing.assert_close(at_worker, original, atol=2e-2, rtol=2e-2)
        else:
            torch.testing.assert_close(at_worker, original, atol=0, rtol=0)

    def test_bfloat16_precision_loss_is_bounded(self):
        """bf16→fp16→bf16 round-trip error should be small."""
        original = torch.randn(256, 256, dtype=torch.bfloat16)
        http_entry = encode_for_http("w", original)
        _, at_serve = decode_from_http(http_entry)
        ipc_entry = encode_for_ipc("w", at_serve)
        _, at_worker = decode_from_ipc(ipc_entry)

        max_err = (at_worker.float() - original.float()).abs().max().item()
        # bf16 has ~8e-3 precision, fp16 has ~1e-3; round-trip error bounded
        assert max_err < 0.05, f"Max error {max_err} exceeds bound"

    def test_bfloat16_numpy_would_crash_without_fix(self):
        """Verify that calling .numpy() on bf16 raises, confirming the fix is needed."""
        t = torch.randn(4, 4, dtype=torch.bfloat16)
        with pytest.raises((RuntimeError, TypeError)):
            t.numpy()
