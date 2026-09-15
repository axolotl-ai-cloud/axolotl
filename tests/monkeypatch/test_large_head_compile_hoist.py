"""Tests for compile-friendly multidoc detection and doc_end vectorization."""

import pytest
import torch

# Import the production helper directly so the parity tests exercise the real
# vectorized path instead of a duplicate that could silently drift.
from axolotl.monkeypatch.attention.flash_attn_d512 import (
    _compute_doc_end as _doc_end_vectorized,
)
from axolotl.monkeypatch.attention.large_head import (
    _multidoc_position_ids,
    set_large_head_packed,
)


@pytest.fixture(autouse=True)
def _reset_packed():
    yield
    set_large_head_packed(None)


class TestStaticPackedFlag:
    def _packed_pos(self):
        return torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]])

    def _single_doc_pos(self):
        return torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])

    def test_runtime_probe_when_undeclared(self):
        set_large_head_packed(None)
        assert _multidoc_position_ids(self._packed_pos()) is not None
        assert _multidoc_position_ids(self._single_doc_pos()) is None

    def test_static_packed_true_skips_probe(self):
        set_large_head_packed(True)
        # even a single-doc row routes varlen: no data-dependent branch
        assert _multidoc_position_ids(self._single_doc_pos()) is not None
        assert _multidoc_position_ids(self._packed_pos()) is not None

    def test_static_packed_false(self):
        set_large_head_packed(False)
        assert _multidoc_position_ids(self._packed_pos()) is None

    def test_none_position_ids(self):
        set_large_head_packed(True)
        assert _multidoc_position_ids(None) is None


def _doc_end_loop_reference(pos: torch.Tensor) -> torch.Tensor:
    B, N = pos.shape
    doc_end = torch.empty_like(pos)
    for b in range(B):
        starts = (pos[b] == 0).nonzero().flatten()
        bounds = torch.cat(
            [starts, torch.tensor([N], device=pos.device, dtype=starts.dtype)]
        )
        doc_end[b] = (
            bounds[1:].repeat_interleave(bounds[1:] - bounds[:-1]).to(pos.dtype)
        )
    return doc_end


class TestDocEndVectorization:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_parity_random_packing(self, seed):
        gen = torch.Generator().manual_seed(seed)
        B, N = 3, 64
        rows = []
        for _ in range(B):
            lengths = []
            remaining = N
            while remaining > 0:
                length = int(torch.randint(1, 17, (1,), generator=gen))
                length = min(length, remaining)
                lengths.append(length)
                remaining -= length
            rows.append(torch.cat([torch.arange(n) for n in lengths]))
        pos = torch.stack(rows).to(torch.int32)
        torch.testing.assert_close(
            _doc_end_vectorized(pos), _doc_end_loop_reference(pos)
        )

    def test_parity_single_doc(self):
        pos = torch.arange(32, dtype=torch.int32)[None]
        torch.testing.assert_close(
            _doc_end_vectorized(pos), _doc_end_loop_reference(pos)
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_flash_d512_varlen_uses_vectorized_doc_end(self):
        # end-to-end: packed flash_d512 still matches per-document SDPA
        from axolotl.monkeypatch.attention.flash_attn_d512 import flash_d512

        torch.manual_seed(0)
        B, H, D = 1, 2, 512
        lengths = [24, 40]
        N = sum(lengths)
        pos = torch.cat([torch.arange(n) for n in lengths])[None].cuda().to(torch.int32)
        q, k, v = (
            torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        out = flash_d512(q, k, v, causal=True, position_ids=pos)

        ref = torch.empty_like(out)
        start = 0
        for n in lengths:
            sl = slice(start, start + n)
            ref[:, :, sl] = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, sl].float(),
                k[:, :, sl].float(),
                v[:, :, sl].float(),
                is_causal=True,
            ).to(out.dtype)
            start += n
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
