"""Tests for the Gemma 4 fused-attention monkey-patch.

These tests exercise the patched ``Gemma4TextAttention.forward`` against
the stock implementation it replaces. The hybrid Gemma 4 model intentionally
mixes a sliding (`head_dim=32`) layer with a full-attention proportional-rope
layer (`global_head_dim=64`, `partial_rotary_factor=0.25`) so that the
partial-rotary RMSNorm+RoPE path through the fused Triton kernel is
exercised end-to-end (this is the bug originally documented in
``GEMMA4_FUSED_ROPE_HYBRID_ATTN_BUG.md``).

The full-model forward also pins that the fused forward keeps accepting
whatever call shape ``Gemma4TextDecoderLayer.forward`` produces in the
installed transformers version — so any future signature drift on
upstream's side trips a clear failure here instead of a confusing
TypeError deep in a training run.
"""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

pytest.importorskip(
    "transformers.models.gemma4",
    reason="fused_attn patch only matters when Gemma 4 is available",
)


@pytest.fixture
def restore_gemma4_attention():
    """Snapshot ``Gemma4TextAttention.forward`` and restore after the test
    so the monkey-patch does not leak across the suite."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    saved = Gemma4TextAttention.forward
    yield Gemma4TextAttention
    Gemma4TextAttention.forward = saved


def _build_hybrid_config():
    """Tiny hybrid Gemma 4 config: one sliding layer + one full-attention
    layer with proportional rope and partial_rotary_factor=0.25. This is
    the same shape pattern as ``google/gemma-4-26B-A4B-it`` but small
    enough to fit on any GPU."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=64,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def _build_model(seed=0):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    torch.manual_seed(seed)
    cfg = _build_hybrid_config()
    return Gemma4TextModel(cfg).cuda().to(torch.bfloat16).eval()


class TestFusedAttnSignature:
    """The fused forward must accept the same call shape as
    ``Gemma4TextDecoderLayer`` produces in the installed transformers
    version. Any signature drift surfaces here as a TypeError."""

    def test_decoder_layer_can_call_fused_forward(self, restore_gemma4_attention):
        """Run a model forward that exercises the real
        ``Gemma4TextDecoderLayer -> Gemma4TextAttention`` call path with
        the fused patch installed."""
        from axolotl.monkeypatch.models.gemma4.fused_attn import (
            patch_gemma4_fused_attn,
        )

        m = _build_model()
        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")

        patch_gemma4_fused_attn()
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask).last_hidden_state

        assert out.shape == (2, 16, 64)
        assert torch.isfinite(out).all()


class TestFusedAttnPerLayerCorrectness:
    """Compare the patched attention layer to the stock implementation
    on a single forward call. This isolates the fused kernel correctness
    from cross-layer numerical drift."""

    def _run_attention(self, model, layer_idx, hidden_states, position_ids):
        """Call ``Gemma4TextAttention.forward`` (whatever is currently
        installed) for one layer and return the output."""
        attn = model.layers[layer_idx].self_attn
        layer_type = model.config.layer_types[layer_idx]
        cos, sin = model.rotary_emb(hidden_states, position_ids, layer_type)
        out, _ = attn(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            shared_kv_states={},
        )
        return out

    @pytest.mark.parametrize(
        "layer_idx",
        [0, 1],
        ids=["sliding_head32", "global_head64_proportional"],
    )
    def test_forward_matches_stock(self, restore_gemma4_attention, layer_idx):
        from axolotl.monkeypatch.models.gemma4.fused_attn import (
            patch_gemma4_fused_attn,
        )

        m = _build_model(seed=1)
        hs = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(16, device="cuda").unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            ref = self._run_attention(m, layer_idx, hs, pos)

        patch_gemma4_fused_attn()
        with torch.no_grad():
            got = self._run_attention(m, layer_idx, hs, pos)

        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, (
            f"layer {layer_idx} fused vs stock cosine_sim={cos_sim:.6f}"
        )
        # bf16 precision: a few millis of absolute drift per element is
        # acceptable for a Q/K/V projection pipeline. Anything larger is
        # a real bug.
        torch.testing.assert_close(got, ref, rtol=5e-2, atol=5e-2)


class TestFusedAttnFullModel:
    """End-to-end model forward + backward through both layer types."""

    def test_full_forward_matches_stock(self, restore_gemma4_attention):
        from axolotl.monkeypatch.models.gemma4.fused_attn import (
            patch_gemma4_fused_attn,
        )

        m = _build_model(seed=2)
        ids = torch.randint(0, 128, (2, 32), device="cuda")
        mask = torch.ones(2, 32, dtype=torch.long, device="cuda")

        with torch.no_grad():
            ref = m(input_ids=ids, attention_mask=mask).last_hidden_state.clone()

        patch_gemma4_fused_attn()
        with torch.no_grad():
            got = m(input_ids=ids, attention_mask=mask).last_hidden_state.clone()

        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        )
        # End-to-end through 2 layers (RMSNorm, attention, MLP/MoE) in bf16
        # accumulates a small amount of numerical drift; we just want to
        # pin that the two paths are computing the same function.
        assert cos_sim > 0.999, f"end-to-end cosine_sim={cos_sim:.6f}"

    def test_backward_grad_flows_through_fused_path(self, restore_gemma4_attention):
        """Gradients must propagate through the fused RMSNorm+RoPE kernels
        for both the sliding and proportional-rope layers."""
        from axolotl.monkeypatch.models.gemma4.fused_attn import (
            patch_gemma4_fused_attn,
        )

        m = _build_model(seed=3).train()
        patch_gemma4_fused_attn()

        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")
        out = m(input_ids=ids, attention_mask=mask).last_hidden_state
        out.sum().backward()

        # Both layers must accumulate gradients on q_norm.weight and
        # k_norm.weight — that proves the fused kernel ran the backward.
        for i, layer in enumerate(m.layers[:2]):
            attn = layer.self_attn
            assert attn.q_norm.weight.grad is not None, f"layer {i} q_norm no grad"
            assert attn.k_norm.weight.grad is not None, f"layer {i} k_norm no grad"
            assert attn.q_norm.weight.grad.isfinite().all()
            assert attn.k_norm.weight.grad.isfinite().all()
            assert attn.q_norm.weight.grad.abs().sum() > 0
            assert attn.k_norm.weight.grad.abs().sum() > 0
