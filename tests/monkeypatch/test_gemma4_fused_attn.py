"""Tests for the Gemma 4 fused-attention monkey-patch (hybrid sliding + partial-rotary layers)."""

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
    """Snapshot ``Gemma4TextAttention.forward`` so the patch can't leak across tests."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    saved = Gemma4TextAttention.forward
    yield Gemma4TextAttention
    Gemma4TextAttention.forward = saved


def _build_hybrid_config():
    """Tiny hybrid Gemma 4: one sliding + one full-attention layer with ``partial_rotary_factor=0.25``."""
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


class TestGemma4FusedAttnLoRACompose:
    """LoRA QKV + fused composition. Gemma 4 in transformers>=5.8.1 has no matching ``QKV_PATCHES`` entry yet, hence the strict xfail below."""

    def _build_cfg(self):
        from axolotl.utils.dict import DictDefault

        return DictDefault(
            {
                "base_model": "fake/gemma4",
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
                "lora_dropout": 0.0,
            }
        )

    @pytest.mark.xfail(
        reason="Gemma 4 QKV_PATCHES need refresh for transformers 5.8.1",
        strict=True,
    )
    def test_lora_qkv_then_fused_does_not_raise(
        self, restore_gemma4_attention, monkeypatch
    ):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        from axolotl.monkeypatch import lora_kernels
        from axolotl.monkeypatch.models.gemma4.fused_attn import (
            patch_gemma4_fused_attn,
        )

        monkeypatch.setattr(
            lora_kernels,
            "get_attention_cls_from_config",
            lambda _cfg: Gemma4TextAttention,
        )

        try:
            delattr(Gemma4TextAttention, "_original_forward")
        except AttributeError:
            pass

        try:
            lora_kernels.patch_self_attn_lora(self._build_cfg())
            assert hasattr(Gemma4TextAttention, "_original_forward"), (
                "patch_self_attn_lora must run on stock source first"
            )
            patch_gemma4_fused_attn()
        finally:
            try:
                delattr(Gemma4TextAttention, "_original_forward")
            except AttributeError:
                pass

    def test_reverse_order_skips_lora_rewrite(
        self, restore_gemma4_attention, monkeypatch, caplog
    ):
        """Fused-then-LoRA must NOT install the LoRA source rewrite. Upstream
        PR #3657 made ``patch_self_attn_lora`` detect ``apply_qkv``/``apply_o``
        on a fused-patched attention and skip; our ``patch_manager`` reorder
        keeps this from happening in practice, but the skip path is the last
        line of defense."""
        import logging

        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        from axolotl.monkeypatch import lora_kernels
        from axolotl.monkeypatch.models.gemma4.fused_attn import (
            patch_gemma4_fused_attn,
        )

        monkeypatch.setattr(
            lora_kernels,
            "get_attention_cls_from_config",
            lambda _cfg: Gemma4TextAttention,
        )

        try:
            delattr(Gemma4TextAttention, "_original_forward")
        except AttributeError:
            pass

        try:
            patch_gemma4_fused_attn()
            logger = logging.getLogger("axolotl.monkeypatch.lora_kernels")
            logger.addHandler(caplog.handler)
            previous_level = logger.level
            logger.setLevel(logging.INFO)
            try:
                lora_kernels.patch_self_attn_lora(self._build_cfg())
            finally:
                logger.removeHandler(caplog.handler)
                logger.setLevel(previous_level)
            assert "fused attention" in caplog.text and "skipping" in caplog.text, (
                "expected lora_kernels to detect the fused path and log a skip; "
                f"got {caplog.text}"
            )
            assert not hasattr(Gemma4TextAttention, "_original_forward"), (
                "lora_kernels installed _original_forward over a fused-patched class"
            )
        finally:
            try:
                delattr(Gemma4TextAttention, "_original_forward")
            except AttributeError:
                pass


def _build_kv_shared_config():
    """Hybrid Gemma 4 with ``num_kv_shared_layers > 0`` so the fused shared-KV branch runs."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=64,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=64,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )
    cfg._attn_implementation = "sdpa"
    return cfg


class TestFusedAttnSharedKV:
    """Regression: ``num_kv_shared_layers > 0`` hit the ``kv_shared_layer_index`` key removed in transformers>=5.8."""

    def test_shared_kv_forward_backward(self, restore_gemma4_attention):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

        from axolotl.monkeypatch.models.gemma4.fused_attn import (
            patch_gemma4_fused_attn,
        )

        torch.manual_seed(4)
        m = Gemma4TextModel(_build_kv_shared_config()).cuda().to(torch.bfloat16).train()
        assert any(layer.self_attn.is_kv_shared_layer for layer in m.layers), (
            "test config must exercise at least one kv-shared layer"
        )

        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")

        with torch.no_grad():
            ref = m(input_ids=ids, attention_mask=mask).last_hidden_state.clone()

        patch_gemma4_fused_attn()
        out = m(input_ids=ids, attention_mask=mask).last_hidden_state
        out.sum().backward()

        assert out.shape == ref.shape
        assert torch.isfinite(out).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), out.detach().flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"shared-kv fused vs stock cosine_sim={cos_sim:.6f}"


class TestFusedAttnSignature:
    """Pin the fused forward against the live ``Gemma4TextDecoderLayer.forward`` call shape."""

    def test_decoder_layer_can_call_fused_forward(self, restore_gemma4_attention):
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
    """Single-layer comparison of patched vs stock attention to isolate kernel correctness from cross-layer drift."""

    def _run_attention(self, model, layer_idx, hidden_states, position_ids):
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
        torch.testing.assert_close(got, ref, rtol=5e-2, atol=5e-2)


class TestFusedAttnFullModel:
    """End-to-end forward + backward through both layer types."""

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
        assert cos_sim > 0.999, f"end-to-end cosine_sim={cos_sim:.6f}"

    def test_backward_grad_flows_through_fused_path(self, restore_gemma4_attention):
        from axolotl.monkeypatch.models.gemma4.fused_attn import (
            patch_gemma4_fused_attn,
        )

        m = _build_model(seed=3).train()
        patch_gemma4_fused_attn()

        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")
        out = m(input_ids=ids, attention_mask=mask).last_hidden_state
        out.sum().backward()

        for i, layer in enumerate(m.layers[:2]):
            attn = layer.self_attn
            assert attn.q_norm.weight.grad is not None, f"layer {i} q_norm no grad"
            assert attn.k_norm.weight.grad is not None, f"layer {i} k_norm no grad"
            assert attn.q_norm.weight.grad.isfinite().all()
            assert attn.k_norm.weight.grad.isfinite().all()
            assert attn.q_norm.weight.grad.abs().sum() > 0
            assert attn.k_norm.weight.grad.abs().sum() > 0
