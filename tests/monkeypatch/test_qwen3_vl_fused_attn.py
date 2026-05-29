"""Tests for the Qwen3-VL text fused-attention monkeypatch."""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

pytest.importorskip("transformers.models.qwen3_vl")


def _clear_patched_flag(cls):
    try:
        delattr(cls, "_axolotl_fused_attn_patched")
    except AttributeError:
        pass


@pytest.fixture
def restore_qwen3_vl_attention():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    saved = Qwen3VLTextAttention.forward
    saved_flag = getattr(Qwen3VLTextAttention, "_axolotl_fused_attn_patched", False)
    yield Qwen3VLTextAttention
    Qwen3VLTextAttention.forward = saved
    if saved_flag:
        Qwen3VLTextAttention._axolotl_fused_attn_patched = saved_flag
    else:
        _clear_patched_flag(Qwen3VLTextAttention)


def _build_qwen3_vl_text_model(seed: int = 0):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(seed)
    cfg = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3VLTextModel(cfg).cuda().to(torch.bfloat16).eval()


def _run_attention(model, layer_idx, hidden_states, position_ids):
    attn = model.layers[layer_idx].self_attn
    cos, sin = model.rotary_emb(hidden_states, position_ids)
    out, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )
    return out


class TestQwen3VLFusedAttnParity:
    """Single-layer parity vs stock Qwen3VLTextAttention."""

    @pytest.mark.parametrize("layer_idx", [0, 1])
    def test_forward_matches_stock(self, restore_qwen3_vl_attention, layer_idx):
        from axolotl.monkeypatch.models.qwen3_vl.fused_attn import (
            patch_qwen3_vl_fused_attn,
        )

        model = _build_qwen3_vl_text_model(seed=1)
        hidden_states = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        position_ids = torch.arange(16, device="cuda").unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            ref = _run_attention(model, layer_idx, hidden_states, position_ids)

        patch_qwen3_vl_fused_attn()
        with torch.no_grad():
            got = _run_attention(model, layer_idx, hidden_states, position_ids)

        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, (
            f"layer {layer_idx} fused vs stock cosine_sim={cos_sim:.6f}"
        )
        torch.testing.assert_close(got, ref, rtol=5e-2, atol=5e-2)


class TestQwen3VLFusedAttnEndToEnd:
    def test_full_forward_matches_stock(self, restore_qwen3_vl_attention):
        from axolotl.monkeypatch.models.qwen3_vl.fused_attn import (
            patch_qwen3_vl_fused_attn,
        )

        model = _build_qwen3_vl_text_model(seed=2)
        input_ids = torch.randint(0, 128, (2, 32), device="cuda")
        attention_mask = torch.ones(2, 32, dtype=torch.long, device="cuda")

        with torch.no_grad():
            ref = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state.clone()

        patch_qwen3_vl_fused_attn()
        with torch.no_grad():
            got = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state.clone()

        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"end-to-end cosine_sim={cos_sim:.6f}"

    def test_backward_grad_flows_through_fused_path(self, restore_qwen3_vl_attention):
        from axolotl.monkeypatch.models.qwen3_vl.fused_attn import (
            patch_qwen3_vl_fused_attn,
        )

        model = _build_qwen3_vl_text_model(seed=3).train()
        patch_qwen3_vl_fused_attn()

        input_ids = torch.randint(0, 128, (2, 16), device="cuda")
        attention_mask = torch.ones(2, 16, dtype=torch.long, device="cuda")
        out = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        out.sum().backward()

        for idx, layer in enumerate(model.layers[:2]):
            attn = layer.self_attn
            assert attn.q_norm.weight.grad is not None, f"layer {idx} q_norm no grad"
            assert attn.k_norm.weight.grad is not None, f"layer {idx} k_norm no grad"
            assert attn.q_norm.weight.grad.isfinite().all()
            assert attn.k_norm.weight.grad.isfinite().all()
            assert attn.q_norm.weight.grad.abs().sum() > 0
            assert attn.k_norm.weight.grad.abs().sum() > 0


class TestQwen3VLPatchManagerDispatch:
    def test_patch_manager_dispatches_qwen3_vl(self, restore_qwen3_vl_attention):
        from types import SimpleNamespace

        from axolotl.loaders.patch_manager import PatchManager

        cfg = SimpleNamespace(
            fused_attn_kernel=True,
            model_config_type="qwen3_vl",
            llama4_linearized_experts=False,
            sample_packing=False,
            context_parallel_size=1,
            attn_uses_flash_lib=False,
        )

        PatchManager(cfg=cfg, model_config=object())._apply_model_specific_patches()

        assert getattr(restore_qwen3_vl_attention, "_axolotl_fused_attn_patched", False)
