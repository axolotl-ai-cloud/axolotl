"""Tests for the Qwen3 / Qwen3-MoE fused-attention monkeypatch."""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

pytest.importorskip("transformers.models.qwen3")
pytest.importorskip("transformers.models.qwen3_moe")


def _clear_patched_flag(cls):
    try:
        delattr(cls, "_axolotl_fused_attn_patched")
    except AttributeError:
        pass


@pytest.fixture
def restore_qwen3_attention():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    saved = Qwen3Attention.forward
    saved_flag = getattr(Qwen3Attention, "_axolotl_fused_attn_patched", False)
    yield Qwen3Attention
    Qwen3Attention.forward = saved
    if saved_flag:
        Qwen3Attention._axolotl_fused_attn_patched = saved_flag
    else:
        _clear_patched_flag(Qwen3Attention)


@pytest.fixture
def restore_qwen3_moe_attention():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

    saved = Qwen3MoeAttention.forward
    saved_flag = getattr(Qwen3MoeAttention, "_axolotl_fused_attn_patched", False)
    yield Qwen3MoeAttention
    Qwen3MoeAttention.forward = saved
    if saved_flag:
        Qwen3MoeAttention._axolotl_fused_attn_patched = saved_flag
    else:
        _clear_patched_flag(Qwen3MoeAttention)


def _build_qwen3_model(seed: int = 0):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3Model(cfg).cuda().to(torch.bfloat16).eval()


def _build_qwen3_moe_model(seed: int = 0):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

    torch.manual_seed(seed)
    cfg = Qwen3MoeConfig(
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
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3MoeModel(cfg).cuda().to(torch.bfloat16).eval()


def _run_attention(model, layer_idx, hidden_states, position_ids):
    attn = model.layers[layer_idx].self_attn
    cos, sin = model.rotary_emb(hidden_states, position_ids)
    out, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )
    return out


class TestQwen3FusedAttnParity:
    """Single-layer parity vs stock."""

    @pytest.mark.parametrize("layer_idx", [0, 1])
    def test_forward_matches_stock(self, restore_qwen3_attention, layer_idx):
        from axolotl.monkeypatch.models.qwen3.fused_attn import patch_qwen3_fused_attn

        m = _build_qwen3_model(seed=1)
        hs = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(16, device="cuda").unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            ref = _run_attention(m, layer_idx, hs, pos)

        patch_qwen3_fused_attn()
        with torch.no_grad():
            got = _run_attention(m, layer_idx, hs, pos)

        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, (
            f"layer {layer_idx} fused vs stock cosine_sim={cos_sim:.6f}"
        )
        torch.testing.assert_close(got, ref, rtol=5e-2, atol=5e-2)


class TestQwen3FusedAttnEndToEnd:
    def test_full_forward_matches_stock(self, restore_qwen3_attention):
        from axolotl.monkeypatch.models.qwen3.fused_attn import patch_qwen3_fused_attn

        m = _build_qwen3_model(seed=2)
        ids = torch.randint(0, 128, (2, 32), device="cuda")
        mask = torch.ones(2, 32, dtype=torch.long, device="cuda")

        with torch.no_grad():
            ref = m(input_ids=ids, attention_mask=mask).last_hidden_state.clone()

        patch_qwen3_fused_attn()
        with torch.no_grad():
            got = m(input_ids=ids, attention_mask=mask).last_hidden_state.clone()

        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"end-to-end cosine_sim={cos_sim:.6f}"

    def test_backward_grad_flows_through_fused_path(self, restore_qwen3_attention):
        from axolotl.monkeypatch.models.qwen3.fused_attn import patch_qwen3_fused_attn

        m = _build_qwen3_model(seed=3).train()
        patch_qwen3_fused_attn()

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


class TestQwen3FusedAttnLoRACompose:
    """Pin that LoRA-QKV runs before the fused patch (``inspect.getsource`` regex misses on the fused body)."""

    def test_lora_qkv_then_fused_does_not_raise(self, restore_qwen3_attention):
        from axolotl.monkeypatch.lora_kernels import patch_self_attn_lora
        from axolotl.monkeypatch.models.qwen3.fused_attn import patch_qwen3_fused_attn
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen3-0.6B",
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
                "lora_dropout": 0.0,
            }
        )

        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        try:
            delattr(Qwen3Attention, "_original_forward")
        except AttributeError:
            pass

        try:
            patch_self_attn_lora(cfg)
            assert hasattr(Qwen3Attention, "_original_forward"), (
                "patch_self_attn_lora must run on stock source first"
            )
            patch_qwen3_fused_attn()
            assert getattr(Qwen3Attention, "_axolotl_fused_attn_patched", False)
        finally:
            try:
                delattr(Qwen3Attention, "_original_forward")
            except AttributeError:
                pass

    def test_reverse_order_breaks(self, restore_qwen3_attention):
        from axolotl.monkeypatch.lora_kernels import patch_self_attn_lora
        from axolotl.monkeypatch.models.qwen3.fused_attn import patch_qwen3_fused_attn
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault(
            {
                "base_model": "Qwen/Qwen3-0.6B",
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
                "lora_dropout": 0.0,
            }
        )

        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        try:
            delattr(Qwen3Attention, "_original_forward")
        except AttributeError:
            pass

        patch_qwen3_fused_attn()
        with pytest.raises(AssertionError, match="Original QKV code not found"):
            patch_self_attn_lora(cfg)


class TestPatchManagerOrdering:
    """Pin the patch-manager ordering invariant."""

    def test_self_attn_lora_runs_before_model_specific(self):
        import inspect

        from axolotl.loaders.patch_manager import PatchManager

        src = inspect.getsource(PatchManager.apply_pre_model_load_patches)
        lora_idx = src.find("_apply_self_attention_lora_patch()")
        specific_idx = src.find("_apply_model_specific_patches()")
        assert lora_idx > 0 and specific_idx > 0
        assert lora_idx < specific_idx, (
            "_apply_self_attention_lora_patch must run before "
            "_apply_model_specific_patches so patch_self_attn_lora sees the "
            "stock attention forward source"
        )


class TestQwen3MoeFusedAttnParity:
    """End-to-end parity on the MoE variant (attention is structurally identical to dense Qwen3)."""

    def test_full_forward_matches_stock(self, restore_qwen3_moe_attention):
        from axolotl.monkeypatch.models.qwen3_moe.fused_attn import (
            patch_qwen3_moe_fused_attn,
        )

        m = _build_qwen3_moe_model(seed=4)
        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")

        with torch.no_grad():
            ref = m(input_ids=ids, attention_mask=mask).last_hidden_state.clone()

        patch_qwen3_moe_fused_attn()
        with torch.no_grad():
            got = m(input_ids=ids, attention_mask=mask).last_hidden_state.clone()

        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"qwen3-moe end-to-end cosine_sim={cos_sim:.6f}"
