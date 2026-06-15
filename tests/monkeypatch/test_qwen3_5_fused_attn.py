"""Tests for the Qwen3.5 / Qwen3.5-MoE fused-attention monkeypatch."""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("transformers.models.qwen3_5_moe")


def _clear_patched_flag(cls):
    try:
        delattr(cls, "_axolotl_fused_attn_patched")
    except AttributeError:
        pass


@pytest.fixture
def restore_qwen3_5_attention():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    saved = Qwen3_5Attention.forward
    saved_flag = getattr(Qwen3_5Attention, "_axolotl_fused_attn_patched", False)
    yield Qwen3_5Attention
    Qwen3_5Attention.forward = saved
    if saved_flag:
        Qwen3_5Attention._axolotl_fused_attn_patched = saved_flag
    else:
        _clear_patched_flag(Qwen3_5Attention)


def _build_qwen3_5_text_model(seed: int = 0):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        layer_types=["full_attention", "full_attention"],
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3_5TextModel(cfg).cuda().to(torch.bfloat16).eval()


def _run_attention(model, layer_idx, hidden_states, position_ids):
    attn = model.layers[layer_idx].self_attn
    cos, sin = model.rotary_emb(hidden_states, position_ids)
    out, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )
    return out


class TestQwen3_5FusedAttnParity:
    """Single-layer parity vs stock."""

    @pytest.mark.parametrize("layer_idx", [0, 1])
    def test_forward_matches_stock(self, restore_qwen3_5_attention, layer_idx):
        from axolotl.monkeypatch.models.qwen3_5.fused_attn import (
            patch_qwen3_5_fused_attn,
        )

        m = _build_qwen3_5_text_model(seed=1)
        hs = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
        pos = torch.arange(16, device="cuda").unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            ref = _run_attention(m, layer_idx, hs, pos)

        patch_qwen3_5_fused_attn()
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


class TestQwen3_5FusedAttnBackward:
    def test_q_k_norm_grads_finite_nonzero(self, restore_qwen3_5_attention):
        from axolotl.monkeypatch.models.qwen3_5.fused_attn import (
            patch_qwen3_5_fused_attn,
        )

        m = _build_qwen3_5_text_model(seed=3).train()
        patch_qwen3_5_fused_attn()

        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")
        out = m(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        out.sum().backward()

        for i, layer in enumerate(m.layers[:2]):
            if m.config.layer_types[i] != "full_attention":
                continue
            attn = layer.self_attn
            assert attn.q_norm.weight.grad is not None, f"layer {i} q_norm no grad"
            assert attn.k_norm.weight.grad is not None, f"layer {i} k_norm no grad"
            assert attn.q_norm.weight.grad.isfinite().all()
            assert attn.k_norm.weight.grad.isfinite().all()
            assert attn.q_norm.weight.grad.abs().sum() > 0
            assert attn.k_norm.weight.grad.abs().sum() > 0


class TestQwen3_5FusedAttnLoRACompose:
    """Pin LoRA-QKV → fused composition; ``QKV_PATCHES`` includes a chunk-2 variant for Qwen3.5's ``q_proj * 2``."""

    def _build_cfg(self):
        from axolotl.utils.dict import DictDefault

        return DictDefault(
            {
                "base_model": "fake/qwen3_5",
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
                "lora_dropout": 0.0,
            }
        )

    def test_lora_qkv_then_fused_does_not_raise(
        self, restore_qwen3_5_attention, monkeypatch
    ):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

        from axolotl.monkeypatch import lora_kernels
        from axolotl.monkeypatch.models.qwen3_5.fused_attn import (
            patch_qwen3_5_fused_attn,
        )

        monkeypatch.setattr(
            lora_kernels,
            "get_attention_cls_from_config",
            lambda _cfg: Qwen3_5Attention,
        )

        try:
            delattr(Qwen3_5Attention, "_original_forward")
        except AttributeError:
            pass

        try:
            lora_kernels.patch_self_attn_lora(self._build_cfg())
            assert hasattr(Qwen3_5Attention, "_original_forward"), (
                "patch_self_attn_lora must capture the stock Qwen3.5 forward — if "
                "this fails, QKV_PATCHES drifted away from the chunk-2 q_proj source"
            )
            patch_qwen3_5_fused_attn()
            assert getattr(Qwen3_5Attention, "_axolotl_fused_attn_patched", False)
        finally:
            try:
                delattr(Qwen3_5Attention, "_original_forward")
            except AttributeError:
                pass


@pytest.fixture
def restore_qwen3_5_moe_attention():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeAttention,
    )

    saved = Qwen3_5MoeAttention.forward
    saved_flag = getattr(Qwen3_5MoeAttention, "_axolotl_fused_attn_patched", False)
    yield Qwen3_5MoeAttention
    Qwen3_5MoeAttention.forward = saved
    if saved_flag:
        Qwen3_5MoeAttention._axolotl_fused_attn_patched = saved_flag
    else:
        _clear_patched_flag(Qwen3_5MoeAttention)


def _build_qwen3_5_moe_model(seed: int = 0):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextModel,
    )

    torch.manual_seed(seed)
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        layer_types=["full_attention", "full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3_5MoeTextModel(cfg).cuda().to(torch.bfloat16).eval()


class TestQwen3_5MoeFusedAttnParity:
    """End-to-end parity on the MoE variant (attention is structurally identical to dense Qwen3.5)."""

    def test_forward_matches_stock(self, restore_qwen3_5_moe_attention):
        from axolotl.monkeypatch.models.qwen3_5_moe.fused_attn import (
            patch_qwen3_5_moe_fused_attn,
        )

        m = _build_qwen3_5_moe_model(seed=5)
        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")

        with torch.no_grad():
            ref = m(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).last_hidden_state.clone()

        patch_qwen3_5_moe_fused_attn()
        with torch.no_grad():
            got = m(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).last_hidden_state.clone()

        assert got.shape == ref.shape
        assert torch.isfinite(got).all()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().float(), got.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"qwen3_5_moe end-to-end cosine_sim={cos_sim:.6f}"


class TestQwen3_5MoeFusedAttnBackward:
    """Backward grad flow through the fused Q/K-norm kernels on Qwen3.5-MoE."""

    def test_q_k_norm_grads_finite_nonzero(self, restore_qwen3_5_moe_attention):
        from axolotl.monkeypatch.models.qwen3_5_moe.fused_attn import (
            patch_qwen3_5_moe_fused_attn,
        )

        m = _build_qwen3_5_moe_model(seed=6).train()
        patch_qwen3_5_moe_fused_attn()

        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")
        out = m(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        out.sum().backward()

        for i, layer in enumerate(m.layers[:2]):
            if m.config.layer_types[i] != "full_attention":
                continue
            attn = layer.self_attn
            assert attn.q_norm.weight.grad is not None, f"layer {i} q_norm no grad"
            assert attn.k_norm.weight.grad is not None, f"layer {i} k_norm no grad"
            assert attn.q_norm.weight.grad.isfinite().all()
            assert attn.k_norm.weight.grad.isfinite().all()
            assert attn.q_norm.weight.grad.abs().sum() > 0
            assert attn.k_norm.weight.grad.abs().sum() > 0


class TestQwen3_5MoeFusedAttnLoRACompose:
    """MoE mirror of the Qwen3.5 LoRA-compose test."""

    def _build_cfg(self):
        from axolotl.utils.dict import DictDefault

        return DictDefault(
            {
                "base_model": "fake/qwen3_5_moe",
                "lora_qkv_kernel": True,
                "lora_o_kernel": True,
                "lora_dropout": 0.0,
            }
        )

    def test_lora_qkv_then_fused_does_not_raise(
        self, restore_qwen3_5_moe_attention, monkeypatch
    ):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeAttention,
        )

        from axolotl.monkeypatch import lora_kernels
        from axolotl.monkeypatch.models.qwen3_5_moe.fused_attn import (
            patch_qwen3_5_moe_fused_attn,
        )

        monkeypatch.setattr(
            lora_kernels,
            "get_attention_cls_from_config",
            lambda _cfg: Qwen3_5MoeAttention,
        )

        try:
            delattr(Qwen3_5MoeAttention, "_original_forward")
        except AttributeError:
            pass

        try:
            lora_kernels.patch_self_attn_lora(self._build_cfg())
            assert hasattr(Qwen3_5MoeAttention, "_original_forward")
            patch_qwen3_5_moe_fused_attn()
            assert getattr(Qwen3_5MoeAttention, "_axolotl_fused_attn_patched", False)
        finally:
            try:
                delattr(Qwen3_5MoeAttention, "_original_forward")
            except AttributeError:
                pass


class TestQwen3_5FusedAttnLigerRMSNormCompose:
    """Liger swaps ``Qwen3_5RMSNorm`` for a subclass that exposes ``variance_epsilon`` instead of ``eps``."""

    def test_forward_survives_liger_rmsnorm_swap(self, restore_qwen3_5_attention):
        from axolotl.monkeypatch.models.qwen3_5.fused_attn import (
            patch_qwen3_5_fused_attn,
        )

        m = _build_qwen3_5_text_model(seed=9)

        class _StubRMSNormVarEps(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.weight = original.weight
                self.variance_epsilon = original.eps

            def forward(self, x):
                return x

        for layer in m.layers:
            if not hasattr(layer, "self_attn"):
                continue
            attn = layer.self_attn
            attn.q_norm = _StubRMSNormVarEps(attn.q_norm)
            attn.k_norm = _StubRMSNormVarEps(attn.k_norm)

        patch_qwen3_5_fused_attn()
        ids = torch.randint(0, 128, (2, 16), device="cuda")
        mask = torch.ones(2, 16, dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask, use_cache=False)
        assert torch.isfinite(out.last_hidden_state).all()


class TestPatchManagerQwen3_5TextDispatch:
    """Pin that ``_apply_model_specific_patches`` covers the ``*_text`` config types of multimodal Qwen3.5 / Qwen3.5-MoE checkpoints."""

    @pytest.mark.parametrize("model_config_type", ["qwen3_5", "qwen3_5_text"])
    def test_qwen3_5_text_variant_is_patched(
        self, restore_qwen3_5_attention, model_config_type
    ):
        from axolotl.loaders.patch_manager import PatchManager
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault(
            {
                "base_model": "fake/qwen3_5",
                "model_config_type": model_config_type,
                "fused_attn_kernel": True,
                "lora_qkv_kernel": False,
                "lora_o_kernel": False,
                "context_parallel_size": 1,
            }
        )
        mc = type("MC", (), {"model_type": model_config_type})()
        pm = PatchManager(cfg=cfg, model_config=mc, inference=False)
        pm._apply_model_specific_patches()

        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

        assert getattr(Qwen3_5Attention, "_axolotl_fused_attn_patched", False), (
            f"PatchManager skipped fused-attn for model_config_type="
            f"{model_config_type!r}; dispatch is missing the _text variant"
        )

    @pytest.mark.parametrize("model_config_type", ["qwen3_5_moe", "qwen3_5_moe_text"])
    def test_qwen3_5_moe_text_variant_is_patched(
        self, restore_qwen3_5_moe_attention, model_config_type
    ):
        from axolotl.loaders.patch_manager import PatchManager
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault(
            {
                "base_model": "fake/qwen3_5_moe",
                "model_config_type": model_config_type,
                "fused_attn_kernel": True,
                "lora_qkv_kernel": False,
                "lora_o_kernel": False,
                "context_parallel_size": 1,
            }
        )
        mc = type("MC", (), {"model_type": model_config_type})()
        pm = PatchManager(cfg=cfg, model_config=mc, inference=False)
        pm._apply_model_specific_patches()

        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeAttention,
        )

        assert getattr(Qwen3_5MoeAttention, "_axolotl_fused_attn_patched", False), (
            f"PatchManager skipped fused-attn for model_config_type="
            f"{model_config_type!r}; dispatch is missing the _text variant"
        )
