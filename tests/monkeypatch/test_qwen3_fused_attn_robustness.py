"""Robustness tests for the Qwen3 / Qwen3.5 fused-attn patches (idempotency, signature drift, GC, cross-device, FA2)."""

import inspect

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

pytest.importorskip("transformers.models.qwen3")


def _clear_patched_flag(cls):
    try:
        delattr(cls, "_axolotl_fused_attn_patched")
    except AttributeError:
        pass


def _saved_state(cls):
    return (cls.forward, getattr(cls, "_axolotl_fused_attn_patched", False))


def _restore_state(cls, state):
    saved_forward, saved_flag = state
    cls.forward = saved_forward
    if saved_flag:
        cls._axolotl_fused_attn_patched = saved_flag
    else:
        _clear_patched_flag(cls)


@pytest.fixture
def restore_qwen3_attention():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    state = _saved_state(Qwen3Attention)
    yield Qwen3Attention
    _restore_state(Qwen3Attention, state)


def _build_tiny_qwen3():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    cfg = Qwen3Config(
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
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3Model(cfg).cuda().to(torch.bfloat16)


class TestPatchIdempotency:
    """Re-applying the patch must be a no-op (``_axolotl_fused_attn_patched`` flag guard)."""

    def test_qwen3_double_patch_is_noop(self, restore_qwen3_attention):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        from axolotl.monkeypatch.models.qwen3.fused_attn import (
            patch_qwen3_fused_attn,
        )

        patch_qwen3_fused_attn()
        forward_after_first = Qwen3Attention.forward
        assert Qwen3Attention._axolotl_fused_attn_patched is True

        patch_qwen3_fused_attn()
        assert Qwen3Attention.forward is forward_after_first, (
            "second patch_qwen3_fused_attn() call replaced .forward — the "
            "_axolotl_fused_attn_patched guard is broken"
        )

    def test_qwen3_5_double_patch_is_noop(self):
        pytest.importorskip("transformers.models.qwen3_5")
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

        from axolotl.monkeypatch.models.qwen3_5.fused_attn import (
            patch_qwen3_5_fused_attn,
        )

        state = _saved_state(Qwen3_5Attention)
        try:
            patch_qwen3_5_fused_attn()
            forward_after_first = Qwen3_5Attention.forward
            patch_qwen3_5_fused_attn()
            assert Qwen3_5Attention.forward is forward_after_first
        finally:
            _restore_state(Qwen3_5Attention, state)


class TestSignatureContract:
    """Pin the stock attention forward signature; transformers drift would otherwise surface as a confusing TypeError mid-training."""

    def test_qwen3_forward_required_params(self):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        sig = inspect.signature(Qwen3Attention.forward)
        params = list(sig.parameters)
        assert params[:4] == [
            "self",
            "hidden_states",
            "position_embeddings",
            "attention_mask",
        ], (
            "Qwen3Attention.forward signature drifted away from the contract "
            "our fused_forward replacement assumes — update the patch."
        )

    def test_qwen3_5_forward_required_params(self):
        pytest.importorskip("transformers.models.qwen3_5")
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

        sig = inspect.signature(Qwen3_5Attention.forward)
        params = list(sig.parameters)
        assert params[:4] == [
            "self",
            "hidden_states",
            "position_embeddings",
            "attention_mask",
        ]


class TestGradientCheckpointingCompose:
    """Pin that the fused forward survives being re-run inside a checkpoint partial during backward."""

    def test_qwen3_fused_under_gradient_checkpointing(self, restore_qwen3_attention):
        from axolotl.monkeypatch.models.qwen3.fused_attn import (
            patch_qwen3_fused_attn,
        )

        m = _build_tiny_qwen3().train()
        m.gradient_checkpointing_enable()
        patch_qwen3_fused_attn()

        ids = torch.randint(0, 128, (2, 32), device="cuda")
        mask = torch.ones(2, 32, dtype=torch.long, device="cuda")
        out = m(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        loss = out.sum()
        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

        for layer in m.layers:
            attn = layer.self_attn
            assert attn.q_norm.weight.grad is not None
            assert attn.k_norm.weight.grad is not None
            assert attn.q_norm.weight.grad.isfinite().all()
            assert attn.k_norm.weight.grad.isfinite().all()
            assert attn.q_norm.weight.grad.abs().sum() > 0
        assert peak_mb < 1024, f"backward peak {peak_mb:.0f} MB looks like a leak"


class TestCrossDeviceNormWeight:
    """Sharded ``device_map='auto'`` can leave norm weights on CPU; the patch must coerce them or Triton raises on the CPU pointer."""

    def test_qwen3_norm_weight_on_cpu_does_not_crash(self, restore_qwen3_attention):
        from axolotl.monkeypatch.models.qwen3.fused_attn import (
            patch_qwen3_fused_attn,
        )

        m = _build_tiny_qwen3()
        patch_qwen3_fused_attn()

        for layer in m.layers:
            attn = layer.self_attn
            attn.q_norm.weight.data = attn.q_norm.weight.data.cpu()
            attn.k_norm.weight.data = attn.k_norm.weight.data.cpu()

        ids = torch.randint(0, 128, (1, 16), device="cuda")
        mask = torch.ones(1, 16, dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask).last_hidden_state
        assert torch.isfinite(out).all()


class TestFlashAttention2Compose:
    """The fused region is upstream of ``attention_interface``; pin clean composition with FA2 if it's installed."""

    def test_qwen3_fused_under_flash_attention_2(self, restore_qwen3_attention):
        pytest.importorskip("flash_attn")
        from axolotl.monkeypatch.models.qwen3.fused_attn import (
            patch_qwen3_fused_attn,
        )

        m = _build_tiny_qwen3()
        m.config._attn_implementation = "flash_attention_2"
        for layer in m.layers:
            layer.self_attn.config._attn_implementation = "flash_attention_2"

        patch_qwen3_fused_attn()
        ids = torch.randint(0, 128, (2, 32), device="cuda")
        mask = torch.ones(2, 32, dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = m(input_ids=ids, attention_mask=mask).last_hidden_state
        assert torch.isfinite(out).all()
