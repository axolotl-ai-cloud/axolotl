"""
Tests for LoRA kernel correctness with bias, dropout, and DoRA support.

Compares fused kernel outputs and gradients against PEFT's reference implementation.
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from axolotl.kernels.lora import (
    LoRA_MLP,
    LoRA_O,
    LoRA_QKV,
    _apply_dropout,
    _compute_dora_scale,
    _lora_only,
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
    get_lora_parameters,
    matmul_lora,
)
from axolotl.monkeypatch.lora_kernels import (
    apply_lora_kernel_patches,
    patch_self_attn_lora,
)
from axolotl.utils.dict import DictDefault

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda"
DTYPE = torch.bfloat16


@pytest.fixture(scope="module")
def model_config():
    return AutoConfig.from_pretrained(MODEL_NAME)


def _make_peft_model(
    lora_dropout=0.0,
    bias="none",
    use_dora=False,
    target_modules=None,
):
    """Create a PEFT model with given config."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, attn_implementation="eager",
    ).to(DEVICE)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=lora_dropout,
        bias=bias,
        use_dora=use_dora,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def _get_layer(peft_model, layer_idx=0):
    """Get a specific transformer layer from the model."""
    return peft_model.model.model.layers[layer_idx]


def _make_input(batch=2, seq_len=16, hidden_size=1024):
    """Create random input tensor."""
    return torch.randn(batch, seq_len, hidden_size, dtype=DTYPE, device=DEVICE,
                        requires_grad=True)


def _compare_tensors(a, b, name="", atol=1e-2, rtol=1e-2):
    """Compare two tensors with informative error messages."""
    if a is None and b is None:
        return
    assert a is not None and b is not None, f"{name}: one is None, other is not"
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert torch.allclose(a, b, atol=atol, rtol=rtol), (
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )


class TestGetLoraParameters:
    """Test the extended get_lora_parameters function."""

    def test_returns_9_values(self):
        model = _make_peft_model()
        layer = _get_layer(model)
        params = get_lora_parameters(layer.self_attn.q_proj)
        assert len(params) == 9
        W, b, quant, A, B, s, lora_bias, dropout, magnitude = params
        assert W is not None
        assert A is not None
        assert B is not None
        assert s is not None
        assert lora_bias is None  # bias="none"
        assert dropout is not None  # should be nn.Identity
        assert magnitude is None  # no DoRA
        del model

    def test_with_bias(self):
        """Qwen3 has no base bias, so PEFT doesn't add lora_bias even with bias='lora_only'.
        This test verifies get_lora_parameters handles this correctly."""
        model = _make_peft_model(bias="lora_only")
        layer = _get_layer(model)
        params = get_lora_parameters(layer.self_attn.q_proj)
        _, _, _, _, _, _, lora_bias, _, _ = params
        # Qwen3 q_proj has no base bias, so PEFT sets lora_bias=False
        assert lora_bias is None
        del model

    def test_with_bias_on_biased_layer(self):
        """Test with manually added bias to verify lora_bias extraction."""
        model = _make_peft_model(bias="lora_only")
        layer = _get_layer(model)
        q_proj = layer.self_attn.q_proj
        adapter = q_proj.active_adapters[0]
        # Manually add bias to lora_B to test extraction
        old_B = q_proj.lora_B[adapter]
        q_proj.lora_B[adapter] = torch.nn.Linear(
            old_B.in_features, old_B.out_features, bias=True, device=DEVICE, dtype=DTYPE
        )
        params = get_lora_parameters(q_proj)
        _, _, _, _, _, _, lora_bias, _, _ = params
        assert lora_bias is not None
        assert lora_bias.shape[0] == old_B.out_features
        del model

    def test_with_dropout(self):
        model = _make_peft_model(lora_dropout=0.1)
        layer = _get_layer(model)
        params = get_lora_parameters(layer.self_attn.q_proj)
        _, _, _, _, _, _, _, dropout, _ = params
        assert dropout is not None
        assert isinstance(dropout, nn.Dropout)
        del model

    def test_with_dora(self):
        model = _make_peft_model(use_dora=True)
        layer = _get_layer(model)
        params = get_lora_parameters(layer.self_attn.q_proj)
        _, _, _, _, _, _, _, _, magnitude = params
        assert magnitude is not None
        del model


class TestMatmulLora:
    """Test matmul_lora with new lora_bias and X_drop parameters."""

    def test_basic(self):
        X = torch.randn(4, 8, dtype=DTYPE, device=DEVICE)
        W = torch.randn(16, 8, dtype=DTYPE, device=DEVICE)
        A = torch.randn(4, 8, dtype=DTYPE, device=DEVICE)  # [rank, in]
        B = torch.randn(16, 4, dtype=DTYPE, device=DEVICE)  # [out, rank]
        s = 2.0

        result = matmul_lora(X, W, None, None, A, B, s)
        expected = X @ W.t() + s * X @ A.t() @ B.t()
        _compare_tensors(result, expected, "basic matmul_lora")

    def test_with_lora_bias(self):
        X = torch.randn(4, 8, dtype=DTYPE, device=DEVICE)
        W = torch.randn(16, 8, dtype=DTYPE, device=DEVICE)
        A = torch.randn(4, 8, dtype=DTYPE, device=DEVICE)
        B = torch.randn(16, 4, dtype=DTYPE, device=DEVICE)
        lora_bias = torch.randn(16, dtype=DTYPE, device=DEVICE)
        s = 2.0

        result = matmul_lora(X, W, None, None, A, B, s, lora_bias=lora_bias)
        expected = X @ W.t() + s * X @ A.t() @ B.t() + s * lora_bias
        _compare_tensors(result, expected, "matmul_lora with lora_bias")

    def test_with_x_drop(self):
        X = torch.randn(4, 8, dtype=DTYPE, device=DEVICE)
        X_drop = X * 0.5  # simulated dropout
        W = torch.randn(16, 8, dtype=DTYPE, device=DEVICE)
        A = torch.randn(4, 8, dtype=DTYPE, device=DEVICE)
        B = torch.randn(16, 4, dtype=DTYPE, device=DEVICE)
        s = 2.0

        result = matmul_lora(X, W, None, None, A, B, s, X_drop=X_drop)
        expected = X @ W.t() + s * X_drop @ A.t() @ B.t()
        _compare_tensors(result, expected, "matmul_lora with X_drop")


class TestDoraScale:
    """Test DoRA magnitude/norm scaling computation."""

    def test_basic(self):
        W = torch.randn(16, 8, dtype=DTYPE, device=DEVICE)
        A = torch.randn(4, 8, dtype=DTYPE, device=DEVICE)
        B = torch.randn(16, 4, dtype=DTYPE, device=DEVICE)
        magnitude = torch.randn(16, dtype=DTYPE, device=DEVICE).abs() + 0.1
        s = 2.0

        scale = _compute_dora_scale(W, None, A, B, s, magnitude, DTYPE)

        # Manual computation
        combined = W + s * B @ A
        weight_norm = torch.linalg.norm(combined, dim=1)
        expected = magnitude / weight_norm

        _compare_tensors(scale, expected, "dora_scale")


# ============================================================
# Integration tests: compare kernel outputs against PEFT reference
# ============================================================


def _run_peft_qkv(layer, X):
    """Run Q, K, V projections through PEFT's standard forward."""
    Q = layer.self_attn.q_proj(X)
    K = layer.self_attn.k_proj(X)
    V = layer.self_attn.v_proj(X)
    return Q, K, V


def _run_kernel_qkv(layer, X):
    """Run Q, K, V projections through our fused kernel."""
    return apply_lora_qkv(layer.self_attn, X, inplace=False)


def _run_peft_o(layer, X):
    """Run O projection through PEFT's standard forward."""
    return layer.self_attn.o_proj(X)


def _run_kernel_o(layer, X):
    """Run O projection through our fused kernel."""
    return apply_lora_o(layer.self_attn, X)


def _run_peft_mlp(layer, X):
    """Run MLP through PEFT's standard forward."""
    return layer.mlp(X)


def _run_kernel_mlp(layer, X):
    """Run MLP through our fused kernel."""
    return apply_lora_mlp_swiglu(layer.mlp, X, inplace=False)


class TestQKVKernel:
    """Test LoRA_QKV kernel against PEFT reference."""

    @pytest.mark.parametrize("bias", ["none", "lora_only"])
    def test_forward_bias(self, bias):
        model = _make_peft_model(bias=bias)
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=model.config.hidden_size)

        with torch.no_grad():
            peft_Q, peft_K, peft_V = _run_peft_qkv(layer, X)
            kern_Q, kern_K, kern_V = _run_kernel_qkv(layer, X)

        _compare_tensors(kern_Q, peft_Q, f"QKV Q (bias={bias})")
        _compare_tensors(kern_K, peft_K, f"QKV K (bias={bias})")
        _compare_tensors(kern_V, peft_V, f"QKV V (bias={bias})")
        del model

    def test_forward_dropout_eval(self):
        """Dropout disabled in eval - should match exactly."""
        model = _make_peft_model(lora_dropout=0.1)
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=model.config.hidden_size)

        with torch.no_grad():
            peft_Q, peft_K, peft_V = _run_peft_qkv(layer, X)
            kern_Q, kern_K, kern_V = _run_kernel_qkv(layer, X)

        _compare_tensors(kern_Q, peft_Q, "QKV Q (dropout eval)")
        _compare_tensors(kern_K, peft_K, "QKV K (dropout eval)")
        _compare_tensors(kern_V, peft_V, "QKV V (dropout eval)")
        del model

    def test_forward_dora(self):
        model = _make_peft_model(use_dora=True)
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=model.config.hidden_size)

        with torch.no_grad():
            peft_Q, peft_K, peft_V = _run_peft_qkv(layer, X)
            kern_Q, kern_K, kern_V = _run_kernel_qkv(layer, X)

        _compare_tensors(kern_Q, peft_Q, "QKV Q (DoRA)")
        _compare_tensors(kern_K, peft_K, "QKV K (DoRA)")
        _compare_tensors(kern_V, peft_V, "QKV V (DoRA)")
        del model

    def test_forward_dora_bias(self):
        model = _make_peft_model(use_dora=True, bias="lora_only")
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=model.config.hidden_size)

        with torch.no_grad():
            peft_Q, peft_K, peft_V = _run_peft_qkv(layer, X)
            kern_Q, kern_K, kern_V = _run_kernel_qkv(layer, X)

        _compare_tensors(kern_Q, peft_Q, "QKV Q (DoRA+bias)")
        _compare_tensors(kern_K, peft_K, "QKV K (DoRA+bias)")
        _compare_tensors(kern_V, peft_V, "QKV V (DoRA+bias)")
        del model

    @pytest.mark.parametrize("bias", ["none", "lora_only"])
    def test_backward_bias(self, bias):
        """Test that gradients match between kernel and PEFT."""
        model = _make_peft_model(bias=bias)
        model.train()
        layer = _get_layer(model)

        # PEFT reference
        X1 = _make_input(hidden_size=model.config.hidden_size)
        pQ, pK, pV = _run_peft_qkv(layer, X1)
        loss_peft = (pQ.sum() + pK.sum() + pV.sum())
        loss_peft.backward()

        peft_grads = {}
        for name, param in layer.self_attn.named_parameters():
            if param.grad is not None:
                peft_grads[name] = param.grad.clone()
        layer.self_attn.zero_grad()

        # Kernel
        X2 = X1.detach().clone().requires_grad_(True)
        kQ, kK, kV = _run_kernel_qkv(layer, X2)
        loss_kern = (kQ.sum() + kK.sum() + kV.sum())
        loss_kern.backward()

        kern_grads = {}
        for name, param in layer.self_attn.named_parameters():
            if param.grad is not None:
                kern_grads[name] = param.grad.clone()
        layer.self_attn.zero_grad()

        # Compare LoRA parameter gradients
        for name in peft_grads:
            if "lora_" in name:
                _compare_tensors(
                    kern_grads.get(name), peft_grads[name],
                    f"grad {name} (bias={bias})", atol=5e-2, rtol=5e-2,
                )
        del model

    def test_backward_dora(self):
        """Test DoRA backward pass gradients."""
        model = _make_peft_model(use_dora=True)
        model.train()
        layer = _get_layer(model)

        X1 = _make_input(hidden_size=model.config.hidden_size)
        pQ, pK, pV = _run_peft_qkv(layer, X1)
        loss_peft = (pQ.sum() + pK.sum() + pV.sum())
        loss_peft.backward()

        peft_grads = {}
        for name, param in layer.self_attn.named_parameters():
            if param.grad is not None:
                peft_grads[name] = param.grad.clone()
        layer.self_attn.zero_grad()

        X2 = X1.detach().clone().requires_grad_(True)
        kQ, kK, kV = _run_kernel_qkv(layer, X2)
        loss_kern = (kQ.sum() + kK.sum() + kV.sum())
        loss_kern.backward()

        kern_grads = {}
        for name, param in layer.self_attn.named_parameters():
            if param.grad is not None:
                kern_grads[name] = param.grad.clone()
        layer.self_attn.zero_grad()

        for name in peft_grads:
            if "lora_" in name or "magnitude" in name:
                _compare_tensors(
                    kern_grads.get(name), peft_grads[name],
                    f"grad {name} (DoRA)", atol=5e-2, rtol=5e-2,
                )
        del model


class TestOKernel:
    """Test LoRA_O kernel against PEFT reference."""

    @staticmethod
    def _o_input_dim(model):
        """o_proj input is num_heads * head_dim (may differ from hidden_size with GQA)."""
        cfg = model.config
        text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        return text_cfg.num_attention_heads * text_cfg.head_dim

    @pytest.mark.parametrize("bias", ["none", "lora_only"])
    def test_forward_bias(self, bias):
        model = _make_peft_model(bias=bias)
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=self._o_input_dim(model))

        with torch.no_grad():
            peft_out = _run_peft_o(layer, X)
            kern_out = _run_kernel_o(layer, X)

        _compare_tensors(kern_out, peft_out, f"O (bias={bias})")
        del model

    def test_forward_dora(self):
        model = _make_peft_model(use_dora=True)
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=self._o_input_dim(model))

        with torch.no_grad():
            peft_out = _run_peft_o(layer, X)
            kern_out = _run_kernel_o(layer, X)

        _compare_tensors(kern_out, peft_out, "O (DoRA)")
        del model

    @pytest.mark.parametrize("bias", ["none", "lora_only"])
    def test_backward_bias(self, bias):
        model = _make_peft_model(bias=bias)
        model.train()
        layer = _get_layer(model)

        X1 = _make_input(hidden_size=self._o_input_dim(model))
        peft_out = _run_peft_o(layer, X1)
        peft_out.sum().backward()
        peft_grads = {n: p.grad.clone() for n, p in layer.self_attn.o_proj.named_parameters() if p.grad is not None}
        layer.self_attn.o_proj.zero_grad()

        X2 = X1.detach().clone().requires_grad_(True)
        kern_out = _run_kernel_o(layer, X2)
        kern_out.sum().backward()
        kern_grads = {n: p.grad.clone() for n, p in layer.self_attn.o_proj.named_parameters() if p.grad is not None}
        layer.self_attn.o_proj.zero_grad()

        for name in peft_grads:
            if "lora_" in name:
                _compare_tensors(kern_grads.get(name), peft_grads[name],
                                 f"O grad {name} (bias={bias})", atol=5e-2, rtol=5e-2)
        del model


class TestMLPKernel:
    """Test LoRA_MLP kernel against PEFT reference."""

    @pytest.mark.parametrize("bias", ["none", "lora_only"])
    def test_forward_bias(self, bias):
        model = _make_peft_model(bias=bias)
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=model.config.hidden_size)

        with torch.no_grad():
            peft_out = _run_peft_mlp(layer, X)
            kern_out = _run_kernel_mlp(layer, X)

        _compare_tensors(kern_out, peft_out, f"MLP (bias={bias})")
        del model

    def test_forward_dropout_eval(self):
        model = _make_peft_model(lora_dropout=0.1)
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=model.config.hidden_size)

        with torch.no_grad():
            peft_out = _run_peft_mlp(layer, X)
            kern_out = _run_kernel_mlp(layer, X)

        _compare_tensors(kern_out, peft_out, "MLP (dropout eval)")
        del model

    def test_forward_dora(self):
        model = _make_peft_model(use_dora=True)
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=model.config.hidden_size)

        with torch.no_grad():
            peft_out = _run_peft_mlp(layer, X)
            kern_out = _run_kernel_mlp(layer, X)

        # Relaxed tolerance for MLP DoRA: 3 projections + activation + DoRA
        # causes bf16 accumulation differences
        _compare_tensors(kern_out, peft_out, "MLP (DoRA)", atol=0.3, rtol=0.05)
        del model

    def test_forward_dora_bias(self):
        model = _make_peft_model(use_dora=True, bias="lora_only")
        model.eval()
        layer = _get_layer(model)
        X = _make_input(hidden_size=model.config.hidden_size)

        with torch.no_grad():
            peft_out = _run_peft_mlp(layer, X)
            kern_out = _run_kernel_mlp(layer, X)

        _compare_tensors(kern_out, peft_out, "MLP (DoRA+bias)", atol=0.3, rtol=0.05)
        del model

    @pytest.mark.parametrize("bias", ["none", "lora_only"])
    def test_backward_bias(self, bias):
        model = _make_peft_model(bias=bias)
        model.train()
        layer = _get_layer(model)
        hidden_size = model.config.hidden_size

        X1 = _make_input(hidden_size=hidden_size)
        peft_out = _run_peft_mlp(layer, X1)
        peft_out.sum().backward()
        peft_grads = {n: p.grad.clone() for n, p in layer.mlp.named_parameters() if p.grad is not None}
        layer.mlp.zero_grad()

        X2 = X1.detach().clone().requires_grad_(True)
        kern_out = _run_kernel_mlp(layer, X2)
        kern_out.sum().backward()
        kern_grads = {n: p.grad.clone() for n, p in layer.mlp.named_parameters() if p.grad is not None}
        layer.mlp.zero_grad()

        # MLP backward has longer chain (3 projections + activation) = more bf16 accumulation error
        for name in peft_grads:
            if "lora_" in name:
                _compare_tensors(kern_grads.get(name), peft_grads[name],
                                 f"MLP grad {name} (bias={bias})", atol=0.5, rtol=0.1)
        del model

    def test_backward_dora(self):
        model = _make_peft_model(use_dora=True)
        model.train()
        layer = _get_layer(model)

        X1 = _make_input(hidden_size=model.config.hidden_size)
        peft_out = _run_peft_mlp(layer, X1)
        peft_out.sum().backward()
        peft_grads = {n: p.grad.clone() for n, p in layer.mlp.named_parameters() if p.grad is not None}
        layer.mlp.zero_grad()

        X2 = X1.detach().clone().requires_grad_(True)
        kern_out = _run_kernel_mlp(layer, X2)
        kern_out.sum().backward()
        kern_grads = {n: p.grad.clone() for n, p in layer.mlp.named_parameters() if p.grad is not None}
        layer.mlp.zero_grad()

        for name in peft_grads:
            if "lora_" in name or "magnitude" in name:
                _compare_tensors(kern_grads.get(name), peft_grads[name],
                                 f"MLP grad {name} (DoRA)", atol=0.5, rtol=0.1)
        del model


class TestFullModelPatch:
    """Test applying kernel patches to a full model."""

    def test_patched_forward_basic(self):
        """Test that patched model forward matches unpatched PEFT model (bias=none, no DoRA)."""
        from peft import PeftModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, attn_implementation="eager",
        ).to(DEVICE)
        lora_config = LoraConfig(
            r=8, lora_alpha=16, bias="none", use_dora=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = PeftModelForCausalLM(base_model, lora_config)
        model.eval()

        # Get PEFT reference output
        input_ids = torch.randint(0, 1000, (1, 32), device=DEVICE)
        with torch.no_grad():
            peft_out = model(input_ids).logits

        # Apply kernel patches
        cfg = DictDefault({
            "base_model": MODEL_NAME,
            "lora_qkv_kernel": True,
            "lora_o_kernel": True,
            "lora_mlp_kernel": True,
        })
        patch_self_attn_lora(cfg)
        apply_lora_kernel_patches(model, cfg)

        # Get kernel output
        with torch.no_grad():
            kern_out = model(input_ids).logits

        _compare_tensors(kern_out, peft_out,
                         "Full model (basic)",
                         atol=5e-1, rtol=1e-1)
        del model


class TestEmbeddingKernel:
    """Test LoRA embedding kernel against PEFT reference."""

    def _make_embedding_model(self, use_dora=False):
        from peft import PeftModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, attn_implementation="eager",
        ).to(DEVICE)
        lora_config = LoraConfig(
            r=8, lora_alpha=16, use_dora=use_dora,
            target_modules=["embed_tokens"],
        )
        return PeftModelForCausalLM(model, lora_config)

    def test_forward_basic(self):
        from axolotl.kernels.lora import apply_lora_embedding
        model = self._make_embedding_model()
        model.eval()

        embed = model.model.model.embed_tokens
        input_ids = torch.randint(0, 1000, (2, 16), device=DEVICE)

        with torch.no_grad():
            peft_out = embed(input_ids)
            kern_out = apply_lora_embedding(embed, input_ids)

        # Cast to same dtype for comparison (PEFT may return float32)
        _compare_tensors(kern_out.to(peft_out.dtype), peft_out, "Embedding basic")
        del model

    def test_forward_dora(self):
        from axolotl.kernels.lora import apply_lora_embedding
        model = self._make_embedding_model(use_dora=True)
        model.eval()

        embed = model.model.model.embed_tokens
        input_ids = torch.randint(0, 1000, (2, 16), device=DEVICE)

        with torch.no_grad():
            peft_out = embed(input_ids)
            kern_out = apply_lora_embedding(embed, input_ids)

        _compare_tensors(kern_out.to(peft_out.dtype), peft_out, "Embedding DoRA", atol=0.3, rtol=0.05)
        del model

    def test_backward(self):
        from axolotl.kernels.lora import apply_lora_embedding
        model = self._make_embedding_model()
        model.train()

        embed = model.model.model.embed_tokens
        input_ids = torch.randint(0, 1000, (2, 16), device=DEVICE)

        # PEFT reference
        peft_out = embed(input_ids)
        peft_out.sum().backward()
        peft_grads = {}
        for n, p in embed.named_parameters():
            if p.grad is not None and "lora" in n:
                peft_grads[n] = p.grad.clone()
        embed.zero_grad()

        # Kernel
        kern_out = apply_lora_embedding(embed, input_ids)
        kern_out.sum().backward()
        kern_grads = {}
        for n, p in embed.named_parameters():
            if p.grad is not None and "lora" in n:
                kern_grads[n] = p.grad.clone()
        embed.zero_grad()

        for name in peft_grads:
            _compare_tensors(
                kern_grads.get(name), peft_grads[name],
                f"Embedding grad {name}", atol=5e-2, rtol=5e-2,
            )
        del model


class TestTiedEmbeddings:
    """Test that tied embeddings work correctly with kernel patching."""

    def test_tied_embed_and_lm_head(self):
        """When both embed_tokens and lm_head have LoRA, PEFT unties them.
        Verify patched model produces valid output (no crashes, finite values)."""
        from peft import PeftModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, attn_implementation="eager",
        ).to(DEVICE)
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj",
                            "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = PeftModelForCausalLM(base, lora_config)
        model.eval()

        cfg = DictDefault({
            "base_model": MODEL_NAME,
            "lora_qkv_kernel": True,
            "lora_o_kernel": True,
            "lora_mlp_kernel": True,
            "lora_embedding_kernel": True,
        })

        # Apply all kernel patches (class + instance level)
        patch_self_attn_lora(cfg)
        apply_lora_kernel_patches(model, cfg)

        input_ids = torch.randint(0, 1000, (1, 32), device=DEVICE)
        with torch.no_grad():
            out = model(input_ids).logits

        # Verify output is valid
        assert out.shape == (1, 32, model.config.vocab_size)
        assert torch.isfinite(out).all(), "Output contains non-finite values"
        assert out.abs().max() > 0, "Output is all zeros"

        # Verify backward works
        model.train()
        out = model(input_ids).logits
        out.sum().backward()
        # Check that LoRA params got gradients
        embed = model.model.model.embed_tokens
        has_embed_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in embed.named_parameters() if "lora" in n
        )
        assert has_embed_grad, "Embedding LoRA params got no gradients"
        del model


class TestQuantizedModels:
    """Test kernels with quantized base weights."""

    def test_nf4_qlora_forward_backward(self):
        """NF4 QLoRA with kernel patches."""
        from peft import PeftModelForCausalLM
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config,
            attn_implementation="eager",
        )
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = PeftModelForCausalLM(base, lora_config)

        cfg = DictDefault({
            "base_model": MODEL_NAME,
            "lora_qkv_kernel": True,
            "lora_o_kernel": True,
            "lora_mlp_kernel": True,
        })
        patch_self_attn_lora(cfg)
        apply_lora_kernel_patches(model, cfg)
        model.train()

        ids = torch.randint(0, 1000, (1, 32), device=DEVICE)
        out = model(ids).logits
        assert torch.isfinite(out).all()
        out.sum().backward()
        has_grads = sum(1 for n, p in model.named_parameters()
                        if p.grad is not None and "lora" in n)
        assert has_grads > 0, "No LoRA gradients"
        del model

    def test_nf4_single_quant(self):
        """NF4 without double quantization."""
        from peft import PeftModelForCausalLM
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config,
            attn_implementation="eager",
        )
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = PeftModelForCausalLM(base, lora_config)

        cfg = DictDefault({
            "base_model": MODEL_NAME,
            "lora_qkv_kernel": True,
            "lora_o_kernel": True,
            "lora_mlp_kernel": True,
        })
        patch_self_attn_lora(cfg)
        apply_lora_kernel_patches(model, cfg)
        model.train()

        ids = torch.randint(0, 1000, (1, 32), device=DEVICE)
        out = model(ids).logits
        assert torch.isfinite(out).all()
        out.sum().backward()
        has_grads = sum(1 for n, p in model.named_parameters()
                        if p.grad is not None and "lora" in n)
        assert has_grads > 0
        del model


class TestTritonDoRA:
    """Test Triton DoRA kernel against reference implementation."""

    def test_triton_dora_scale(self):
        from axolotl.kernels.dora import triton_dora_scale
        from axolotl.kernels.lora import _compute_dora_scale

        # Random weights matching Qwen3-1.7B dimensions
        out_feat, in_feat, rank = 1024, 1024, 8
        W = torch.randn(out_feat, in_feat, dtype=DTYPE, device=DEVICE)
        A = torch.randn(rank, in_feat, dtype=DTYPE, device=DEVICE)
        B = torch.randn(out_feat, rank, dtype=DTYPE, device=DEVICE)
        magnitude = torch.randn(out_feat, dtype=DTYPE, device=DEVICE).abs() + 0.1
        s = 2.0

        # Clear cache to force recomputation
        if hasattr(magnitude, "_dora_cache"):
            del magnitude._dora_cache

        ref = _compute_dora_scale(W, None, A, B, s, magnitude, DTYPE)
        tri = triton_dora_scale(W, None, A, B, s, magnitude, DTYPE)

        _compare_tensors(tri, ref, "Triton DoRA scale", atol=1e-2, rtol=1e-2)

    def test_triton_dora_scale_small(self):
        """Test with K/V projection dimensions (smaller out_features)."""
        from axolotl.kernels.dora import triton_dora_scale
        from axolotl.kernels.lora import _compute_dora_scale

        out_feat, in_feat, rank = 128, 1024, 8
        W = torch.randn(out_feat, in_feat, dtype=DTYPE, device=DEVICE)
        A = torch.randn(rank, in_feat, dtype=DTYPE, device=DEVICE)
        B = torch.randn(out_feat, rank, dtype=DTYPE, device=DEVICE)
        magnitude = torch.randn(out_feat, dtype=DTYPE, device=DEVICE).abs() + 0.1
        s = 2.0

        if hasattr(magnitude, "_dora_cache"):
            del magnitude._dora_cache

        ref = _compute_dora_scale(W, None, A, B, s, magnitude, DTYPE)
        tri = triton_dora_scale(W, None, A, B, s, magnitude, DTYPE)

        _compare_tensors(tri, ref, "Triton DoRA scale (small)", atol=1e-2, rtol=1e-2)
