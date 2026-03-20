"""
End-to-end gradient and convergence tests for SonicMoE integration.

Requires:
    - H100/H200 GPU (SonicMoE CUTLASS kernels target sm_90)
    - sonicmoe package installed
    - transformers with Qwen3MoE support

Usage:
    pytest tests/e2e/integrations/test_sonicmoe.py -v -s
"""

import importlib.util
import math

import pytest
import torch

_sonicmoe_available = importlib.util.find_spec("sonicmoe") is not None
_is_hopper = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU"),
    pytest.mark.skipif(
        not _is_hopper, reason="SonicMoE CUTLASS kernels require Hopper (sm_90)"
    ),
    pytest.mark.skipif(not _sonicmoe_available, reason="SonicMoE not installed"),
]


def _create_tiny_qwen3_config():
    """Create a minimal Qwen3MoE config for fast testing."""
    from transformers import AutoConfig

    config = AutoConfig.for_model("qwen3_moe")
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.moe_intermediate_size = 64
    config.num_attention_heads = 16
    config.num_key_value_heads = 2
    config.head_dim = 32
    config.num_hidden_layers = 2
    config.num_experts = 8
    config.num_experts_per_tok = 2
    config.vocab_size = 1000
    config.max_position_embeddings = 128
    config.norm_topk_prob = True
    config.torch_dtype = torch.bfloat16
    return config


def _interleave_gate_up_weights(model):
    """Interleave all gate_up_proj parameters in-place for SonicMoE."""
    from axolotl.integrations.kernels.libs.sonicmoe.weight_converter import (
        interleave_gate_up,
    )

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "gate_up_proj" in name:
                param.copy_(interleave_gate_up(param))


def _unpatch_sonicmoe():
    """Restore original forward on the MoE block class if it was patched."""
    from axolotl.integrations.kernels.constants import resolve_moe_block_classes

    for moe_cls in resolve_moe_block_classes("qwen3_moe"):
        if hasattr(moe_cls, "_original_forward"):
            moe_cls.forward = moe_cls._original_forward
            del moe_cls._original_forward


class TestSonicMoEForwardCorrectness:
    """Verify SonicMoE-patched model produces same output as original."""

    def teardown_method(self):
        _unpatch_sonicmoe()

    def test_forward_output_matches(self):
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

        # Original model
        model_orig = AutoModelForCausalLM.from_config(config).cuda().bfloat16()

        with torch.no_grad():
            out_orig = model_orig(input_ids)

        # Patched model (same weights, interleaved for SonicMoE)
        model_patched = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        model_patched.load_state_dict(model_orig.state_dict())

        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model_patched)

        with torch.no_grad():
            out_patched = model_patched(input_ids)

        max_diff = (out_orig.logits - out_patched.logits).abs().max().item()
        assert torch.allclose(
            out_orig.logits, out_patched.logits, atol=1e-1, rtol=1e-1
        ), f"Output mismatch: max diff={max_diff:.6f}"


class TestSonicMoEGradientCorrectness:
    """Compare gradients between original HuggingFace and SonicMoE-patched forward."""

    def teardown_method(self):
        _unpatch_sonicmoe()

    def test_gradients_match(self):
        """Verify all parameter gradients match between original and patched."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe
        from axolotl.integrations.kernels.libs.sonicmoe.weight_converter import (
            deinterleave_gate_up,
        )

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

        # ---------- Original model ----------
        model_orig = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        out_orig = model_orig(input_ids, labels=input_ids)
        out_orig.loss.backward()
        grads_orig = {
            n: p.grad.float().clone()
            for n, p in model_orig.named_parameters()
            if p.grad is not None
        }
        loss_orig = out_orig.loss.item()

        # ---------- SonicMoE-patched model (same weights, interleaved) ----------
        model_patched = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        model_patched.load_state_dict(model_orig.state_dict())

        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model_patched)

        out_patched = model_patched(input_ids, labels=input_ids)
        out_patched.loss.backward()
        grads_patched = {}
        for n, p in model_patched.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.float().clone()
            # gate_up_proj grads are in interleaved layout, de-interleave to match orig
            if "gate_up_proj" in n:
                g = deinterleave_gate_up(g)
            grads_patched[n] = g
        loss_patched = out_patched.loss.item()

        # ---------- Compare ----------
        assert abs(loss_orig - loss_patched) < 0.5, (
            f"Loss mismatch: orig={loss_orig:.4f}, patched={loss_patched:.4f}"
        )

        # All parameters with gradients in original should have them in patched
        missing = set(grads_orig.keys()) - set(grads_patched.keys())
        assert not missing, f"Missing gradients in patched model: {missing}"

        # Compare gradient values
        # bf16 with different GEMM impls (cuBLAS vs CUTLASS) can diverge,
        # so use generous tolerance: flag only if both rel >10% AND abs >1e-2
        mismatches = []
        for name in grads_orig:
            if name not in grads_patched:
                continue
            g_orig = grads_orig[name]
            g_patched = grads_patched[name]
            max_diff = (g_orig - g_patched).abs().max().item()
            rel_diff = max_diff / (g_orig.abs().max().item() + 1e-8)

            if rel_diff > 0.1 and max_diff > 1e-2:
                mismatches.append(
                    f"  {name}: max_abs_diff={max_diff:.6f}, rel_diff={rel_diff:.4f}"
                )

        assert not mismatches, (
            "Gradient mismatches (rel_diff > 10% and abs_diff > 1e-2):\n"
            + "\n".join(mismatches)
        )

    def test_router_weights_receive_gradients(self):
        """Verify that router (gate) weights get non-zero gradients."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

        model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model)

        out = model(input_ids, labels=input_ids)
        out.loss.backward()

        gate_grads_found = False
        for name, param in model.named_parameters():
            if "gate" in name and "weight" in name:
                gate_grads_found = True
                assert param.grad is not None, f"No gradient for router: {name}"
                assert param.grad.abs().max() > 0, f"Zero gradient for router: {name}"

        assert gate_grads_found, "No gate.weight parameters found in model"


class TestSonicMoETrainingConvergence:
    """Verify loss decreases during training with SonicMoE."""

    def teardown_method(self):
        _unpatch_sonicmoe()

    def test_loss_decreases(self):
        """Run 30 training steps, verify loss decreases and no NaN/Inf."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")

        model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []

        for step in range(30):
            out = model(input_ids, labels=input_ids)
            loss = out.loss
            assert not math.isnan(loss.item()), f"NaN loss at step {step}"
            assert not math.isinf(loss.item()), f"Inf loss at step {step}"
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_expert_weights_update(self):
        """Verify expert weights change during training (not frozen)."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")

        model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model)

        # Snapshot expert weights before training
        expert_weights_before = {}
        for name, param in model.named_parameters():
            if "experts" in name:
                expert_weights_before[name] = param.data.clone()

        assert expert_weights_before, "No expert parameters found"

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(5):
            out = model(input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Check that expert weights changed
        changed = 0
        for name, param in model.named_parameters():
            if name in expert_weights_before:
                if not torch.equal(param.data, expert_weights_before[name]):
                    changed += 1

        assert changed > 0, "No expert weights changed after 5 training steps"
