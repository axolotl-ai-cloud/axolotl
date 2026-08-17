"""End-to-end gradient and convergence tests for SonicMoE integration.

Flow:

    register_sonicmoe_experts()                # plug into ALL_EXPERTS_FUNCTIONS
    config._experts_implementation = "sonicmoe"
    model = AutoModelForCausalLM.from_config(config)   # transformers dispatches

No weight interleaving needed (``concat_layout=True``).

Requires:
    - Hopper (sm_90) or Blackwell (sm_100+) GPU
    - sonic-moe >= 0.1.2 installed from source
    - transformers >= 5.8 with Qwen3MoE Experts class
"""

import importlib.util
import math

import pytest
import torch


def _is_hopper_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU"),
    pytest.mark.skipif(
        not _is_hopper_or_newer(),
        reason="SonicMoE requires Hopper (sm_90) or Blackwell (sm_100+)",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("kernels") is None,
        reason="HF `kernels` package not installed",
    ),
]


def _create_tiny_qwen3_config(experts_implementation: str):
    """Create a minimal Qwen3MoE config bound to the requested experts impl."""
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
    config._experts_implementation = experts_implementation
    return config


def _build_model(experts_implementation: str):
    from transformers import AutoModelForCausalLM

    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        register_sonicmoe_experts,
    )

    register_sonicmoe_experts()
    config = _create_tiny_qwen3_config(experts_implementation)
    return AutoModelForCausalLM.from_config(config).cuda().bfloat16(), config


class TestSonicMoEForwardCorrectness:
    """SonicMoE-dispatched model produces output close to eager baseline."""

    def test_forward_output_matches_eager(self):
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda")

        eager_model, _ = _build_model("eager")
        with torch.no_grad():
            out_eager = eager_model(input_ids).logits

        sonic_model, _ = _build_model("sonicmoe")
        sonic_model.load_state_dict(eager_model.state_dict())

        with torch.no_grad():
            out_sonic = sonic_model(input_ids).logits

        max_diff = (out_eager - out_sonic).abs().max().item()
        assert torch.allclose(out_eager, out_sonic, atol=1e-1, rtol=1e-1), (
            f"Output mismatch: max diff={max_diff:.6f}"
        )


class TestSonicMoEGradientCorrectness:
    """Compare gradients between eager and SonicMoE-dispatched forward."""

    def test_gradients_match(self):
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda")

        eager_model, _ = _build_model("eager")
        out_eager = eager_model(input_ids, labels=input_ids)
        out_eager.loss.backward()
        grads_eager = {
            n: p.grad.float().clone()
            for n, p in eager_model.named_parameters()
            if p.grad is not None
        }
        loss_eager = out_eager.loss.item()

        sonic_model, _ = _build_model("sonicmoe")
        sonic_model.load_state_dict(eager_model.state_dict())
        out_sonic = sonic_model(input_ids, labels=input_ids)
        out_sonic.loss.backward()
        grads_sonic = {
            n: p.grad.float().clone()
            for n, p in sonic_model.named_parameters()
            if p.grad is not None
        }
        loss_sonic = out_sonic.loss.item()

        assert abs(loss_eager - loss_sonic) < 0.5, (
            f"Loss mismatch: eager={loss_eager:.4f}, sonic={loss_sonic:.4f}"
        )

        missing = set(grads_eager.keys()) - set(grads_sonic.keys())
        assert not missing, f"Missing gradients in sonicmoe model: {missing}"

        # bf16 + different GEMM backends can diverge; tolerate both rel >10% AND
        # abs >1e-2 together.
        mismatches = []
        for name, g_eager in grads_eager.items():
            g_sonic = grads_sonic[name]
            max_diff = (g_eager - g_sonic).abs().max().item()
            rel_diff = max_diff / (g_eager.abs().max().item() + 1e-8)
            if rel_diff > 0.1 and max_diff > 1e-2:
                mismatches.append(
                    f"  {name}: max_abs_diff={max_diff:.6f}, rel_diff={rel_diff:.4f}"
                )

        assert not mismatches, (
            "Gradient mismatches (rel_diff > 10% and abs_diff > 1e-2):\n"
            + "\n".join(mismatches)
        )

    def test_router_weights_receive_gradients(self):
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda")
        model, _ = _build_model("sonicmoe")
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

    def test_loss_decreases(self):
        input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
        model, _ = _build_model("sonicmoe")

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
        input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
        model, _ = _build_model("sonicmoe")

        expert_weights_before = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if "experts" in name
        }
        assert expert_weights_before, "No expert parameters found"

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(5):
            out = model(input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        changed = sum(
            1
            for name, param in model.named_parameters()
            if name in expert_weights_before
            and not torch.equal(param.data, expert_weights_before[name])
        )
        assert changed > 0, "No expert weights changed after 5 training steps"
