# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
End-to-end tests for SonicMoE + LoRA integration.

Verifies that PEFT-wrapped MoE models work correctly with SonicMoE's
runtime LoRA materialization: gradients flow to adapters, base weights
stay frozen, and loss converges.

Requires:
    - H100/H200 GPU (SonicMoE CUTLASS kernels target sm_90)
    - sonicmoe package installed
    - peft package installed
    - transformers with Qwen3MoE support

Usage:
    pytest tests/e2e/integrations/test_sonicmoe_lora.py -v -s
"""

import importlib.util
import math

import pytest
import torch

_sonicmoe_available = importlib.util.find_spec("sonicmoe") is not None
_peft_available = importlib.util.find_spec("peft") is not None
_is_hopper = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU"),
    pytest.mark.skipif(
        not _is_hopper, reason="SonicMoE CUTLASS kernels require Hopper (sm_90)"
    ),
    pytest.mark.skipif(not _sonicmoe_available, reason="SonicMoE not installed"),
    pytest.mark.skipif(not _peft_available, reason="PEFT not installed"),
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


def _apply_lora(model, target_modules):
    """Apply PEFT LoRA to the model."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, lora_config)


class TestSonicMoELoRATraining:
    """Verify SonicMoE + LoRA training works end-to-end."""

    def teardown_method(self):
        _unpatch_sonicmoe()

    def test_loss_decreases(self):
        """Run 30 training steps with LoRA on experts, verify loss decreases."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")

        model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model)
        model = _apply_lora(model, ["gate_up_proj", "down_proj"])

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
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

    def test_base_weights_frozen(self):
        """Verify base (non-LoRA) weights don't change during training."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")

        model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model)
        model = _apply_lora(model, ["gate_up_proj", "down_proj"])

        # Snapshot frozen weights
        frozen_before = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen_before[name] = param.data.clone()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        for _ in range(5):
            out = model(input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for name, param in model.named_parameters():
            if name in frozen_before:
                assert torch.equal(param.data, frozen_before[name]), (
                    f"Frozen weight changed: {name}"
                )

    def test_lora_adapters_receive_gradients(self):
        """Verify LoRA A and B matrices get non-zero gradients."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")

        model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model)
        model = _apply_lora(model, ["gate_up_proj", "down_proj"])

        out = model(input_ids, labels=input_ids)
        out.loss.backward()

        lora_grads_found = 0
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                assert param.grad is not None, f"No gradient for LoRA param: {name}"
                assert param.grad.abs().max() > 0, (
                    f"Zero gradient for LoRA param: {name}"
                )
                lora_grads_found += 1

        assert lora_grads_found > 0, "No LoRA parameters found with gradients"

    def test_lora_adapters_update(self):
        """Verify LoRA adapter weights change during training."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")

        model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model)
        model = _apply_lora(model, ["gate_up_proj", "down_proj"])

        # Snapshot LoRA weights
        lora_before = {}
        for name, param in model.named_parameters():
            if "lora_" in name and param.requires_grad:
                lora_before[name] = param.data.clone()

        assert lora_before, "No LoRA parameters found"

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        for _ in range(5):
            out = model(input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        changed = sum(
            1
            for name, param in model.named_parameters()
            if name in lora_before and not torch.equal(param.data, lora_before[name])
        )
        assert changed > 0, "No LoRA weights changed after 5 training steps"


class TestSonicMoEGateOnlyLoRA:
    """Verify LoRA targeting only the gate (router) works with SonicMoE."""

    def teardown_method(self):
        _unpatch_sonicmoe()

    def test_gate_only_lora_loss_decreases(self):
        """LoRA only on gate — expert path should have zero materialization overhead."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")

        model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model)
        # Only target the gate (router), not expert projections
        model = _apply_lora(model, ["gate"])

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        losses = []

        for step in range(20):
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


class TestSonicMoENoLoRARegression:
    """Verify SonicMoE without LoRA still works after LoRA code was added."""

    def teardown_method(self):
        _unpatch_sonicmoe()

    def test_no_lora_loss_decreases(self):
        """Full fine-tuning (no PEFT) with SonicMoE — regression test."""
        from transformers import AutoModelForCausalLM

        from axolotl.integrations.kernels.libs.sonicmoe.patch import patch_sonicmoe

        config = _create_tiny_qwen3_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 32), device="cuda")

        model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
        patch_sonicmoe("qwen3_moe")
        _interleave_gate_up_weights(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []

        for step in range(20):
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
