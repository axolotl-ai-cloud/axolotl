# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""End-to-end tests for SonicMoE + LoRA.

Flow:

    register_sonicmoe_experts()                # plug into ALL_EXPERTS_FUNCTIONS
    config._experts_implementation = "sonicmoe"
    model = AutoModelForCausalLM.from_config(config)
    model = get_peft_model(model, lora_config)   # PEFT wraps params/modules

``sonicmoe_experts_forward_with_lora`` detects the PEFT wrappers and
materializes ``W_eff = W + scaling * (B @ A)`` via :class:`MoELoRAMaterialize`,
so adapters train through the CUTLASS kernels.

Requires:
    - Hopper (sm_90) or Blackwell (sm_100+) GPU
    - sonic-moe >= 0.1.2 installed from source
    - peft installed
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
    pytest.mark.skipif(
        importlib.util.find_spec("peft") is None, reason="PEFT not installed"
    ),
]


def _create_tiny_qwen3_config():
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
    config._experts_implementation = "sonicmoe"
    return config


def _build_sonic_model():
    from transformers import AutoModelForCausalLM

    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        register_sonicmoe_experts,
    )

    register_sonicmoe_experts()
    config = _create_tiny_qwen3_config()
    return AutoModelForCausalLM.from_config(config).cuda().bfloat16()


def _apply_lora(model, target_modules):
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
    """SonicMoE + LoRA on expert projections trains end-to-end."""

    def test_loss_decreases(self):
        input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
        model = _build_sonic_model()
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
        input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
        model = _build_sonic_model()
        model = _apply_lora(model, ["gate_up_proj", "down_proj"])

        frozen_before = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if not param.requires_grad
        }

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        for _ in range(5):
            out = model(input_ids, labels=input_ids)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for name, before in frozen_before.items():
            after = dict(model.named_parameters())[name]
            assert torch.equal(after.data, before), f"Frozen weight changed: {name}"

    def test_lora_adapters_receive_gradients(self):
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda")
        model = _build_sonic_model()
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
        input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
        model = _build_sonic_model()
        model = _apply_lora(model, ["gate_up_proj", "down_proj"])

        lora_before = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if "lora_" in name and param.requires_grad
        }
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
    """LoRA only on the router (gate) — expert path takes the no-LoRA fast path."""

    def test_gate_only_lora_loss_decreases(self):
        input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
        model = _build_sonic_model()
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
    """Full fine-tuning (no PEFT) still works through the registered forward."""

    def test_no_lora_loss_decreases(self):
        input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
        model = _build_sonic_model()

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
