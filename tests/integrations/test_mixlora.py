# Copyright 2025 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
import sys
import types

import pytest
import safetensors.torch
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from axolotl.integrations.base import PluginManager
from axolotl.integrations.mixlora import MixLoraPlugin
from axolotl.integrations.mixlora.loss import collect_mixlora_aux_loss
from axolotl.integrations.mixlora.constants import MIXLORA_WEIGHTS_NAME
from axolotl.integrations.mixlora.model import (
    MixLoraFFN,
    load_mixlora_state_dict,
    mixlora_state_dict,
)
from axolotl.integrations.mixlora.patching import patch_model_with_mixlora
from axolotl.utils.dict import DictDefault


class TestMixLora:
    """Test suite for MixLoRA components and patching."""

    @pytest.fixture
    def mock_cfg(self):
        return DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "adapter": "mixlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "mixlora_num_experts": 4,
                "mixlora_top_k": 2,
                "mixlora_router_aux_loss_coef": 0.01,
            }
        )

    @pytest.fixture
    def mock_swiglu_ffn(self):
        """Creates a mock SwiGLU FFN."""
        hidden_size = 128
        intermediate_size = 512

        class MockSwiGLUFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return MockSwiGLUFFN()

    def test_mixlora_ffn_forward(self, mock_swiglu_ffn, mock_cfg):
        """Test the forward pass of a single MixLoraFFN block."""
        hidden_size = 128
        seq_len = 10
        batch_size = 2

        mixlora_ffn = MixLoraFFN(
            original_ffn=mock_swiglu_ffn,
            num_experts=mock_cfg.mixlora_num_experts,
            top_k=mock_cfg.mixlora_top_k,
            lora_r=mock_cfg.lora_r,
            lora_alpha=mock_cfg.lora_alpha,
            lora_dropout=mock_cfg.lora_dropout,
        )

        # Verify freezing of base FFN
        for param in mixlora_ffn.base_ffn.parameters():
            assert not param.requires_grad

        # Verify trainable params (router + experts)
        for param in mixlora_ffn.router.parameters():
            assert param.requires_grad
        for param in mixlora_ffn.experts.parameters():
            assert param.requires_grad

        # Forward pass
        x = torch.randn(batch_size, seq_len, hidden_size)
        out = mixlora_ffn(x)

        # Output shape should match input shape
        assert out.shape == x.shape

        # Aux loss should be populated
        assert mixlora_ffn.aux_loss is not None
        assert mixlora_ffn.aux_loss.item() > 0

    def test_mixlora_ffn_training_step_updates_router(self, mock_swiglu_ffn, mock_cfg):
        """Smoke test that gradients flow through MixLoRA router/expert params."""
        mixlora_ffn = MixLoraFFN(
            original_ffn=mock_swiglu_ffn,
            num_experts=mock_cfg.mixlora_num_experts,
            top_k=mock_cfg.mixlora_top_k,
            lora_r=mock_cfg.lora_r,
            lora_alpha=mock_cfg.lora_alpha,
            lora_dropout=mock_cfg.lora_dropout,
        )

        optimizer = torch.optim.AdamW(
            [p for p in mixlora_ffn.parameters() if p.requires_grad], lr=1e-3
        )

        router_before = mixlora_ffn.router.gate.weight.detach().clone()
        x = torch.randn(2, 10, 128)

        optimizer.zero_grad()
        out = mixlora_ffn(x)
        loss = out.pow(2).mean() + collect_mixlora_aux_loss(
            mixlora_ffn,
            router_aux_loss_coef=mock_cfg.mixlora_router_aux_loss_coef,
        )
        loss.backward()
        optimizer.step()

        assert mixlora_ffn.router.gate.weight.grad is not None
        assert not torch.allclose(router_before, mixlora_ffn.router.gate.weight)

    def test_mixlora_state_dict_roundtrip(self, mock_swiglu_ffn, mock_cfg, tmp_path):
        """Checkpoint roundtrip for MixLoRA-only router/expert weights."""
        source = MixLoraFFN(
            original_ffn=mock_swiglu_ffn,
            num_experts=mock_cfg.mixlora_num_experts,
            top_k=mock_cfg.mixlora_top_k,
            lora_r=mock_cfg.lora_r,
            lora_alpha=mock_cfg.lora_alpha,
            lora_dropout=mock_cfg.lora_dropout,
        )
        target = MixLoraFFN(
            original_ffn=mock_swiglu_ffn,
            num_experts=mock_cfg.mixlora_num_experts,
            top_k=mock_cfg.mixlora_top_k,
            lora_r=mock_cfg.lora_r,
            lora_alpha=mock_cfg.lora_alpha,
            lora_dropout=mock_cfg.lora_dropout,
        )

        with torch.no_grad():
            source.router.gate.weight.add_(0.123)

        state = mixlora_state_dict(source)
        ckpt_path = tmp_path / MIXLORA_WEIGHTS_NAME
        safetensors.torch.save_file(state, str(ckpt_path), metadata={"format": "pt"})

        loaded_state = safetensors.torch.load_file(str(ckpt_path))
        load_mixlora_state_dict(target, loaded_state, strict=True)

        assert torch.allclose(source.router.gate.weight, target.router.gate.weight)

    def test_mixlora_plugin_registers_trainer(self, mock_cfg, monkeypatch):
        """Test that the MixLoRA plugin wires up the integration trainer."""
        fake_trainer_module = types.ModuleType("axolotl.integrations.mixlora.trainer")

        class DummyMixLoraTrainer:
            pass

        fake_trainer_module.MixLoraTrainer = DummyMixLoraTrainer
        monkeypatch.setitem(
            sys.modules, "axolotl.integrations.mixlora.trainer", fake_trainer_module
        )

        plugin_manager = PluginManager.get_instance()
        original_plugins = plugin_manager.plugins

        try:
            plugin_manager.plugins = OrderedDict()
            plugin_manager.register("axolotl.integrations.mixlora.MixLoraPlugin")

            trainer_cls = plugin_manager.get_trainer_cls(mock_cfg)

            assert trainer_cls is DummyMixLoraTrainer
            assert isinstance(
                plugin_manager.plugins["axolotl.integrations.mixlora.MixLoraPlugin"],
                MixLoraPlugin,
            )
        finally:
            plugin_manager.plugins = original_plugins

    @pytest.mark.slow
    def test_patch_model_with_mixlora(self, mock_cfg):
        """Test patching a full model architecture (requires causal LM model)."""
        model = AutoModelForCausalLM.from_pretrained(mock_cfg.base_model)

        # Patch the model
        patched_model = patch_model_with_mixlora(model, mock_cfg)

        # Verify FFN layers are replaced
        has_mixlora_block = False
        for module in patched_model.modules():
            if isinstance(module, MixLoraFFN):
                has_mixlora_block = True
                break

        assert has_mixlora_block

        # Forward pass on small input
        x = torch.randint(0, 1000, (1, 10))
        outputs = patched_model(x)
        assert outputs.logits.size(0) == 1
        assert outputs.logits.size(1) == 10

        # Test loss collection
        aux_loss = collect_mixlora_aux_loss(
            patched_model,
            router_aux_loss_coef=mock_cfg.mixlora_router_aux_loss_coef
        )
        assert aux_loss.item() > 0
