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

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from axolotl.integrations.mixlora.loss import collect_mixlora_aux_loss
from axolotl.integrations.mixlora.model import MixLoraFFN
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
