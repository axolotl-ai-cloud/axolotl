"""
Tests for optimized MoE kernels
"""

import pytest
import torch

from axolotl.integrations.moe_optimized.plugin import (
    MoeOptimizedPlugin,
    sort_tokens_by_expert,
)
from axolotl.kernels.moe import (
    cg_grouped_gemm_forward,
)


class TestMoeKernels:
    """Test the optimized MoE kernel functions"""

    @pytest.fixture
    def setup_tensors(self):
        """Setup test tensors"""
        batch_size = 32
        hidden_dim = 256
        intermediate_dim = 512
        num_experts = 8
        top_k = 2

        # Create test tensors
        inputs = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device="cuda")
        expert_weights = torch.randn(
            num_experts,
            intermediate_dim,
            hidden_dim,
            dtype=torch.float16,
            device="cuda",
        )
        expert_indices = torch.randint(
            0, num_experts, (batch_size,), dtype=torch.long, device="cuda"
        )

        return {
            "inputs": inputs,
            "expert_weights": expert_weights,
            "expert_indices": expert_indices,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "num_experts": num_experts,
            "top_k": top_k,
        }

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cg_grouped_gemm_forward(self, setup_tensors):
        """Test forward pass of grouped GEMM"""
        data = setup_tensors

        # Run forward pass
        output = cg_grouped_gemm_forward(
            data["inputs"],
            data["expert_weights"],
            data["expert_indices"],
            group_size_m=8,
        )

        # Check output shape
        assert output.shape == (data["batch_size"], data["intermediate_dim"])
        assert output.dtype == data["inputs"].dtype

        # Check that output is not all zeros
        assert torch.abs(output).sum() > 0

    def test_sort_tokens_by_expert(self):
        """Test token sorting function"""
        batch_size = 16
        hidden_dim = 128
        num_experts = 4
        top_k = 2

        # Create test data
        hidden_states = torch.randn(batch_size, hidden_dim)
        expert_indices = torch.randint(0, num_experts, (batch_size, top_k))

        # Sort tokens
        sorted_states, sort_idx, inverse_idx, flat_experts = sort_tokens_by_expert(
            hidden_states, expert_indices, top_k
        )

        # Check shapes
        assert sorted_states.shape[0] == batch_size * top_k
        assert sorted_states.shape[1] == hidden_dim
        assert len(sort_idx) == batch_size * top_k
        assert len(inverse_idx) == batch_size * top_k
        assert len(flat_experts) == batch_size * top_k

        # Check that we can restore original order
        restored = sorted_states[inverse_idx]
        assert restored.shape[0] == batch_size * top_k


class TestMoeOptimizedPlugin:
    """Test the MoE optimization plugin"""

    def test_plugin_initialization(self):
        """Test plugin initialization"""
        plugin = MoeOptimizedPlugin()

        assert plugin.enabled is False
        assert plugin.group_size_m == 128
        assert plugin.models is None

    def test_plugin_registration(self):
        """Test plugin registration with config"""
        plugin = MoeOptimizedPlugin()

        cfg = {
            "moe_optimized": True,
            "moe_group_size": 64,
            "moe_kernel_models": ["mixtral", "qwen3_moe"],
        }

        plugin.register(cfg)

        assert plugin.enabled is True
        assert plugin.group_size_m == 64
        assert plugin.models == ["mixtral", "qwen3_moe"]

    def test_get_input_args(self):
        """Test input args specification"""
        plugin = MoeOptimizedPlugin()
        args_path = plugin.get_input_args()

        assert args_path == "axolotl.integrations.moe_optimized.args.MoeOptimizedArgs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
