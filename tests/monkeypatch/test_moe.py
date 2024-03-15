import torch
import pytest
from torch import nn
from torch.nn import functional as F
from axolotl.monkeypatch.moe.mlp import FusedExperts
from axolotl.monkeypatch.moe.moe import SparseMoeBlock

from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralConfig

def test_fused_mixtral_moe():
    # NOTE: Requires torch 2.2.0
    # Set random seeds for reproducibility
    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    # Define the configuration for the MixtralSparseMoeBlock
    config = MixtralConfig(
        hidden_size=128,
        intermediate_size=512,
        num_local_experts=8,
        num_experts_per_tok=2,
    )

    # Initialize the MixtralSparseMoeBlock and SparseMoeBlock with the same configuration
    mixtral_moe = MixtralSparseMoeBlock(config)
    sparse_moe = SparseMoeBlock(
        experts=mixtral_moe.experts,
        gate=mixtral_moe.gate,
        hidden_dim=config.hidden_size,
        ffn_dim=config.intermediate_size,
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok
    )

    assert torch.cat([
        mixtral_moe.experts[0].w1.weight.data,
        mixtral_moe.experts[0].w3.weight.data], dim=0
    ).equal(sparse_moe.experts.experts.weight[0])

    # Generate random input data
    batch_size = 16
    sequence_length = 32
    input_data = torch.randn(batch_size, sequence_length, config.hidden_size)

    # Run the forward pass with gradients for both models
    with torch.no_grad():
        mixtral_output, mixtral_router_logits = mixtral_moe(input_data)
        sparse_output, sparse_router_logits = sparse_moe(input_data)

    # Compute the difference between the outputs
    output_diff = torch.abs(mixtral_output - sparse_output).mean().item()
    router_diff = torch.abs(mixtral_router_logits - sparse_router_logits).mean().item()

    # Define the tolerance for the difference
    tolerance = 0.05

    # # Check if the difference is within the tolerance
    assert output_diff < 0.05, f"Output difference is {output_diff}, which is greater than the tolerance of {tolerance}"
    assert router_diff == 0, f"Output difference is {output_diff}, which is greater than the tolerance of {tolerance}"
