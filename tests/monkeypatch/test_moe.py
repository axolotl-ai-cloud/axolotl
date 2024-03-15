import torch
from copy import deepcopy
from axolotl.monkeypatch.moe.mlp import FusedExperts
from axolotl.monkeypatch.moe.moe import SparseMoeBlock
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralConfig

def test_fused_mixtral_moe():
    # NOTE: Requires torch 2.2.0
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cuda")

    # Define the configuration for the MixtralSparseMoeBlock
    config = MixtralConfig(
        hidden_size=128,
        intermediate_size=512,
        num_local_experts=8,
        num_experts_per_tok=2,
    )

    # Initialize the MixtralSparseMoeBlock and SparseMoeBlock with the same configuration
    mixtral_moe = MixtralSparseMoeBlock(config)
    mixtral_moe_copy = deepcopy(mixtral_moe)

    experts = FusedExperts(
        experts=mixtral_moe_copy.experts,
        input_size=mixtral_moe_copy.ffn_dim,
        hidden_size=mixtral_moe_copy.hidden_dim,
        num_experts=mixtral_moe_copy.num_experts,
        top_k=mixtral_moe_copy.top_k,
        activation=mixtral_moe_copy.experts[0].act_fn
    )
    sparse_moe = SparseMoeBlock(
        experts,
        hidden_dim=config.hidden_size,
        ffn_dim=config.intermediate_size,
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok
    )

    # Generate random input data
    batch_size = 16
    sequence_length = 32
    input_data = torch.randn(batch_size, sequence_length, config.hidden_size)

    # Run the forward pass with gradients for both models
    with torch.no_grad():
        mixtral_output, _ = mixtral_moe(input_data)
        sparse_output, _ = sparse_moe(input_data)

    # Compute the difference between the outputs and router logits
    output_diff = torch.abs(mixtral_output - sparse_output).mean().item()

    # Define the tolerance for the difference
    tolerance = 0.1

    # # Check if the difference is within the tolerance
    assert output_diff < tolerance, f"Output difference is {output_diff}, which is greater than the tolerance of {tolerance}"