"""Validation tests for Gemma 4 MoE compatibility with ScatterMoE and SonicMoE.

Gemma 4 has a unique MoE architecture:
- No separate SparseMoeBlock — MoE is embedded in the decoder layer
- Hybrid MLP+MoE: dense MLP runs in parallel with sparse MoE, outputs summed
- Custom router (Gemma4TextRouter): RMSNorm → scale → proj → softmax → topk → renorm → per_expert_scale
- Router is `self.router` (not `self.gate`)
- Experts use standard 3D param layout with @use_experts_implementation

These tests validate that:
1. ScatterMoE kernels produce correct output for Gemma4 expert layout
2. ScatterMoE + LoRA produces correct output
3. SonicMoE integration code handles Gemma4 routing correctly
4. Weight layouts are compatible
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# ============================================================================
# Gemma4 reference implementation (extracted from transformers)
# ============================================================================


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.with_scale:
            return (self.weight * x).to(x.dtype)
        return x.to(x.dtype)


class Gemma4TextRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.scalar_root_size = hidden_size**-0.5
        self.eps = eps

        self.norm = Gemma4RMSNorm(hidden_size, eps=eps, with_scale=False)
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states.to(self.proj.weight.dtype))
        router_probabilities = F.softmax(expert_scores, dim=-1)

        top_k_weights, top_k_index = torch.topk(
            router_probabilities, k=self.top_k, dim=-1
        )

        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]

        return router_probabilities, top_k_weights, top_k_index


class Gemma4TextExperts(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, act_fn):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(
                2, dim=-1
            )
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def gemma4_config():
    """Small Gemma4 MoE config for testing."""
    return {
        "hidden_size": 128,
        "num_experts": 8,
        "top_k": 2,
        "intermediate_size": 64,
        "eps": 1e-6,
    }


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


@pytest.fixture
def gemma4_moe_layer(gemma4_config, device):
    """Create a Gemma4 MoE layer (router + experts) on GPU."""
    from transformers.activations import ACT2FN

    act_fn = ACT2FN["gelu_pytorch_tanh"]

    router = Gemma4TextRouter(
        hidden_size=gemma4_config["hidden_size"],
        num_experts=gemma4_config["num_experts"],
        top_k=gemma4_config["top_k"],
        eps=gemma4_config["eps"],
    )
    experts = Gemma4TextExperts(
        num_experts=gemma4_config["num_experts"],
        hidden_size=gemma4_config["hidden_size"],
        intermediate_size=gemma4_config["intermediate_size"],
        act_fn=act_fn,
    )

    # Initialize weights
    nn.init.kaiming_uniform_(experts.gate_up_proj)
    nn.init.kaiming_uniform_(experts.down_proj)
    nn.init.normal_(router.proj.weight, std=0.01)

    router = router.to(device).to(torch.bfloat16)
    experts = experts.to(device).to(torch.bfloat16)

    return router, experts


# ============================================================================
# ScatterMoE Tests
# ============================================================================


class TestGemma4ScatterMoE:
    """Test ScatterMoE kernel compatibility with Gemma4 expert layout."""

    def test_scattermoe_experts_match_reference(
        self, gemma4_moe_layer, gemma4_config, device
    ):
        """ScatterMoE kernel output matches reference expert computation."""
        from transformers.activations import ACT2FN

        from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
            flatten_sort_count,
            parallel_linear,
        )

        router, experts = gemma4_moe_layer
        act_fn = ACT2FN["gelu_pytorch_tanh"]
        T = 16  # num tokens
        H = gemma4_config["hidden_size"]
        K = gemma4_config["top_k"]
        E = gemma4_config["num_experts"]

        hidden_states = torch.randn(T, H, device=device, dtype=torch.bfloat16)

        # Reference forward
        _, top_k_weights, top_k_index = router(hidden_states)
        ref_output = experts(hidden_states, top_k_index, top_k_weights)

        # ScatterMoE forward
        routing_weights = top_k_weights.to(hidden_states.dtype)
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
            top_k_index, num_experts=E
        )

        # gate_up_proj is [E, 2*inter, H], ScatterMoE expects transposed: [E, H, 2*I]
        gate_up_weight = experts.gate_up_proj.transpose(2, 1)
        down_weight = experts.down_proj.transpose(2, 1)

        gates_h = parallel_linear(
            hidden_states,
            gate_up_weight,
            K,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False,
            grouped_out=True,
        )
        gates, h = gates_h.chunk(2, dim=-1)
        h = act_fn(gates) * h

        scatter_output = parallel_linear(
            h,
            down_weight,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            grouped_out=False,
            gates=routing_weights,
        )

        # Allow bf16 tolerance
        torch.testing.assert_close(scatter_output, ref_output, atol=1e-2, rtol=1e-2)

    def test_scattermoe_with_lora(self, gemma4_moe_layer, gemma4_config, device):
        """ScatterMoE + LoRA kernel matches reference LoRA computation."""
        from transformers.activations import ACT2FN

        from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
            flatten_sort_count,
            parallel_linear,
        )
        from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (
            parallel_linear_lora,
        )

        router, experts = gemma4_moe_layer
        act_fn = ACT2FN["gelu_pytorch_tanh"]
        T = 16
        H = gemma4_config["hidden_size"]
        K = gemma4_config["top_k"]
        E = gemma4_config["num_experts"]
        inter = gemma4_config["intermediate_size"]
        rank = 4
        scaling = 0.5

        hidden_states = torch.randn(T, H, device=device, dtype=torch.bfloat16)

        # Create LoRA weights for gate_up_proj
        # ScatterMoE layout: A=[r*E, K], B=[N, r*E]
        lora_A_gup = (
            torch.randn(rank * E, H, device=device, dtype=torch.bfloat16) * 0.01
        )
        lora_B_gup = (
            torch.randn(2 * inter, rank * E, device=device, dtype=torch.bfloat16) * 0.01
        )

        # Reference: manual LoRA application per expert
        _, top_k_weights, top_k_index = router(hidden_states)
        ref_output = torch.zeros(T, H, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for eidx in expert_hit:
            eidx = eidx[0]
            if eidx == E:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            current_state = hidden_states[token_idx]

            # Base gate_up + LoRA delta
            base_out = F.linear(current_state, experts.gate_up_proj[eidx])
            lora_a_slice = lora_A_gup[eidx * rank : (eidx + 1) * rank, :]
            lora_b_slice = lora_B_gup[:, eidx * rank : (eidx + 1) * rank]
            lora_delta = (
                F.linear(F.linear(current_state, lora_a_slice), lora_b_slice) * scaling
            )
            combined = base_out + lora_delta

            gate, up = combined.chunk(2, dim=-1)
            h = act_fn(gate) * up
            h = F.linear(h, experts.down_proj[eidx])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            ref_output.index_add_(0, token_idx, h.to(ref_output.dtype))

        # ScatterMoE LoRA forward
        routing_weights = top_k_weights.to(hidden_states.dtype)
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
            top_k_index, num_experts=E
        )

        gate_up_weight = experts.gate_up_proj.transpose(2, 1)
        down_weight = experts.down_proj.transpose(2, 1)

        gates_h = parallel_linear_lora(
            hidden_states,
            gate_up_weight,
            K,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            lora_A_gup,
            lora_B_gup,
            scaling,
            grouped_in=False,
            grouped_out=True,
        )
        gates, h = gates_h.chunk(2, dim=-1)
        h = act_fn(gates) * h

        scatter_output = parallel_linear(
            h,
            down_weight,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            grouped_out=False,
            gates=routing_weights,
        )

        torch.testing.assert_close(scatter_output, ref_output, atol=5e-2, rtol=5e-2)

    def test_gemma4_routing_correctness(self, gemma4_moe_layer, gemma4_config, device):
        """Gemma4 custom routing (norm+scale+per_expert_scale) produces valid outputs."""
        router, _ = gemma4_moe_layer
        T = 32
        H = gemma4_config["hidden_size"]
        K = gemma4_config["top_k"]
        E = gemma4_config["num_experts"]

        hidden_states = torch.randn(T, H, device=device, dtype=torch.bfloat16)
        router_probs, top_k_weights, top_k_index = router(hidden_states)

        # Check shapes
        assert router_probs.shape == (T, E)
        assert top_k_weights.shape == (T, K)
        assert top_k_index.shape == (T, K)

        # Router probs should be valid probability distribution
        assert (router_probs >= 0).all()
        assert torch.allclose(
            router_probs.sum(dim=-1),
            torch.ones(T, device=device, dtype=router_probs.dtype),
            atol=1e-3,
        )

        # Top-k indices should be valid expert indices
        assert (top_k_index >= 0).all()
        assert (top_k_index < E).all()

        # Top-k weights should be non-negative (per_expert_scale can change sign though)
        # Just verify finite
        assert top_k_weights.isfinite().all()

    def test_scattermoe_gradients_flow(self, gemma4_moe_layer, gemma4_config, device):
        """Verify gradients flow through ScatterMoE kernels for Gemma4."""
        from transformers.activations import ACT2FN

        from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
            flatten_sort_count,
            parallel_linear,
        )

        router, experts = gemma4_moe_layer

        # Enable grad for expert weights
        experts.gate_up_proj.requires_grad_(True)
        experts.down_proj.requires_grad_(True)

        act_fn = ACT2FN["gelu_pytorch_tanh"]
        T = 16
        H = gemma4_config["hidden_size"]
        K = gemma4_config["top_k"]
        E = gemma4_config["num_experts"]

        hidden_states = torch.randn(T, H, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            _, top_k_weights, top_k_index = router(hidden_states)

        routing_weights = top_k_weights.to(hidden_states.dtype)
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
            top_k_index, num_experts=E
        )

        gate_up_weight = experts.gate_up_proj.transpose(2, 1)
        down_weight = experts.down_proj.transpose(2, 1)

        gates_h = parallel_linear(
            hidden_states,
            gate_up_weight,
            K,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False,
            grouped_out=True,
        )
        gates, h = gates_h.chunk(2, dim=-1)
        h = act_fn(gates) * h

        output = parallel_linear(
            h,
            down_weight,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            grouped_out=False,
            gates=routing_weights,
        )

        loss = output.sum()
        loss.backward()

        assert experts.gate_up_proj.grad is not None
        assert experts.down_proj.grad is not None
        assert experts.gate_up_proj.grad.isfinite().all()
        assert experts.down_proj.grad.isfinite().all()


# ============================================================================
# SonicMoE Tests
# ============================================================================


def _can_import_sonicmoe():
    try:
        from sonicmoe.enums import ActivationType  # noqa: F401

        return True
    except Exception:
        return False


class TestGemma4SonicMoE:
    """Test SonicMoE compatibility with Gemma4.

    SonicMoE requires Hopper/Blackwell GPU. Tests that need sonicmoe
    import are skipped on unsupported GPUs.
    """

    @pytest.mark.skipif(
        not _can_import_sonicmoe(),
        reason="sonicmoe requires Hopper/Blackwell GPU",
    )
    def test_gemma4_routing_function_config(self, gemma4_config):
        """Gemma4 is registered with correct routing config."""
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            get_model_moe_config,
        )

        routing_fn, activation, router_attr = get_model_moe_config("gemma4_text")

        assert router_attr == "router"
        assert routing_fn is not None
        assert routing_fn.__name__ == "gemma4_routing"

        from sonicmoe.enums import ActivationType

        assert activation == ActivationType.GEGLU

    @pytest.mark.skipif(
        not _can_import_sonicmoe(),
        reason="sonicmoe requires Hopper/Blackwell GPU",
    )
    def test_gemma4_routing_matches_reference(self, gemma4_config):
        """Routing function output matches reference Gemma4TextRouter."""
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            get_model_moe_config,
        )

        routing_fn, _, _ = get_model_moe_config("gemma4_text")
        H = gemma4_config["hidden_size"]
        E = gemma4_config["num_experts"]
        K = gemma4_config["top_k"]
        T = 16

        router = Gemma4TextRouter(H, E, K)
        nn.init.normal_(router.proj.weight, std=0.01)

        class MockGemma4MoeBlock:
            pass

        mock_block = MockGemma4MoeBlock()
        mock_block.router = router

        hidden_states = torch.randn(T, H)

        # Reference
        _ref_probs, ref_weights, ref_indices = router(hidden_states)

        # Routing function
        flat_scores, flat_token_idx, flat_expert_idx, router_logits = routing_fn(
            hidden_states, mock_block
        )

        # Check shapes
        assert flat_scores.shape == (T * K,)
        assert flat_token_idx.shape == (T * K,)
        assert flat_expert_idx.shape == (T * K,)
        assert router_logits.shape == (T, E)

        # Reconstruct per-token routing from flat output and compare
        for t in range(T):
            mask = flat_token_idx == t
            assert mask.sum() == K, f"Token {t} should have {K} entries"

            flat_experts_for_t = flat_expert_idx[mask].sort().values
            ref_experts_for_t = ref_indices[t].sort().values.to(torch.int32)
            assert torch.equal(flat_experts_for_t, ref_experts_for_t), (
                f"Token {t}: experts mismatch"
            )

        # Verify scores match reference per-token
        for t in range(T):
            mask = flat_token_idx == t
            flat_experts_t = flat_expert_idx[mask]
            flat_scores_t = flat_scores[mask]

            sort_idx = flat_experts_t.argsort()
            flat_scores_sorted = flat_scores_t[sort_idx]

            ref_sort_idx = ref_indices[t].argsort()
            ref_scores_sorted = ref_weights[t][ref_sort_idx].float()

            torch.testing.assert_close(
                flat_scores_sorted, ref_scores_sorted, atol=1e-4, rtol=1e-4
            )

    def test_gemma4_weight_layout_compatible(self, gemma4_config):
        """Verify Gemma4 expert weight layout is compatible with SonicMoE."""
        E = gemma4_config["num_experts"]
        H = gemma4_config["hidden_size"]
        inter = gemma4_config["intermediate_size"]

        gate_up_proj = torch.randn(E, 2 * inter, H)
        down_proj = torch.randn(E, H, inter)

        # SonicMoE expects [dim, dim, E] (experts last)
        gate_up_sonic = gate_up_proj.permute(1, 2, 0)
        down_sonic = down_proj.permute(1, 2, 0)

        assert gate_up_sonic.shape == (2 * inter, H, E)
        assert down_sonic.shape == (H, inter, E)

        # Verify roundtrip
        recovered_gate_up = gate_up_sonic.permute(2, 0, 1)
        assert torch.equal(gate_up_proj, recovered_gate_up)

    def test_gemma4_is_experts_only_model(self):
        """Verify gemma4_text is recognized as experts-only model."""
        from axolotl.integrations.kernels.constants import (
            is_experts_only_model,
            resolve_experts_class,
        )

        assert is_experts_only_model("gemma4_text")
        cls = resolve_experts_class("gemma4_text")
        assert cls is not None
        assert cls.__name__ == "Gemma4TextExperts"

    def test_gemma4_not_in_sparse_moe_block(self):
        """Verify gemma4_text is NOT in SPARSE_MOE_BLOCK (has no SparseMoeBlock)."""
        from axolotl.integrations.kernels.constants import SPARSE_MOE_BLOCK

        assert "gemma4_text" not in SPARSE_MOE_BLOCK


# ============================================================================
# Integration Tests (full layer with real model config)
# ============================================================================


class TestGemma4FullLayerIntegration:
    """Test with realistic Gemma4 config (26B-A4B dimensions, single layer)."""

    @pytest.fixture
    def real_config(self):
        return {
            "hidden_size": 2816,
            "num_experts": 128,
            "top_k": 8,
            "intermediate_size": 704,
            "eps": 1e-6,
        }

    def test_scattermoe_real_dimensions(self, real_config, device):
        """ScatterMoE works with real Gemma4-26B-A4B expert dimensions."""
        from transformers.activations import ACT2FN

        from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
            flatten_sort_count,
            parallel_linear,
        )

        act_fn = ACT2FN["gelu_pytorch_tanh"]
        H = real_config["hidden_size"]
        E = real_config["num_experts"]
        K = real_config["top_k"]
        inter = real_config["intermediate_size"]
        T = 32

        # Create experts on GPU
        gate_up_proj = (
            torch.randn(E, 2 * inter, H, device=device, dtype=torch.bfloat16) * 0.01
        )
        down_proj = torch.randn(E, H, inter, device=device, dtype=torch.bfloat16) * 0.01
        hidden_states = torch.randn(T, H, device=device, dtype=torch.bfloat16)

        # Simulate routing (random valid assignment)
        top_k_index = torch.stack(
            [torch.randperm(E, device=device)[:K] for _ in range(T)]
        )
        top_k_weights = torch.softmax(
            torch.randn(T, K, device=device, dtype=torch.bfloat16), dim=-1
        )

        # Reference
        ref_output = torch.zeros(T, H, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for eidx in expert_hit:
            eidx = eidx[0]
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, gate_up_proj[eidx]).chunk(2, dim=-1)
            h = act_fn(gate) * up
            h = F.linear(h, down_proj[eidx])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            ref_output.index_add_(0, token_idx, h.to(ref_output.dtype))

        # ScatterMoE
        routing_weights = top_k_weights.to(hidden_states.dtype)
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
            top_k_index, num_experts=E
        )

        gates_h = parallel_linear(
            hidden_states,
            gate_up_proj.transpose(2, 1),
            K,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False,
            grouped_out=True,
        )
        gates, h = gates_h.chunk(2, dim=-1)
        h = act_fn(gates) * h

        scatter_output = parallel_linear(
            h,
            down_proj.transpose(2, 1),
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            grouped_out=False,
            gates=routing_weights,
        )

        torch.testing.assert_close(scatter_output, ref_output, atol=5e-2, rtol=5e-2)

    def test_scattermoe_lora_real_dimensions(self, real_config, device):
        """ScatterMoE + LoRA works with real Gemma4-26B-A4B dimensions."""
        from transformers.activations import ACT2FN

        from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
            flatten_sort_count,
            parallel_linear,
        )
        from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (
            parallel_linear_lora,
        )

        act_fn = ACT2FN["gelu_pytorch_tanh"]
        H = real_config["hidden_size"]
        E = real_config["num_experts"]
        K = real_config["top_k"]
        inter = real_config["intermediate_size"]
        T = 32
        rank = 8
        scaling = 0.5

        gate_up_proj = (
            torch.randn(E, 2 * inter, H, device=device, dtype=torch.bfloat16) * 0.01
        )
        down_proj = torch.randn(E, H, inter, device=device, dtype=torch.bfloat16) * 0.01
        lora_A = torch.randn(rank * E, H, device=device, dtype=torch.bfloat16) * 0.01
        lora_B = (
            torch.randn(2 * inter, rank * E, device=device, dtype=torch.bfloat16) * 0.01
        )
        hidden_states = torch.randn(T, H, device=device, dtype=torch.bfloat16)

        # Random routing
        top_k_index = torch.stack(
            [torch.randperm(E, device=device)[:K] for _ in range(T)]
        )
        top_k_weights = torch.softmax(
            torch.randn(T, K, device=device, dtype=torch.bfloat16), dim=-1
        )

        routing_weights = top_k_weights.to(hidden_states.dtype)
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
            top_k_index, num_experts=E
        )

        # ScatterMoE + LoRA on gate_up
        gates_h = parallel_linear_lora(
            hidden_states,
            gate_up_proj.transpose(2, 1),
            K,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            lora_A,
            lora_B,
            scaling,
            grouped_in=False,
            grouped_out=True,
        )
        gates, h = gates_h.chunk(2, dim=-1)
        h = act_fn(gates) * h

        output = parallel_linear(
            h,
            down_proj.transpose(2, 1),
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            grouped_out=False,
            gates=routing_weights,
        )

        # Basic sanity: output should be finite and right shape
        assert output.shape == (T, H)
        assert output.isfinite().all()


class TestExpertsInterfaceIntegration:
    """Test the ExpertsInterface registration (the clean transformers hook)."""

    @staticmethod
    def _make_gemma4_config():
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

        return Gemma4TextConfig(
            hidden_size=128,
            num_experts=8,
            top_k_experts=2,
            moe_intermediate_size=64,
            hidden_activation="gelu_pytorch_tanh",
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            intermediate_size=256,
            enable_moe_block=True,
        )

    def test_register_scattermoe_in_experts_interface(self):
        """register_scattermoe_experts adds 'scattermoe' to the global interface."""
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

        from axolotl.integrations.kernels.libs.scattermoe_lora.gemma4_experts import (
            register_scattermoe_experts,
            scattermoe_experts_forward,
        )

        register_scattermoe_experts()

        assert "scattermoe" in ALL_EXPERTS_FUNCTIONS
        assert ALL_EXPERTS_FUNCTIONS["scattermoe"] is scattermoe_experts_forward

    def test_experts_implementation_dispatches_to_scattermoe(self, device):
        """Setting config._experts_implementation='scattermoe' dispatches correctly."""
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextExperts as HFGemma4TextExperts,
        )

        from axolotl.integrations.kernels.libs.scattermoe_lora.gemma4_experts import (
            register_scattermoe_experts,
        )

        register_scattermoe_experts()

        cfg = self._make_gemma4_config()
        cfg._experts_implementation = "scattermoe"

        with torch.device("meta"):
            hf_experts = HFGemma4TextExperts(cfg)

        hf_experts = hf_experts.to_empty(device=device)
        nn.init.kaiming_uniform_(hf_experts.gate_up_proj)
        nn.init.kaiming_uniform_(hf_experts.down_proj)
        hf_experts = hf_experts.to(torch.bfloat16)

        T, K = 16, 2
        hidden_states = torch.randn(T, 128, device=device, dtype=torch.bfloat16)
        top_k_index = torch.stack(
            [torch.randperm(8, device=device)[:K] for _ in range(T)]
        )
        top_k_weights = torch.softmax(
            torch.randn(T, K, device=device, dtype=torch.bfloat16), dim=-1
        )

        # Get reference output with eager implementation
        cfg_eager = self._make_gemma4_config()
        cfg_eager._experts_implementation = "eager"
        with torch.device("meta"):
            eager_experts = HFGemma4TextExperts(cfg_eager)
        eager_experts = eager_experts.to_empty(device=device).to(torch.bfloat16)
        # Copy weights from scattermoe experts
        eager_experts.gate_up_proj.data.copy_(hf_experts.gate_up_proj.data)
        eager_experts.down_proj.data.copy_(hf_experts.down_proj.data)

        ref_output = eager_experts(hidden_states, top_k_index, top_k_weights)

        # ScatterMoE dispatched output
        scatter_output = hf_experts(hidden_states, top_k_index, top_k_weights)

        torch.testing.assert_close(scatter_output, ref_output, atol=1e-2, rtol=1e-2)

    def test_validation_accepts_scattermoe(self):
        """get_correct_experts_implementation accepts 'scattermoe' after registration."""
        from transformers.modeling_utils import PreTrainedModel

        from axolotl.integrations.kernels.libs.scattermoe_lora.gemma4_experts import (
            register_scattermoe_experts,
        )

        register_scattermoe_experts()

        # Should not raise
        result = PreTrainedModel.get_correct_experts_implementation(None, "scattermoe")
        assert result == "scattermoe"

    def test_eager_still_works_after_registration(self, device):
        """Registering scattermoe doesn't break eager dispatch."""
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextExperts as HFGemma4TextExperts,
        )

        from axolotl.integrations.kernels.libs.scattermoe_lora.gemma4_experts import (
            register_scattermoe_experts,
        )

        register_scattermoe_experts()

        cfg = self._make_gemma4_config()
        cfg._experts_implementation = "eager"

        with torch.device("meta"):
            hf_experts = HFGemma4TextExperts(cfg)

        hf_experts = hf_experts.to_empty(device=device)
        nn.init.kaiming_uniform_(hf_experts.gate_up_proj)
        nn.init.kaiming_uniform_(hf_experts.down_proj)
        hf_experts = hf_experts.to(torch.bfloat16)

        T, K = 16, 2
        hidden_states = torch.randn(T, 128, device=device, dtype=torch.bfloat16)
        top_k_index = torch.stack(
            [torch.randperm(8, device=device)[:K] for _ in range(T)]
        )
        top_k_weights = torch.softmax(
            torch.randn(T, K, device=device, dtype=torch.bfloat16), dim=-1
        )

        # Should use eager (original) forward without error
        output = hf_experts(hidden_states, top_k_index, top_k_weights)
        assert output.shape == (T, 128)
        assert output.isfinite().all()


class TestScatterMoEExpertsInterfaceMultiModel:
    """Test that the registered scattermoe ExpertsInterface works across model types.

    All @use_experts_implementation Experts classes share the same layout:
    gate_up_proj [E, 2*inter, H], down_proj [E, H, inter], forward(hidden_states, top_k_index, top_k_weights).
    """

    MODEL_EXPERTS = [
        (
            "transformers.models.gemma4.modeling_gemma4",
            "Gemma4TextExperts",
            {
                "hidden_size": 128,
                "num_experts": 8,
                "moe_intermediate_size": 64,
                "hidden_activation": "gelu_pytorch_tanh",
                "top_k_experts": 2,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "intermediate_size": 256,
                "enable_moe_block": True,
            },
            "transformers.models.gemma4.configuration_gemma4.Gemma4TextConfig",
        ),
        (
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "Qwen3MoeExperts",
            {
                "hidden_size": 128,
                "num_experts": 8,
                "moe_intermediate_size": 64,
                "hidden_act": "silu",
                "num_experts_per_tok": 2,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 256,
            },
            "transformers.models.qwen3_moe.configuration_qwen3_moe.Qwen3MoeConfig",
        ),
        (
            "transformers.models.olmoe.modeling_olmoe",
            "OlmoeExperts",
            {
                "hidden_size": 128,
                "num_experts": 8,
                "intermediate_size": 64,
                "hidden_act": "silu",
                "num_experts_per_tok": 2,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            },
            "transformers.models.olmoe.configuration_olmoe.OlmoeConfig",
        ),
        (
            "transformers.models.mixtral.modeling_mixtral",
            "MixtralExperts",
            {
                "hidden_size": 128,
                "num_local_experts": 8,
                "intermediate_size": 64,
                "hidden_act": "silu",
                "num_experts_per_tok": 2,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            },
            "transformers.models.mixtral.configuration_mixtral.MixtralConfig",
        ),
    ]

    @pytest.fixture(
        params=[m[1] for m in MODEL_EXPERTS], ids=[m[1] for m in MODEL_EXPERTS]
    )
    def model_setup(self, request, device):
        """Create an Experts instance for each model type."""
        import importlib

        from axolotl.integrations.kernels.libs.scattermoe_lora.gemma4_experts import (
            register_scattermoe_experts,
        )

        register_scattermoe_experts()

        for module_path, cls_name, cfg_kwargs, config_cls_path in self.MODEL_EXPERTS:
            if cls_name == request.param:
                # Import config class
                config_module, config_class = config_cls_path.rsplit(".", 1)
                config_cls = getattr(
                    importlib.import_module(config_module), config_class
                )
                cfg = config_cls(**cfg_kwargs)

                # Import experts class
                module = importlib.import_module(module_path)
                experts_cls = getattr(module, cls_name)

                # Create eager reference
                cfg_eager = config_cls(**cfg_kwargs)
                cfg_eager._experts_implementation = "eager"
                with torch.device("meta"):
                    eager = experts_cls(cfg_eager)
                eager = eager.to_empty(device=device).to(torch.bfloat16)
                nn.init.kaiming_uniform_(eager.gate_up_proj)
                nn.init.kaiming_uniform_(eager.down_proj)

                # Create scattermoe version with same weights
                cfg._experts_implementation = "scattermoe"
                with torch.device("meta"):
                    scatter = experts_cls(cfg)
                scatter = scatter.to_empty(device=device).to(torch.bfloat16)
                scatter.gate_up_proj.data.copy_(eager.gate_up_proj.data)
                scatter.down_proj.data.copy_(eager.down_proj.data)

                return (
                    cls_name,
                    eager,
                    scatter,
                    cfg_kwargs.get(
                        "num_experts", cfg_kwargs.get("num_local_experts", 8)
                    ),
                )

    def test_scattermoe_matches_eager(self, model_setup, device):
        """ScatterMoE ExpertsInterface output matches eager for each model type."""
        cls_name, eager, scatter, num_experts = model_setup
        T, K = 16, 2

        hidden_states = torch.randn(T, 128, device=device, dtype=torch.bfloat16)
        top_k_index = torch.stack(
            [torch.randperm(num_experts, device=device)[:K] for _ in range(T)]
        )
        top_k_weights = torch.softmax(
            torch.randn(T, K, device=device, dtype=torch.bfloat16), dim=-1
        )

        ref_output = eager(hidden_states, top_k_index, top_k_weights)
        scatter_output = scatter(hidden_states, top_k_index, top_k_weights)

        torch.testing.assert_close(
            scatter_output,
            ref_output,
            atol=1e-2,
            rtol=1e-2,
            msg=f"{cls_name}: ScatterMoE output doesn't match eager",
        )
