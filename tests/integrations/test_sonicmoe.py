"""Unit tests for the SonicMoE integration."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.kernels.args import KernelsArgs
from axolotl.integrations.kernels.libs.sonicmoe.routing import (
    sigmoid_topk_routing,
    softmax_topk_routing,
)
from axolotl.integrations.kernels.libs.sonicmoe.weight_converter import (
    ConcatenatedToInterleaved,
    InterleavedToConcatenated,
    register_sonicmoe_weight_converter,
)


class TestKernelsArgs:
    def test_mutual_exclusivity_raises(self):
        with pytest.raises(ValueError, match="Cannot use both"):
            KernelsArgs.model_validate({"use_scattermoe": True, "use_sonicmoe": True})

    def test_sonicmoe_only(self):
        result = KernelsArgs.model_validate({"use_sonicmoe": True})
        assert result.use_sonicmoe is True
        assert result.use_scattermoe is None

    def test_scattermoe_only(self):
        result = KernelsArgs.model_validate({"use_scattermoe": True})
        assert result.use_scattermoe is True
        assert result.use_sonicmoe is None

    def test_neither_set(self):
        result = KernelsArgs.model_validate({})
        assert result.use_scattermoe is None
        assert result.use_sonicmoe is None

    def test_disables_mlp_kernel_when_sonicmoe(self):
        data = {"use_sonicmoe": True, "lora_mlp_kernel": True}
        result = KernelsArgs.disable_mlp_kernel(data)
        assert result["lora_mlp_kernel"] is False
        assert result["mlp_kernel"] is False


class TestConcatenatedToInterleaved:
    @pytest.fixture
    def sample_tensor(self):
        """Create a test tensor [E=2, 2*I=4, H=3] with distinct gate/up values."""
        E, I, H = 2, 2, 3  # noqa: E741
        gate = torch.arange(1, E * I * H + 1, dtype=torch.float32).reshape(E, I, H)
        up = torch.arange(100, 100 + E * I * H, dtype=torch.float32).reshape(E, I, H)
        return torch.cat([gate, up], dim=1)

    def test_interleave_rows_alternate(self, sample_tensor):
        op = ConcatenatedToInterleaved(dim=1)
        result = op.convert(
            {"test": sample_tensor},
            source_patterns=["test"],
            target_patterns=["test"],
        )
        interleaved = result["test"]

        # For expert 0: even rows should be gate, odd rows should be up
        E, two_I, H = sample_tensor.shape
        I = two_I // 2  # noqa: E741
        gate_orig = sample_tensor[:, :I, :]
        up_orig = sample_tensor[:, I:, :]

        assert torch.equal(interleaved[:, 0::2, :], gate_orig)
        assert torch.equal(interleaved[:, 1::2, :], up_orig)

    def test_interleave_handles_list_input(self, sample_tensor):
        op = ConcatenatedToInterleaved(dim=1)
        result = op.convert(
            {"test": [sample_tensor]},
            source_patterns=["test"],
            target_patterns=["test"],
        )
        assert result["test"].shape == sample_tensor.shape

    def test_reverse_op_type(self):
        op = ConcatenatedToInterleaved(dim=1)
        assert isinstance(op.reverse_op, InterleavedToConcatenated)
        assert op.reverse_op.dim == 1


class TestInterleavedToConcatenated:
    @pytest.fixture
    def interleaved_tensor(self):
        """Create an interleaved tensor [E=2, 2*I=4, H=3]."""
        E, I, H = 2, 2, 3  # noqa: E741
        gate = torch.arange(1, E * I * H + 1, dtype=torch.float32).reshape(E, I, H)
        up = torch.arange(100, 100 + E * I * H, dtype=torch.float32).reshape(E, I, H)
        interleaved = torch.empty(E, 2 * I, H)
        interleaved[:, 0::2, :] = gate
        interleaved[:, 1::2, :] = up
        return interleaved

    def test_deinterleave_gate_up_separated(self, interleaved_tensor):
        op = InterleavedToConcatenated(dim=1)
        result = op.convert(
            {"test": interleaved_tensor},
            source_patterns=["test"],
            target_patterns=["test"],
        )
        concatenated = result["test"]

        E, two_I, H = concatenated.shape
        I = two_I // 2  # noqa: E741

        # First half should be gate (even rows from interleaved)
        assert torch.equal(concatenated[:, :I, :], interleaved_tensor[:, 0::2, :])
        # Second half should be up (odd rows from interleaved)
        assert torch.equal(concatenated[:, I:, :], interleaved_tensor[:, 1::2, :])

    def test_reverse_op_type(self):
        op = InterleavedToConcatenated(dim=1)
        assert isinstance(op.reverse_op, ConcatenatedToInterleaved)
        assert op.reverse_op.dim == 1


class TestRoundTrip:
    @pytest.fixture
    def concat_tensor(self):
        E, I, H = 4, 8, 16  # noqa: E741
        gate = torch.randn(E, I, H)
        up = torch.randn(E, I, H)
        return torch.cat([gate, up], dim=1)

    def test_interleave_then_deinterleave_is_identity(self, concat_tensor):
        fwd = ConcatenatedToInterleaved(dim=1)
        rev = InterleavedToConcatenated(dim=1)

        interleaved = fwd.convert(
            {"k": concat_tensor}, source_patterns=["k"], target_patterns=["k"]
        )["k"]
        recovered = rev.convert(
            {"k": interleaved}, source_patterns=["k"], target_patterns=["k"]
        )["k"]

        assert torch.equal(concat_tensor, recovered)

    def test_reverse_op_chain_is_identity(self, concat_tensor):
        """Verify that op.reverse_op produces an exact inverse."""
        op = ConcatenatedToInterleaved(dim=1)
        rev = op.reverse_op

        interleaved = op.convert(
            {"k": concat_tensor}, source_patterns=["k"], target_patterns=["k"]
        )["k"]
        recovered = rev.convert(
            {"k": interleaved}, source_patterns=["k"], target_patterns=["k"]
        )["k"]

        assert torch.equal(concat_tensor, recovered)

    def test_various_shapes(self):
        """Test with different expert counts and dimensions."""
        fwd = ConcatenatedToInterleaved(dim=1)
        rev = InterleavedToConcatenated(dim=1)

        for E, I, H in [(1, 4, 8), (8, 16, 32), (16, 128, 256)]:  # noqa: E741
            concat = torch.randn(E, 2 * I, H)
            interleaved = fwd.convert(
                {"k": concat}, source_patterns=["k"], target_patterns=["k"]
            )["k"]
            recovered = rev.convert(
                {"k": interleaved}, source_patterns=["k"], target_patterns=["k"]
            )["k"]
            assert torch.equal(concat, recovered), (
                f"Failed for shape ({E}, {2 * I}, {H})"
            )


class TestWeightConverterRegistration:
    def test_register_appends_interleave_op(self):
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping

        register_sonicmoe_weight_converter("qwen3_moe")

        modified = get_checkpoint_conversion_mapping("qwen3_moe")
        # Find the gate_up_proj converter
        gate_up_converter = None
        for conv in modified:
            if hasattr(conv, "operations") and any(
                "gate_up_proj" in pat for pat in conv.target_patterns
            ):
                gate_up_converter = conv
                break

        assert gate_up_converter is not None
        assert isinstance(gate_up_converter.operations[-1], ConcatenatedToInterleaved)

    def test_double_registration_is_idempotent(self):
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping

        register_sonicmoe_weight_converter("qwen3_moe")
        register_sonicmoe_weight_converter("qwen3_moe")

        modified = get_checkpoint_conversion_mapping("qwen3_moe")
        for conv in modified:
            if hasattr(conv, "operations") and any(
                "gate_up_proj" in pat for pat in conv.target_patterns
            ):
                interleave_count = sum(
                    isinstance(op, ConcatenatedToInterleaved) for op in conv.operations
                )
                assert interleave_count == 1, (
                    f"Expected 1 ConcatenatedToInterleaved op, got {interleave_count}"
                )
                break

    def test_register_adds_same_key_converter(self):
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping

        register_sonicmoe_weight_converter("qwen3_moe")

        modified = get_checkpoint_conversion_mapping("qwen3_moe")
        # Should have a same-key converter for already-fused checkpoints
        same_key = [
            c
            for c in modified
            if hasattr(c, "source_patterns")
            and c.source_patterns == ["mlp.experts.gate_up_proj"]
            and c.target_patterns == ["mlp.experts.gate_up_proj"]
        ]
        assert len(same_key) == 1
        assert isinstance(same_key[0].operations[0], ConcatenatedToInterleaved)

    def test_register_creates_mapping_when_none(self):
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping

        # qwen3_5_moe has no conversion mapping in transformers
        register_sonicmoe_weight_converter("qwen3_5_moe")

        mapping = get_checkpoint_conversion_mapping("qwen3_5_moe")
        assert mapping is not None
        same_key = [
            c
            for c in mapping
            if hasattr(c, "source_patterns")
            and c.source_patterns == ["mlp.experts.gate_up_proj"]
            and c.target_patterns == ["mlp.experts.gate_up_proj"]
        ]
        assert len(same_key) == 1
        assert isinstance(same_key[0].operations[0], ConcatenatedToInterleaved)


def _make_qwen_moe_block(T=8, H=16, E=4, K=2):
    """Create a mock qwen-style MoE block for routing tests."""
    gate = SimpleNamespace(
        weight=torch.randn(E, H),
        top_k=K,
        num_experts=E,
        norm_topk_prob=True,
    )
    return SimpleNamespace(gate=gate), T, H, E, K


def _make_glm_moe_block(T=8, H=16, E=16, K=4, n_group=2, topk_group=1):
    """Create a mock GLM5-style MoE block for routing tests."""
    gate = SimpleNamespace(
        weight=torch.randn(E, H),
        e_score_correction_bias=torch.zeros(E),
    )
    moe_block = SimpleNamespace(
        gate=gate,
        top_k=K,
        n_routed_experts=E,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    return moe_block, T, H, E, K


def _make_minimax_m2_moe_block(T=8, H=16, E=16, K=4):
    """Create a mock minimax_m2-style MoE block for routing tests.

    minimax_m2 uses sigmoid->topk WITHOUT group selection:
    - e_score_correction_bias is on the moe_block (not on gate)
    - No n_group / topk_group attributes
    - Always normalizes (norm_topk_prob defaults to True)
    - No routed_scaling_factor (defaults to 1.0)
    """
    gate = SimpleNamespace(
        weight=torch.randn(E, H),
        top_k=K,
    )
    moe_block = SimpleNamespace(
        gate=gate,
        top_k=K,
        e_score_correction_bias=torch.zeros(E),
    )
    return moe_block, T, H, E, K


class TestSoftmaxTopkRouting:
    def test_output_shapes(self):
        moe_block, T, H, E, K = _make_qwen_moe_block()
        hidden = torch.randn(T, H)

        scores, token_idx, expert_idx, logits = softmax_topk_routing(hidden, moe_block)

        assert scores.shape == (T * K,)
        assert token_idx.shape == (T * K,)
        assert expert_idx.shape == (T * K,)
        assert logits.shape == (T, E)

    def test_scores_are_float32(self):
        moe_block, T, H, E, K = _make_qwen_moe_block()
        hidden = torch.randn(T, H)

        scores, _, _, _ = softmax_topk_routing(hidden, moe_block)
        assert scores.dtype == torch.float32

    def test_token_indices_sorted_ascending(self):
        moe_block, T, H, E, K = _make_qwen_moe_block()
        hidden = torch.randn(T, H)

        _, token_idx, _, _ = softmax_topk_routing(hidden, moe_block)

        # Token indices must be sorted ascending (SonicMoE requirement)
        diffs = token_idx[1:] - token_idx[:-1]
        assert (diffs >= 0).all()

    def test_expert_indices_in_range(self):
        moe_block, T, H, E, K = _make_qwen_moe_block()
        hidden = torch.randn(T, H)

        _, _, expert_idx, _ = softmax_topk_routing(hidden, moe_block)

        assert (expert_idx >= 0).all()
        assert (expert_idx < E).all()

    def test_renormalized_scores_sum_to_one(self):
        moe_block, T, H, E, K = _make_qwen_moe_block()
        hidden = torch.randn(T, H)

        scores, _, _, _ = softmax_topk_routing(hidden, moe_block)
        per_token_sums = scores.reshape(T, K).sum(dim=-1)
        assert torch.allclose(per_token_sums, torch.ones(T), atol=1e-5)


class TestSigmoidTopkRouting:
    def test_output_shapes(self):
        moe_block, T, H, E, K = _make_glm_moe_block()
        hidden = torch.randn(T, H)

        scores, token_idx, expert_idx, logits = sigmoid_topk_routing(hidden, moe_block)

        assert scores.shape == (T * K,)
        assert token_idx.shape == (T * K,)
        assert expert_idx.shape == (T * K,)
        assert logits.shape == (T, E)

    def test_scores_are_float32(self):
        moe_block, T, H, E, K = _make_glm_moe_block()
        hidden = torch.randn(T, H)

        scores, _, _, _ = sigmoid_topk_routing(hidden, moe_block)
        assert scores.dtype == torch.float32

    def test_token_indices_sorted_ascending(self):
        moe_block, T, H, E, K = _make_glm_moe_block()
        hidden = torch.randn(T, H)

        _, token_idx, _, _ = sigmoid_topk_routing(hidden, moe_block)

        diffs = token_idx[1:] - token_idx[:-1]
        assert (diffs >= 0).all()

    def test_expert_indices_in_range(self):
        moe_block, T, H, E, K = _make_glm_moe_block()
        hidden = torch.randn(T, H)

        _, _, expert_idx, _ = sigmoid_topk_routing(hidden, moe_block)

        assert (expert_idx >= 0).all()
        assert (expert_idx < E).all()

    def test_scores_are_nonnegative(self):
        """Sigmoid outputs are in [0, 1], so scores should be non-negative."""
        moe_block, T, H, E, K = _make_glm_moe_block()
        hidden = torch.randn(T, H)

        scores, _, _, _ = sigmoid_topk_routing(hidden, moe_block)
        assert (scores >= 0).all()

    def test_scaling_factor_applied(self):
        moe_block, T, H, E, K = _make_glm_moe_block()
        hidden = torch.randn(T, H)

        # Get scores with scaling_factor=1.0
        scores_1x, _, _, _ = sigmoid_topk_routing(hidden, moe_block)

        # Get scores with scaling_factor=2.0
        moe_block.routed_scaling_factor = 2.0
        scores_2x, _, _, _ = sigmoid_topk_routing(hidden, moe_block)

        assert torch.allclose(scores_2x, scores_1x * 2.0, atol=1e-5)

    def test_group_selection_restricts_experts(self):
        """With n_group=4 and topk_group=1, only 1/4 of experts should be selectable."""
        moe_block, T, H, E, K = _make_glm_moe_block(E=16, K=2, n_group=4, topk_group=1)
        hidden = torch.randn(T, H)

        _, _, expert_idx, _ = sigmoid_topk_routing(hidden, moe_block)

        # Each token's experts should all fall within a single group (size E//n_group=4)
        expert_idx_2d = expert_idx.reshape(T, K)
        for t in range(T):
            experts = expert_idx_2d[t]
            groups = experts // (E // moe_block.n_group)
            # All selected experts should be from the same group
            assert (groups == groups[0]).all()


class TestMiniMaxM2SigmoidRouting:
    """Tests for minimax_m2 routing: sigmoid->topk without group selection."""

    def test_output_shapes(self):
        """Validates getattr defaults work: n_group=1, E from gate.weight.shape[0]."""
        moe_block, T, H, E, K = _make_minimax_m2_moe_block()
        hidden = torch.randn(T, H)

        scores, token_idx, expert_idx, logits = sigmoid_topk_routing(hidden, moe_block)

        assert scores.shape == (T * K,)
        assert token_idx.shape == (T * K,)
        assert expert_idx.shape == (T * K,)
        assert logits.shape == (T, E)

    def test_bias_on_block_not_gate(self):
        """Verify that e_score_correction_bias on the block (not gate) is used."""
        T, H, E, K = 8, 16, 8, 2
        gate = SimpleNamespace(
            weight=torch.randn(E, H),
            top_k=K,
        )
        # Large positive bias on expert 0 should make it selected more often
        bias = torch.zeros(E)
        bias[0] = 100.0
        moe_block = SimpleNamespace(
            gate=gate,
            top_k=K,
            e_score_correction_bias=bias,
        )
        hidden = torch.randn(T, H)

        _, _, expert_idx, _ = sigmoid_topk_routing(hidden, moe_block)

        # Expert 0 should appear for every token due to the large bias
        expert_idx_2d = expert_idx.reshape(T, K)
        for t in range(T):
            assert 0 in expert_idx_2d[t]


# ============================================================================
# Ernie 4.5 MoE: softmax -> bias correction -> topk
# ============================================================================


def _make_ernie_moe_block(T=8, H=16, E=8, K=2, norm_min=1e-20):
    """Create a mock Ernie 4.5 MoE block for routing tests.

    Ernie 4.5 uses a gate.moe_statics module that adds bias to softmax probs
    before topk selection, then gathers from original probs.
    """
    bias = torch.zeros(E)

    class MockMoeStatics:
        def __init__(self, bias_tensor):
            self.e_score_correction_bias = bias_tensor

        def __call__(self, probs):
            return probs + self.e_score_correction_bias

    gate = SimpleNamespace(
        weight=torch.randn(E, H),
        top_k=K,
        moe_statics=MockMoeStatics(bias),
        norm_min=norm_min,
    )
    moe_block = SimpleNamespace(gate=gate)
    return moe_block, bias, T, H, E, K


class TestSoftmaxBiasTopkRouting:
    """Tests for Ernie 4.5 MoE routing (softmax_bias_topk_routing)."""

    def test_output_shapes(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_bias_topk_routing,
        )

        moe_block, _, T, H, E, K = _make_ernie_moe_block()
        hidden = torch.randn(T, H)

        scores, token_idx, expert_idx, logits = softmax_bias_topk_routing(
            hidden, moe_block
        )

        assert scores.shape == (T * K,)
        assert token_idx.shape == (T * K,)
        assert expert_idx.shape == (T * K,)
        assert logits.shape == (T, E)

    def test_scores_are_float32(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_bias_topk_routing,
        )

        moe_block, _, T, H, E, K = _make_ernie_moe_block()
        hidden = torch.randn(T, H)

        scores, _, _, _ = softmax_bias_topk_routing(hidden, moe_block)
        assert scores.dtype == torch.float32

    def test_token_indices_sorted_ascending(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_bias_topk_routing,
        )

        moe_block, _, T, H, E, K = _make_ernie_moe_block()
        hidden = torch.randn(T, H)

        _, token_idx, _, _ = softmax_bias_topk_routing(hidden, moe_block)
        diffs = token_idx[1:] - token_idx[:-1]
        assert (diffs >= 0).all()

    def test_expert_indices_in_range(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_bias_topk_routing,
        )

        moe_block, _, T, H, E, K = _make_ernie_moe_block()
        hidden = torch.randn(T, H)

        _, _, expert_idx, _ = softmax_bias_topk_routing(hidden, moe_block)
        assert (expert_idx >= 0).all()
        assert (expert_idx < E).all()

    def test_renormalized_scores_sum_to_one(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_bias_topk_routing,
        )

        moe_block, _, T, H, E, K = _make_ernie_moe_block()
        hidden = torch.randn(T, H)

        scores, _, _, _ = softmax_bias_topk_routing(hidden, moe_block)
        per_token_sums = scores.reshape(T, K).sum(dim=-1)
        assert torch.allclose(per_token_sums, torch.ones(T), atol=1e-5)

    def test_bias_affects_expert_selection(self):
        """Large positive bias on expert 0 should make it always selected."""
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_bias_topk_routing,
        )

        moe_block, bias, T, H, E, K = _make_ernie_moe_block()
        bias[0] = 100.0  # mutate the bias to strongly favor expert 0
        hidden = torch.randn(T, H)

        _, _, expert_idx, _ = softmax_bias_topk_routing(hidden, moe_block)
        expert_idx_2d = expert_idx.reshape(T, K)
        for t in range(T):
            assert 0 in expert_idx_2d[t]


# ============================================================================
# DeepSeek V2: softmax -> group_limited_greedy / greedy -> topk
# ============================================================================


def _make_deepseek_v2_moe_block(
    T=8, H=16, E=16, K=4, num_group=2, topk_group=1, topk_method="group_limited_greedy"
):
    """Create a mock DeepSeek V2 MoE block for routing tests.

    DeepSeek V2 uses num_group (not n_group), gate is nn.Linear,
    and supports greedy / group_limited_greedy topk methods.
    """
    gate = SimpleNamespace(weight=torch.randn(E, H))
    moe_block = SimpleNamespace(
        gate=gate,
        top_k=K,
        num_group=num_group,
        topk_group=topk_group,
        topk_method=topk_method,
        routed_scaling_factor=1.0,
    )
    return moe_block, T, H, E, K


class TestSoftmaxGroupLimitedTopkRouting:
    """Tests for DeepSeek V2 routing (softmax_group_limited_topk_routing)."""

    def test_output_shapes_group_limited(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_group_limited_topk_routing,
        )

        moe_block, T, H, E, K = _make_deepseek_v2_moe_block(
            topk_method="group_limited_greedy"
        )
        hidden = torch.randn(T, H)

        scores, token_idx, expert_idx, logits = softmax_group_limited_topk_routing(
            hidden, moe_block
        )

        assert scores.shape == (T * K,)
        assert token_idx.shape == (T * K,)
        assert expert_idx.shape == (T * K,)
        assert logits.shape == (T, E)

    def test_output_shapes_greedy(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_group_limited_topk_routing,
        )

        moe_block, T, H, E, K = _make_deepseek_v2_moe_block(topk_method="greedy")
        hidden = torch.randn(T, H)

        scores, token_idx, expert_idx, logits = softmax_group_limited_topk_routing(
            hidden, moe_block
        )

        assert scores.shape == (T * K,)
        assert logits.shape == (T, E)

    def test_scores_are_float32(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_group_limited_topk_routing,
        )

        moe_block, T, H, E, K = _make_deepseek_v2_moe_block()
        hidden = torch.randn(T, H)

        scores, _, _, _ = softmax_group_limited_topk_routing(hidden, moe_block)
        assert scores.dtype == torch.float32

    def test_token_indices_sorted_ascending(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_group_limited_topk_routing,
        )

        moe_block, T, H, E, K = _make_deepseek_v2_moe_block()
        hidden = torch.randn(T, H)

        _, token_idx, _, _ = softmax_group_limited_topk_routing(hidden, moe_block)
        diffs = token_idx[1:] - token_idx[:-1]
        assert (diffs >= 0).all()

    def test_expert_indices_in_range(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_group_limited_topk_routing,
        )

        moe_block, T, H, E, K = _make_deepseek_v2_moe_block()
        hidden = torch.randn(T, H)

        _, _, expert_idx, _ = softmax_group_limited_topk_routing(hidden, moe_block)
        assert (expert_idx >= 0).all()
        assert (expert_idx < E).all()

    def test_scaling_factor_applied(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_group_limited_topk_routing,
        )

        moe_block, T, H, E, K = _make_deepseek_v2_moe_block(topk_method="greedy")
        hidden = torch.randn(T, H)

        scores_1x, _, _, _ = softmax_group_limited_topk_routing(hidden, moe_block)

        moe_block.routed_scaling_factor = 2.5
        scores_2x, _, _, _ = softmax_group_limited_topk_routing(hidden, moe_block)

        assert torch.allclose(scores_2x, scores_1x * 2.5, atol=1e-5)

    def test_group_selection_restricts_experts(self):
        """With num_group=4 and topk_group=1, experts should come from selected groups."""
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_group_limited_topk_routing,
        )

        moe_block, T, H, E, K = _make_deepseek_v2_moe_block(
            E=16, K=2, num_group=4, topk_group=1, topk_method="group_limited_greedy"
        )
        hidden = torch.randn(T, H)

        _, _, expert_idx, _ = softmax_group_limited_topk_routing(hidden, moe_block)
        expert_idx_2d = expert_idx.reshape(T, K)
        group_size = E // moe_block.num_group
        for t in range(T):
            experts = expert_idx_2d[t]
            groups = experts // group_size
            # All selected experts should be from the same group
            assert (groups == groups[0]).all()

    def test_unsupported_topk_method_raises(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_group_limited_topk_routing,
        )

        moe_block, T, H, E, K = _make_deepseek_v2_moe_block(topk_method="invalid")
        hidden = torch.randn(T, H)

        with pytest.raises(ValueError, match="unsupported topk_method"):
            softmax_group_limited_topk_routing(hidden, moe_block)


# ============================================================================
# HunYuan V1 MoE: softmax -> topk -> renorm (via gate.wg)
# ============================================================================


def _make_hunyuan_moe_block(T=8, H=16, E=8, K=2):
    """Create a mock HunYuan V1 MoE block for routing tests.

    HunYuan V1 uses gate.wg (nn.Linear-like) instead of gate.weight,
    and top_k on the moe_block instead of the gate.
    """
    wg = SimpleNamespace(weight=torch.randn(E, H))
    gate = SimpleNamespace(wg=wg)
    moe_block = SimpleNamespace(gate=gate, top_k=K)
    return moe_block, T, H, E, K


class TestSoftmaxTopkWgRouting:
    """Tests for HunYuan V1 MoE routing (softmax_topk_wg_routing)."""

    def test_output_shapes(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_topk_wg_routing,
        )

        moe_block, T, H, E, K = _make_hunyuan_moe_block()
        hidden = torch.randn(T, H)

        scores, token_idx, expert_idx, logits = softmax_topk_wg_routing(
            hidden, moe_block
        )

        assert scores.shape == (T * K,)
        assert token_idx.shape == (T * K,)
        assert expert_idx.shape == (T * K,)
        assert logits.shape == (T, E)

    def test_scores_are_float32(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_topk_wg_routing,
        )

        moe_block, T, H, E, K = _make_hunyuan_moe_block()
        hidden = torch.randn(T, H)

        scores, _, _, _ = softmax_topk_wg_routing(hidden, moe_block)
        assert scores.dtype == torch.float32

    def test_token_indices_sorted_ascending(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_topk_wg_routing,
        )

        moe_block, T, H, E, K = _make_hunyuan_moe_block()
        hidden = torch.randn(T, H)

        _, token_idx, _, _ = softmax_topk_wg_routing(hidden, moe_block)
        diffs = token_idx[1:] - token_idx[:-1]
        assert (diffs >= 0).all()

    def test_expert_indices_in_range(self):
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_topk_wg_routing,
        )

        moe_block, T, H, E, K = _make_hunyuan_moe_block()
        hidden = torch.randn(T, H)

        _, _, expert_idx, _ = softmax_topk_wg_routing(hidden, moe_block)
        assert (expert_idx >= 0).all()
        assert (expert_idx < E).all()

    def test_renormalized_scores_sum_to_one(self):
        """HunYuan V1 always renormalizes (no norm_topk_prob flag)."""
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_topk_wg_routing,
        )

        moe_block, T, H, E, K = _make_hunyuan_moe_block()
        hidden = torch.randn(T, H)

        scores, _, _, _ = softmax_topk_wg_routing(hidden, moe_block)
        per_token_sums = scores.reshape(T, K).sum(dim=-1)
        assert torch.allclose(per_token_sums, torch.ones(T), atol=1e-5)

    def test_uses_gate_wg_weight(self):
        """Verify that modifying gate.wg.weight changes the routing output."""
        from axolotl.integrations.kernels.libs.sonicmoe.routing import (
            softmax_topk_wg_routing,
        )

        moe_block, T, H, E, K = _make_hunyuan_moe_block()
        hidden = torch.randn(T, H)

        scores1, _, _, _ = softmax_topk_wg_routing(hidden, moe_block)

        # Change the wg weight and verify scores change
        moe_block.gate.wg.weight = torch.randn(E, H)
        scores2, _, _, _ = softmax_topk_wg_routing(hidden, moe_block)

        assert not torch.equal(scores1, scores2)
