"""Tests for eager selective activation checkpointing (SAC)."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import CheckpointPolicy, checkpoint

from axolotl.monkeypatch.selective_checkpointing import (
    SacPolicyState,
    apply_selective_checkpointing,
    build_sac_context_fn,
    build_sac_policy,
)


class _FakeOp:
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name


class _FakeSchemaArg:
    def __init__(self, name: str):
        self.name = name


class _FakeSchema:
    def __init__(self, arg_names: list[str]):
        self.arguments = [_FakeSchemaArg(n) for n in arg_names]


class _FakeFlashOp(_FakeOp):
    """Mimics flash-attn's registered custom op with flattened window args."""

    ARG_NAMES = [
        "q",
        "k",
        "v",
        "dropout_p",
        "softmax_scale",
        "is_causal",
        "window_size_left",
        "window_size_right",
    ]

    def __init__(self, name: str = "flash_attn::_flash_attn_forward"):
        super().__init__(name)
        self._schema = _FakeSchema(self.ARG_NAMES)

    @classmethod
    def args_with_window(cls, left: int, right: int) -> tuple:
        return (None, None, None, 0.0, 1.0, True, left, right)


class TestSacPolicy:
    def test_sdpa_ops_saved(self):
        policy = build_sac_policy(["attention"])
        for packet in (
            torch.ops.aten._scaled_dot_product_flash_attention,
            torch.ops.aten._scaled_dot_product_efficient_attention,
            torch.ops.aten._scaled_dot_product_cudnn_attention,
        ):
            assert policy(None, packet.default) == CheckpointPolicy.MUST_SAVE

    def test_other_ops_recomputed(self):
        policy = build_sac_policy(["attention"])
        assert (
            policy(None, torch.ops.aten.mm.default) == CheckpointPolicy.PREFER_RECOMPUTE
        )
        assert (
            policy(None, torch.ops.aten._softmax.default)
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_flash_attn_custom_op_name_matched(self):
        policy = build_sac_policy(["attention"])
        assert (
            policy(None, _FakeOp("flash_attn::_flash_attn_forward"))
            == CheckpointPolicy.MUST_SAVE
        )
        assert (
            policy(None, _FakeOp("flash_attn::_flash_attn_varlen_forward"))
            == CheckpointPolicy.MUST_SAVE
        )
        assert (
            policy(None, _FakeOp("flash_attn::_flash_attn_backward"))
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_substring_spec(self):
        policy = build_sac_policy(["aten::mm"])
        assert policy(None, torch.ops.aten.mm.default) == CheckpointPolicy.MUST_SAVE
        assert (
            policy(None, torch.ops.aten._scaled_dot_product_flash_attention.default)
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_state_records_saved_ops(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state)
        policy(None, torch.ops.aten._scaled_dot_product_flash_attention.default)
        assert state.saved_op_names == {"aten::_scaled_dot_product_flash_attention"}


class TestSlidingWindowDiscrimination:
    def test_full_attention_saved(self):
        policy = build_sac_policy(["attention"])
        op = _FakeFlashOp()
        args = _FakeFlashOp.args_with_window(-1, -1)
        assert policy(None, op, *args) == CheckpointPolicy.MUST_SAVE

    def test_sliding_window_recomputed(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state)
        op = _FakeFlashOp()
        args = _FakeFlashOp.args_with_window(4095, 0)
        assert policy(None, op, *args) == CheckpointPolicy.PREFER_RECOMPUTE
        assert state.sliding_op_names == {"flash_attn::_flash_attn_forward"}

    def test_causal_right_bound_is_not_sliding(self):
        policy = build_sac_policy(["attention"])
        op = _FakeFlashOp()
        args = _FakeFlashOp.args_with_window(-1, 0)
        assert policy(None, op, *args) == CheckpointPolicy.MUST_SAVE

    def test_sliding_window_kwarg(self):
        policy = build_sac_policy(["attention"])
        op = _FakeFlashOp()
        assert (
            policy(None, op, window_size_left=1024) == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_save_sliding_window_overrides(self):
        policy = build_sac_policy(["attention"], save_sliding_window=True)
        op = _FakeFlashOp()
        args = _FakeFlashOp.args_with_window(4095, 0)
        assert policy(None, op, *args) == CheckpointPolicy.MUST_SAVE

    def test_sdpa_without_window_schema_saved(self):
        policy = build_sac_policy(["attention"])
        op = torch.ops.aten._scaled_dot_product_flash_attention.default
        assert policy(None, op) == CheckpointPolicy.MUST_SAVE


class TestEnableWrap:
    class _FakeModel:
        def __init__(self):
            self.seen_kwargs = None

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.seen_kwargs = gradient_checkpointing_kwargs

    def test_injects_context_fn_and_non_reentrant(self):
        model = self._FakeModel()
        apply_selective_checkpointing(model)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
        assert model.seen_kwargs["use_reentrant"] is False
        assert callable(model.seen_kwargs["context_fn"])

    def test_injects_with_none_kwargs(self):
        model = self._FakeModel()
        apply_selective_checkpointing(model)
        model.gradient_checkpointing_enable()
        assert model.seen_kwargs["use_reentrant"] is False
        assert callable(model.seen_kwargs["context_fn"])

    def test_idempotent(self):
        model = self._FakeModel()
        apply_selective_checkpointing(model)
        wrapped = model.gradient_checkpointing_enable
        apply_selective_checkpointing(model)
        assert model.gradient_checkpointing_enable is wrapped


class TestSacFunctional:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA sdpa")
    def test_checkpointed_attention_grads_match_baseline(self):
        torch.manual_seed(0)
        device = "cuda"
        batch, heads, seq, dim = 2, 4, 128, 64

        def make_inputs():
            gen = torch.Generator(device="cpu").manual_seed(42)
            qkv = torch.randn(
                3, batch, heads, seq, dim, dtype=torch.float32, generator=gen
            )
            return [t.to(device).detach().clone().requires_grad_(True) for t in qkv]

        def attn_block(q, k, v):
            out = F.scaled_dot_product_attention(q, k, v)
            return out.relu() @ v.transpose(-2, -1)

        # baseline: no checkpointing
        q0, k0, v0 = make_inputs()
        attn_block(q0, k0, v0).sum().backward()

        # SAC: checkpointed with save-attention policy
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state)

        def context_fn():
            from torch.utils.checkpoint import create_selective_checkpoint_contexts

            return create_selective_checkpoint_contexts(policy)

        q1, k1, v1 = make_inputs()
        out = checkpoint(
            attn_block, q1, k1, v1, use_reentrant=False, context_fn=context_fn
        )
        out.sum().backward()

        assert state.saved_op_names, "no attention op was matched/saved"
        torch.testing.assert_close(q0.grad, q1.grad)
        torch.testing.assert_close(k0.grad, k1.grad)
        torch.testing.assert_close(v0.grad, v1.grad)

    def test_context_fn_returns_fresh_contexts(self):
        context_fn = build_sac_context_fn(["attention"])
        c1 = context_fn()
        c2 = context_fn()
        assert c1 is not c2


class TestLayerTypeDiscrimination:
    SDPA_OP = torch.ops.aten._scaled_dot_product_flash_attention.default

    def test_sliding_layer_type_recomputed(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state)
        state.current_layer_type = "sliding_attention"
        assert policy(None, self.SDPA_OP) == CheckpointPolicy.PREFER_RECOMPUTE
        state.current_layer_type = "chunked_attention"
        assert policy(None, self.SDPA_OP) == CheckpointPolicy.PREFER_RECOMPUTE

    def test_full_or_unknown_layer_type_saved(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state)
        state.current_layer_type = "full_attention"
        assert policy(None, self.SDPA_OP) == CheckpointPolicy.MUST_SAVE
        state.current_layer_type = None
        assert policy(None, self.SDPA_OP) == CheckpointPolicy.MUST_SAVE

    def test_save_sliding_window_overrides_layer_type(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state, save_sliding_window=True)
        state.current_layer_type = "sliding_attention"
        assert policy(None, self.SDPA_OP) == CheckpointPolicy.MUST_SAVE

    def test_hooks_publish_layer_type(self):
        from transformers import GradientCheckpointingLayer

        from axolotl.monkeypatch.selective_checkpointing import (
            install_layer_type_hooks,
        )

        state = SacPolicyState()
        seen = []

        class _Layer(GradientCheckpointingLayer):
            def __init__(self, layer_type):
                super().__init__()
                self.layer_type = layer_type

            def forward(self):
                seen.append(state.current_layer_type)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [_Layer("full_attention"), _Layer("sliding_attention")]
                )

            def forward(self):
                for layer in self.layers:
                    layer()

        model = _Model()
        hooked = install_layer_type_hooks(model, state)
        assert hooked == 2
        model()
        assert seen == ["full_attention", "sliding_attention"]
        assert state.current_layer_type is None
