"""Tests for eager selective activation checkpointing (SAC)."""

import logging

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import (
    CheckpointPolicy,
    SelectiveCheckpointContext,
    checkpoint,
)

import axolotl.monkeypatch.selective_checkpointing as selective_checkpointing
from axolotl.monkeypatch.selective_checkpointing import (
    _NO_MATCH_WARN_REGIONS,
    SacPolicyState,
    _module_name_matches,
    apply_selective_checkpointing,
    build_sac_context_fn,
    build_sac_policy,
    install_module_scope_hooks,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
DEVICES = ["cpu", pytest.param("cuda", marks=requires_cuda)]

MM = torch.ops.aten.mm.default
ADDMM = torch.ops.aten.addmm.default
SDPA = torch.ops.aten._scaled_dot_product_flash_attention.default


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

    @pytest.mark.parametrize("supports_kwarg", [False, True])
    @pytest.mark.parametrize(
        "caller_kwargs",
        [
            {"preserve_rng_state": False},
            {"respect_saved_tensors_hooks": True},
        ],
    )
    def test_respect_saved_tensors_hooks_is_feature_gated(
        self, monkeypatch, supports_kwarg, caller_kwargs
    ):
        monkeypatch.setattr(
            selective_checkpointing,
            "_SUPPORTS_RESPECT_SAVED_TENSORS_HOOKS",
            supports_kwarg,
        )
        model = self._FakeModel()
        original_kwargs = dict(caller_kwargs)
        apply_selective_checkpointing(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=caller_kwargs)

        assert caller_kwargs == original_kwargs
        if supports_kwarg:
            expected = caller_kwargs.get("respect_saved_tensors_hooks", False)
            assert model.seen_kwargs["respect_saved_tensors_hooks"] is expected
        else:
            assert "respect_saved_tensors_hooks" not in model.seen_kwargs

    def test_idempotent(self):
        model = self._FakeModel()
        apply_selective_checkpointing(model)
        wrapped = model.gradient_checkpointing_enable
        apply_selective_checkpointing(model)
        assert model.gradient_checkpointing_enable is wrapped

    def test_rules_flow_to_context_fn(self, sac_log):
        class _Model(_TinyMlpStack):
            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self.seen_kwargs = gradient_checkpointing_kwargs

        model = _Model()
        apply_selective_checkpointing(
            model, save_modules=["down"], save_matmul_min_k=256
        )
        model.gradient_checkpointing_enable()

        for layer in model.layers:
            assert layer.down._forward_pre_hooks
            assert layer.down._forward_hooks
            assert not layer.up._forward_pre_hooks
        assert callable(model.seen_kwargs["context_fn"])
        assert "save_modules=['down']" in sac_log.text
        assert "save_matmul_min_k=256" in sac_log.text


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


@pytest.fixture(name="sac_log")
def fixture_sac_log(caplog):
    # axolotl's logging config may stop propagation above this logger, so attach
    # directly and stop propagating to avoid double capture via root
    logger = logging.getLogger("axolotl.monkeypatch.selective_checkpointing")
    propagate = logger.propagate
    logger.addHandler(caplog.handler)
    logger.propagate = False
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.propagate = propagate


class _MlpLayer(torch.nn.Module):
    def __init__(self, hidden: int, inter: int, post_norm: bool = False):
        super().__init__()
        self.up = torch.nn.Linear(hidden, inter, bias=False)
        self.down = torch.nn.Linear(inter, hidden, bias=True)
        self.norm = torch.nn.LayerNorm(hidden) if post_norm else None

    def forward(self, x):
        out = self.down(F.relu(self.up(x)))
        if self.norm is not None:
            out = self.norm(out)
        return x + out


class _TinyMlpStack(torch.nn.Module):
    """``up`` has no bias (aten::mm, K=hidden); ``down`` has one (aten::addmm, K=inter)."""

    def __init__(
        self, n_layers: int = 2, hidden: int = 64, inter: int = 256, **layer_kwargs
    ):
        super().__init__()
        self.hidden = hidden
        self.layers = torch.nn.ModuleList(
            [_MlpLayer(hidden, inter, **layer_kwargs) for _ in range(n_layers)]
        )

    def forward(self, x, context_fn=None):
        for layer in self.layers:
            if context_fn is None:
                x = layer(x)
            else:
                x = checkpoint(layer, x, use_reentrant=False, context_fn=context_fn)
        return x


def _sac_region():
    """A SAC forward dispatch mode, so module-scope hooks count as in-region."""
    forward_mode, _ = torch.utils.checkpoint.create_selective_checkpoint_contexts(
        lambda ctx, op, *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE
    )
    return forward_mode


def _run_stack(model, hidden, device, context_fn=None, seq=32, dtype=torch.float32):
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, seq, hidden, generator=gen).to(device, dtype)
    x.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model(x, context_fn=context_fn).float().pow(2).mean().backward()
    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return x.grad.clone(), grads


def _assert_grads_match(baseline, actual):
    x0, g0 = baseline
    x1, g1 = actual
    torch.testing.assert_close(x0, x1)
    assert g0.keys() == g1.keys()
    for name, grad in g0.items():
        torch.testing.assert_close(grad, g1[name], msg=name)


def _peft_stack(**lora_kwargs):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = _TinyMlpStack()
    return base, lambda: get_peft_model(
        base,
        LoraConfig(
            r=4, target_modules=["down"], init_lora_weights=False, **lora_kwargs
        ),
    )


class TestModuleNameMatching:
    NAME = "model.layers.0.mlp.down_proj"
    PEFT_NAME = "base_model.model.model.layers.0.mlp.down_proj"

    def test_exact_and_suffix(self):
        assert _module_name_matches(self.NAME, self.NAME)
        assert _module_name_matches(self.NAME, "down_proj")
        assert _module_name_matches(self.NAME, "mlp.down_proj")
        assert not _module_name_matches(self.NAME, "proj")
        assert not _module_name_matches(self.NAME, "own_proj")
        assert not _module_name_matches(self.NAME + ".base_layer", "down_proj")

    def test_glob_is_anchored(self):
        assert _module_name_matches(self.NAME, "*.mlp.down_proj")
        assert _module_name_matches(self.PEFT_NAME, "*.mlp.down_proj")
        assert _module_name_matches(self.NAME, "model.layers.*.mlp.down_proj")
        assert not _module_name_matches(self.PEFT_NAME, "model.layers.*.mlp.down_proj")
        assert _module_name_matches(self.PEFT_NAME, "*down_proj*")
        assert not _module_name_matches(self.NAME, "down_pro?")
        assert _module_name_matches(self.NAME, "*.[du][op]*_proj")

    def test_install_resolves_peft_base_layer(self):
        _, wrap = _peft_stack()
        model = wrap()
        state = SacPolicyState()
        assert install_module_scope_hooks(model, state, ["down"]) == {"down": 2}
        for layer in model.base_model.model.layers:
            assert layer.down.base_layer._forward_pre_hooks
            assert layer.down.base_layer._forward_hooks
            assert not layer.down._forward_pre_hooks
            assert not layer.down.lora_A["default"]._forward_pre_hooks
            assert not layer.down.lora_B["default"]._forward_pre_hooks

    def test_install_glob_dedupes_nested(self):
        _, wrap = _peft_stack()
        model = wrap()
        state = SacPolicyState()
        assert install_module_scope_hooks(model, state, ["*down*"]) == {"*down*": 2}
        for layer in model.base_model.model.layers:
            assert len(layer.down.base_layer._forward_pre_hooks) == 1
            assert not layer.down._forward_pre_hooks
            assert not layer.down.lora_A._forward_pre_hooks
            assert not layer.down.lora_A["default"]._forward_pre_hooks

    def test_container_entry_hooks_container_only(self):
        model = _TinyMlpStack()
        state = SacPolicyState()
        assert install_module_scope_hooks(model, state, ["layers.0"]) == {"layers.0": 1}
        assert model.layers[0]._forward_pre_hooks
        assert not model.layers[0].down._forward_pre_hooks

    def test_same_module_under_two_entries(self):
        model = _TinyMlpStack()
        state = SacPolicyState()
        hooked = install_module_scope_hooks(model, state, ["down", "layers.0.down"])
        assert hooked == {"down": 2, "layers.0.down": 1}
        with _sac_region():
            model(torch.randn(1, 4, 64))
        assert state.hook_fires == {"down": 2, "layers.0.down": 1}

    def test_install_zero_match_warns(self, sac_log):
        model = _TinyMlpStack()
        state = SacPolicyState()
        assert install_module_scope_hooks(model, state, ["down_proj"]) == {
            "down_proj": 0
        }
        assert "'down_proj' matched no module in _TinyMlpStack" in sac_log.text
        hint = sac_log.text.split("include: ", 1)[1]
        assert "down" in hint and "up" in hint

    def test_install_without_named_modules_warns(self, sac_log):
        state = SacPolicyState()
        assert install_module_scope_hooks(object(), state, ["down"]) == {}
        assert "'down' matched no module" in sac_log.text


class TestModuleScopeRules:
    def test_depth_counter_brackets_forward(self):
        model = _TinyMlpStack(n_layers=1)
        state = SacPolicyState()
        install_module_scope_hooks(model, state, ["down"])
        seen = []
        model.layers[0].down.register_forward_pre_hook(
            lambda mod, args: seen.append(state.module_depth["down"])
        )
        assert state.module_depth["down"] == 0
        with _sac_region():
            model(torch.randn(1, 4, 64))
        assert seen == [1]
        assert state.module_depth["down"] == 0
        assert state.hook_fires["down"] == 1

    def test_depth_resets_when_forward_raises(self):
        class _Boom(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("boom")

        model = torch.nn.Sequential()
        model.add_module("down", _Boom())
        state = SacPolicyState()
        install_module_scope_hooks(model, state, ["down"])
        with pytest.raises(RuntimeError, match="boom"), _sac_region():
            model(torch.randn(2))
        assert state.module_depth["down"] == 0
        assert state.scope_stack["down"] == []
        assert state.hook_fires["down"] == 1

    def test_nested_entries_track_independently(self):
        model = _TinyMlpStack(n_layers=1)
        state = SacPolicyState()
        install_module_scope_hooks(model, state, ["layers.0", "down"])
        inside_down, inside_up = [], []
        model.layers[0].down.register_forward_pre_hook(
            lambda mod, args: inside_down.append(dict(state.module_depth))
        )
        model.layers[0].up.register_forward_pre_hook(
            lambda mod, args: inside_up.append(dict(state.module_depth))
        )
        with _sac_region():
            model(torch.randn(1, 4, 64))
        assert inside_up == [{"layers.0": 1, "down": 0}]
        assert inside_down == [{"layers.0": 1, "down": 1}]
        assert state.module_depth == {"layers.0": 0, "down": 0}

    def test_scope_outside_region_not_counted(self):
        model = _TinyMlpStack(n_layers=1)
        state = SacPolicyState()
        install_module_scope_hooks(model, state, ["down"])
        seen = []
        model.layers[0].down.register_forward_pre_hook(
            lambda mod, args: seen.append(state.module_depth["down"])
        )
        model(torch.randn(1, 4, 64))
        assert seen == [0]
        assert state.hook_fires["down"] == 0
        assert state.hook_fires_outside["down"] == 1
        assert state.scope_stack["down"] == []

    def test_install_skips_modules_enclosing_checkpointed_layers(self, sac_log):
        from transformers import GradientCheckpointingLayer

        class _GcLayer(GradientCheckpointingLayer):
            def __init__(self):
                super().__init__()
                self.down = torch.nn.Linear(8, 8)

        class _Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([_GcLayer(), _GcLayer()])
                self.lm_head = torch.nn.Linear(8, 8)

        model = _Outer()
        state = SacPolicyState()
        hooked = install_module_scope_hooks(model, state, ["model", "*", "down"])
        assert hooked == {"model": 0, "*": 3, "down": 2}
        assert not model._forward_pre_hooks
        assert not model.model._forward_pre_hooks
        assert not model.model.layers._forward_pre_hooks
        for layer in model.model.layers:
            assert len(layer._forward_pre_hooks) == 1
            assert len(layer.down._forward_pre_hooks) == 1
        assert "'model' matches 1 module(s) that enclose" in sac_log.text
        assert "'<root>' contains 'model.layers.0'" in sac_log.text
        assert "matched no module" not in sac_log.text
        assert state.module_targets == {"model": 0, "*": 3, "down": 2}

    def test_root_glob_match_drops_descendants(self):
        model = _TinyMlpStack()
        state = SacPolicyState()
        assert install_module_scope_hooks(model, state, ["*"]) == {"*": 1}
        assert model._forward_pre_hooks
        assert not model.layers[0]._forward_pre_hooks
        assert not model.layers[0].down._forward_pre_hooks

    def test_save_list_preemption_credits_rules(self):
        state = SacPolicyState()
        policy = build_sac_policy(
            ["attention", "aten::mm"],
            state,
            save_modules=["down"],
            save_matmul_min_k=8,
        )
        a, b = torch.randn(4, 8), torch.randn(8, 3)
        assert policy(None, MM, a, b) == CheckpointPolicy.MUST_SAVE
        assert state.rule_saves == {"save": 1, "module:down": 0, "shape": 1}
        state.module_depth["down"] = 1
        policy(None, MM, a, b)
        assert state.rule_saves == {"save": 2, "module:down": 1, "shape": 1}

    def test_sliding_layer_save_list_falls_through_to_rules(self):
        a, b = torch.randn(4, 8), torch.randn(8, 3)
        state = SacPolicyState()
        policy = build_sac_policy(["aten::mm"], state)
        state.current_layer_type = "sliding_attention"
        assert policy(None, MM, a, b) == CheckpointPolicy.PREFER_RECOMPUTE

        state = SacPolicyState()
        policy = build_sac_policy(["aten::mm"], state, save_matmul_min_k=8)
        state.current_layer_type = "sliding_attention"
        assert policy(None, MM, a, b) == CheckpointPolicy.MUST_SAVE
        assert state.rule_saves == {"save": 0, "shape": 1}

    def test_saves_matmul_in_scope(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state, save_modules=["down"])
        state.module_depth["down"] = 1
        a, b = torch.randn(4, 8), torch.randn(8, 3)
        assert policy(None, MM, a, b) == CheckpointPolicy.MUST_SAVE
        assert policy(None, ADDMM, torch.randn(3), a, b) == CheckpointPolicy.MUST_SAVE
        for name in ("bitsandbytes::gemm_4bit", "aten::_grouped_mm", "aten::linear"):
            assert policy(None, _FakeOp(name)) == CheckpointPolicy.MUST_SAVE
        assert state.saved_op_names >= {
            "module:down:aten::mm",
            "module:down:aten::addmm",
        }

    def test_recomputes_outside_scope_and_non_matmul(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state, save_modules=["down"])
        a, b = torch.randn(4, 8), torch.randn(8, 3)
        assert policy(None, MM, a, b) == CheckpointPolicy.PREFER_RECOMPUTE
        state.module_depth["down"] = 1
        for op in (
            torch.ops.aten._softmax.default,
            _FakeOp("aten::addmm_"),
            _FakeOp("aten::bmm"),
            _FakeOp("aten::matmul"),
            _FakeOp("aten::mm_out"),
        ):
            assert policy(None, op) == CheckpointPolicy.PREFER_RECOMPUTE
        assert state.rule_saves["module:down"] == 0

    def test_counts_forward_only(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state, save_modules=["down"])
        state.module_depth["down"] = 1
        a, b = torch.randn(4, 8), torch.randn(8, 3)

        recompute = SelectiveCheckpointContext(is_recompute=True)
        assert policy(recompute, MM, a, b) == CheckpointPolicy.MUST_SAVE
        assert state.rule_saves["module:down"] == 0
        assert state.rule_replays == {"module:down": 1}
        assert state.recompute_seen
        policy(SelectiveCheckpointContext(is_recompute=False), MM, a, b)
        assert state.rule_saves["module:down"] == 1
        policy(None, MM, a, b)
        assert state.rule_saves["module:down"] == 2
        assert state.rule_replays == {"module:down": 1}

    def test_active_entries_all_credited(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state, save_modules=["mlp", "down"])
        state.module_depth.update({"mlp": 1, "down": 1})
        policy(None, MM, torch.randn(4, 8), torch.randn(8, 3))
        assert state.rule_saves["module:mlp"] == 1
        assert state.rule_saves["module:down"] == 1

    def test_attention_still_saved_with_rules(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state, save_modules=["down"])
        assert policy(None, SDPA) == CheckpointPolicy.MUST_SAVE
        assert state.rule_saves["save"] == 1
        assert state.rule_saves["module:down"] == 0
        state.current_layer_type = "sliding_attention"
        assert policy(None, SDPA) == CheckpointPolicy.PREFER_RECOMPUTE

    def test_matmul_in_scope_ignores_sliding_layer_type(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state, save_modules=["down"])
        state.current_layer_type = "sliding_attention"
        state.module_depth["down"] = 1
        assert (
            policy(None, MM, torch.randn(4, 8), torch.randn(8, 3))
            == CheckpointPolicy.MUST_SAVE
        )


class TestShapeRules:
    def test_mm_and_addmm_k_positions(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state, save_matmul_min_k=256)
        a256, b256 = torch.randn(4, 256), torch.randn(256, 64)
        a64, b64 = torch.randn(4, 64), torch.randn(64, 256)
        assert policy(None, MM, a256, b256) == CheckpointPolicy.MUST_SAVE
        assert policy(None, MM, a64, b64) == CheckpointPolicy.PREFER_RECOMPUTE
        assert (
            policy(None, ADDMM, torch.randn(64), a256, b256)
            == CheckpointPolicy.MUST_SAVE
        )
        assert (
            policy(None, ADDMM, torch.randn(256), a64, b64)
            == CheckpointPolicy.PREFER_RECOMPUTE
        )
        assert state.rule_saves["shape"] == 2
        assert state.saved_op_names == {"shape:aten::mm", "shape:aten::addmm"}

    def test_grouped_mm_and_gemm4bit_k(self):
        policy = build_sac_policy(["attention"], save_matmul_min_k=512)
        grouped = _FakeOp("aten::_grouped_mm")
        a = torch.randn(16, 512)
        assert (
            policy(None, grouped, a, torch.randn(4, 512, 64))
            == CheckpointPolicy.MUST_SAVE
        )
        assert (
            policy(None, grouped, a, torch.randn(4, 256, 64))
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

        policy = build_sac_policy(["attention"], save_matmul_min_k=8192)
        gemm = _FakeOp("bitsandbytes::gemm_4bit")
        assert policy(None, gemm, torch.empty(2, 7, 8192)) == CheckpointPolicy.MUST_SAVE
        assert (
            policy(None, gemm, torch.empty(2, 7, 4096))
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_linear_k_from_weight(self):
        policy = build_sac_policy(["attention"], save_matmul_min_k=128)
        linear = _FakeOp("aten::linear")
        x = torch.randn(2, 128)
        assert (
            policy(None, linear, x, torch.randn(8, 128)) == CheckpointPolicy.MUST_SAVE
        )
        assert (
            policy(None, linear, torch.randn(2, 64), torch.randn(8, 64))
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_bmm_and_inplace_never_match(self):
        policy = build_sac_policy(["attention"], save_matmul_min_k=8)
        huge = torch.empty(1, 32768, 32768, device="meta")
        for name in ("aten::bmm", "aten::addmm_", "aten::matmul", "aten::baddbmm"):
            assert (
                policy(None, _FakeOp(name), huge, huge, huge)
                == CheckpointPolicy.PREFER_RECOMPUTE
            )

    def test_malformed_args_never_raise(self):
        policy = build_sac_policy(["attention"], save_matmul_min_k=8)
        for args in (
            (),
            (torch.randn(4, 8),),
            ("a", "b"),
            (torch.randn(4, 8), None),
            (torch.randn(8), torch.randn(8)),
            (torch.randn(4, 8), 3),
        ):
            assert policy(None, MM, *args) == CheckpointPolicy.PREFER_RECOMPUTE
        assert policy(None, ADDMM, None, None) == CheckpointPolicy.PREFER_RECOMPUTE
        assert (
            policy(None, _FakeOp("bitsandbytes::gemm_4bit"), torch.tensor(3.0))
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_counts_forward_only(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state, save_matmul_min_k=8)
        a, b = torch.randn(4, 8), torch.randn(8, 3)
        policy(SelectiveCheckpointContext(is_recompute=True), MM, a, b)
        assert state.rule_saves["shape"] == 0
        assert state.rule_replays == {"shape": 1}
        policy(SelectiveCheckpointContext(is_recompute=False), MM, a, b)
        assert state.rule_saves["shape"] == 1

    def test_no_rules_is_byte_for_byte(self):
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state)
        assert state.save_modules == []
        assert state.save_matmul_min_k is None
        assert state.rule_saves == {"save": 0}
        assert state.module_depth == {}
        state.module_depth["down"] = 1
        big = torch.randn(4, 8192), torch.randn(8192, 4)
        assert policy(None, MM, *big) == CheckpointPolicy.PREFER_RECOMPUTE
        assert policy(None, SDPA) == CheckpointPolicy.MUST_SAVE
        assert state.rule_saves == {"save": 1}


class TestRuleWarnings:
    @staticmethod
    def _drain(context_fn):
        for _ in range(_NO_MATCH_WARN_REGIONS):
            context_fn()

    @staticmethod
    def _warnings(caplog):
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    def test_no_warning_before_threshold(self, sac_log):
        context_fn = build_sac_context_fn(["attention"], save_matmul_min_k=8192)
        for _ in range(_NO_MATCH_WARN_REGIONS - 1):
            context_fn()
        assert not self._warnings(sac_log)

    def test_module_rule_hooks_never_fired(self, sac_log):
        state = SacPolicyState()
        state.rule_saves["save"] = 1
        self._drain(build_sac_context_fn(state=state, save_modules=["down"]))
        (warning,) = self._warnings(sac_log)
        assert "'down'" in warning
        assert "module hooks never fired" in warning

    def test_module_rule_hooks_fired_no_matmul(self, sac_log):
        state = SacPolicyState()
        state.rule_saves["save"] = 1
        context_fn = build_sac_context_fn(state=state, save_modules=["down"])
        state.hook_fires["down"] = 5
        self._drain(context_fn)
        (warning,) = self._warnings(sac_log)
        assert "does not dispatch a visible matmul" in warning

    def test_shape_rule_never_matched(self, sac_log):
        state = SacPolicyState()
        state.rule_saves["save"] = 1
        self._drain(build_sac_context_fn(state=state, save_matmul_min_k=8192))
        (warning,) = self._warnings(sac_log)
        assert "save_matmul_min_k=8192 never matched" in warning

    def test_attention_warning_persists_when_rule_saved(self, sac_log):
        state = SacPolicyState()
        context_fn = build_sac_context_fn(state=state, save_modules=["down"])
        state.rule_saves["module:down"] = 3
        self._drain(context_fn)
        (warning,) = self._warnings(sac_log)
        assert "no op matched the save policy" in warning
        assert "Module/shape rules did save tensors" in warning
        assert "everything is being recomputed" not in warning

    def test_silent_when_everything_saved(self, sac_log):
        state = SacPolicyState()
        context_fn = build_sac_context_fn(
            state=state, save_modules=["down"], save_matmul_min_k=8192
        )
        state.rule_saves.update({"save": 1, "module:down": 1, "shape": 1})
        self._drain(context_fn)
        context_fn()
        assert not self._warnings(sac_log)
        assert state.warned_no_match

    def test_no_rules_message_unchanged(self, sac_log):
        self._drain(build_sac_context_fn(["attention"]))
        assert self._warnings(sac_log) == [
            f"selective_checkpointing: no op matched the save policy after "
            f"{_NO_MATCH_WARN_REGIONS} checkpoint regions. Your attention "
            "implementation may not be dispatcher-visible (e.g. a custom kernel "
            "not registered via torch.library); everything is being recomputed "
            "as with plain gradient checkpointing."
        ]

    def test_warns_once(self, sac_log):
        context_fn = build_sac_context_fn(["attention"], save_matmul_min_k=8192)
        self._drain(context_fn)
        self._drain(context_fn)
        assert len(self._warnings(sac_log)) == 2

    def test_module_rule_only_outside_regions(self, sac_log):
        state = SacPolicyState()
        state.rule_saves["save"] = 1
        context_fn = build_sac_context_fn(state=state, save_modules=["lm_head"])
        state.hook_fires_outside["lm_head"] = 5
        self._drain(context_fn)
        (warning,) = self._warnings(sac_log)
        assert "only runs outside the checkpoint regions" in warning

    def test_entry_without_targets_not_rewarned(self, sac_log):
        state = SacPolicyState()
        state.rule_saves["save"] = 1
        context_fn = build_sac_context_fn(state=state, save_modules=["model"])
        state.module_targets["model"] = 0
        self._drain(context_fn)
        assert not self._warnings(sac_log)

    @pytest.mark.parametrize("post_norm", [False, True])
    def test_dead_save_warning(self, sac_log, post_norm):
        torch.manual_seed(0)
        model = _TinyMlpStack(post_norm=post_norm)
        state = SacPolicyState()
        context_fn = build_sac_context_fn(state=state, save_matmul_min_k=256)
        _run_stack(model, model.hidden, "cpu", context_fn)
        assert state.rule_saves["shape"] == 2
        assert state.rule_replays.get("shape", 0) == (2 if post_norm else 0)
        state.regions_seen = _NO_MATCH_WARN_REGIONS
        _run_stack(model, model.hidden, "cpu", context_fn)
        dead = [w for w in self._warnings(sac_log) if "never read one" in w]
        if post_norm:
            assert not dead
        else:
            (warning,) = dead
            assert "save_matmul_min_k=256 saved 2 tensors" in warning
            assert "early stop" in warning
        assert state.warned_dead_saves

    def test_dead_save_waits_for_backward(self, sac_log):
        state = SacPolicyState()
        context_fn = build_sac_context_fn(state=state, save_modules=["down"])
        state.rule_saves.update({"save": 1, "module:down": 70})
        self._drain(context_fn)
        assert not self._warnings(sac_log)
        assert not state.warned_dead_saves
        state.recompute_seen = True
        context_fn()
        (warning,) = self._warnings(sac_log)
        assert "save_modules entry 'down' saved 70 tensors" in warning

    def test_offload_context_fn_also_warns(self, sac_log):
        from axolotl.monkeypatch.selective_checkpointing_offload import (
            SacOffloadEngine,
            build_sac_offload_context_fn,
        )

        state = SacPolicyState()
        context_fn = build_sac_offload_context_fn(
            ["attention"],
            state=state,
            engine=SacOffloadEngine(),
            save_modules=["down"],
        )
        self._drain(context_fn)
        warnings = self._warnings(sac_log)
        assert any("no op matched the save policy" in w for w in warnings)
        assert any("module hooks never fired" in w for w in warnings)


class TestRuleFunctional:
    @pytest.mark.parametrize("device", DEVICES)
    def test_module_rule_grads_match_baseline(self, device):
        torch.manual_seed(0)
        model = _TinyMlpStack().to(device)
        baseline = _run_stack(model, model.hidden, device)

        state = SacPolicyState()
        install_module_scope_hooks(model, state, ["down"])
        context_fn = build_sac_context_fn(state=state, save_modules=["down"])
        _assert_grads_match(
            baseline, _run_stack(model, model.hidden, device, context_fn)
        )
        assert state.rule_saves["module:down"] == 2
        assert state.saved_op_names == {"module:down:aten::addmm"}
        assert state.module_depth == {"down": 0}

    @pytest.mark.parametrize("device", DEVICES)
    def test_shape_rule_grads_match_baseline(self, device):
        torch.manual_seed(0)
        model = _TinyMlpStack().to(device)
        baseline = _run_stack(model, model.hidden, device)

        state = SacPolicyState()
        context_fn = build_sac_context_fn(state=state, save_matmul_min_k=256)
        _assert_grads_match(
            baseline, _run_stack(model, model.hidden, device, context_fn)
        )
        assert state.rule_saves["shape"] == 2
        assert state.saved_op_names == {"shape:aten::addmm"}

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "post_norm,early_stop,consumed",
        [(False, True, 0), (False, False, 2), (True, True, 2), (True, False, 2)],
    )
    def test_saved_output_consumed_only_when_recompute_reaches_it(
        self, device, post_norm, early_stop, consumed
    ):
        # early stop ends the replay once the last saved-for-backward tensor is
        # rebuilt; with only a residual add after it, down is never replayed
        torch.manual_seed(0)
        model = _TinyMlpStack(post_norm=post_norm).to(device)
        state = SacPolicyState()
        policy = build_sac_policy(state=state, save_matmul_min_k=256)
        recomputed = []

        def spy(ctx, op, *args, **kwargs):
            result = policy(ctx, op, *args, **kwargs)
            if ctx.is_recompute and result == CheckpointPolicy.MUST_SAVE:
                recomputed.append(selective_checkpointing._op_name(op))
            return result

        def context_fn():
            return torch.utils.checkpoint.create_selective_checkpoint_contexts(spy)

        baseline = _run_stack(model, model.hidden, device)
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            actual = _run_stack(model, model.hidden, device, context_fn)
        _assert_grads_match(baseline, actual)
        assert state.rule_saves["shape"] == 2
        assert recomputed == ["aten::addmm"] * consumed

    @pytest.mark.parametrize("device", DEVICES)
    def test_enclosing_entry_does_not_desync_cache(self, device):
        # without a GradientCheckpointingLayer to detect at install, only the
        # region gate keeps an enclosing scope out of forward-only decisions
        class _Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _TinyMlpStack(hidden=64, inter=64)
                self.hidden = 64

            def forward(self, x, context_fn=None):
                return self.model(x, context_fn=context_fn)

        torch.manual_seed(0)
        model = _Outer().to(device)
        baseline = _run_stack(model, 64, device)
        state = SacPolicyState()
        entries = ["model", "down"]
        assert install_module_scope_hooks(model, state, entries) == {
            "model": 1,
            "down": 2,
        }
        context_fn = build_sac_context_fn(state=state, save_modules=entries)
        _assert_grads_match(baseline, _run_stack(model, 64, device, context_fn))
        assert state.rule_saves["module:model"] == 0
        assert state.rule_saves["module:down"] == 2
        assert state.hook_fires_outside["model"] == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_peft_lora_module_rule(self, device):
        base, wrap = _peft_stack()
        state = SacPolicyState()
        install_module_scope_hooks(base, state, ["down"])
        model = wrap().to(device)
        baseline = _run_stack(model, 64, device)

        context_fn = build_sac_context_fn(state=state, save_modules=["down"])
        actual = _run_stack(model, 64, device, context_fn)
        _assert_grads_match(baseline, actual)
        assert any("lora_A" in name for name in actual[1])
        assert state.rule_saves["module:down"] == 2
        assert state.saved_op_names == {"module:down:aten::addmm"}

    @pytest.mark.parametrize("device", DEVICES)
    def test_peft_lora_shape_rule(self, device):
        _, wrap = _peft_stack()
        model = wrap().to(device)
        baseline = _run_stack(model, 64, device)

        state = SacPolicyState()
        context_fn = build_sac_context_fn(state=state, save_matmul_min_k=256)
        _assert_grads_match(baseline, _run_stack(model, 64, device, context_fn))
        assert state.rule_saves["shape"] == 4
        assert state.saved_op_names == {"shape:aten::addmm", "shape:aten::mm"}

    @requires_cuda
    @pytest.mark.skipif(
        not hasattr(torch, "_grouped_mm"), reason="requires torch._grouped_mm"
    )
    def test_grouped_mm_shape_rule(self):
        n_experts, tokens, k, n = 4, 64, 512, 64
        gen = torch.Generator(device="cpu").manual_seed(3)
        weight = (
            torch.randn(n_experts, k, n, generator=gen)
            .to("cuda", torch.bfloat16)
            .requires_grad_(True)
        )
        proj = torch.randn(n, k, generator=gen).to("cuda", torch.bfloat16)
        offs = torch.tensor([16, 32, 48, 64], device="cuda", dtype=torch.int32)
        tokens_in = torch.randn(tokens, k, generator=gen).to("cuda", torch.bfloat16)

        def expert_block(h):
            return F.relu(torch._grouped_mm(h, weight, offs=offs)) @ proj

        def run(context_fn=None):
            weight.grad = None
            h = tokens_in
            for _ in range(2):
                if context_fn is None:
                    h = expert_block(h)
                else:
                    h = checkpoint(
                        expert_block, h, use_reentrant=False, context_fn=context_fn
                    )
            h.float().pow(2).mean().backward()
            return weight.grad.clone()

        baseline = run()
        state = SacPolicyState()
        grad = run(build_sac_context_fn(state=state, save_matmul_min_k=512))
        torch.testing.assert_close(baseline, grad)
        assert state.rule_saves["shape"] == 2
        assert state.saved_op_names == {"shape:aten::_grouped_mm"}

    @requires_cuda
    def test_bnb_gemm_4bit_module_rule(self):
        bnb = pytest.importorskip("bitsandbytes")
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(0)
        base = _TinyMlpStack().to(torch.bfloat16)
        for layer in base.layers:
            quantized = bnb.nn.Linear4bit(
                256,
                64,
                bias=False,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
            )
            quantized.weight = bnb.nn.Params4bit(
                layer.down.weight.data.clone(), requires_grad=False, quant_type="nf4"
            )
            layer.down = quantized
        base = base.to("cuda")
        state = SacPolicyState()
        install_module_scope_hooks(base, state, ["down"])
        model = get_peft_model(
            base, LoraConfig(r=4, target_modules=["down"], init_lora_weights=False)
        )

        baseline = _run_stack(model, 64, "cuda", dtype=torch.bfloat16)
        context_fn = build_sac_context_fn(state=state, save_modules=["down"])
        actual = _run_stack(model, 64, "cuda", context_fn, dtype=torch.bfloat16)
        _assert_grads_match(baseline, actual)
        assert state.rule_saves["module:down"] == 2
        assert state.saved_op_names == {"module:down:bitsandbytes::gemm_4bit"}


class TestRegisteredSaves:
    SDPA_OP = torch.ops.aten._scaled_dot_product_flash_attention.default

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        selective_checkpointing.clear_registered_saves()
        yield
        selective_checkpointing.clear_registered_saves()

    def test_mandatory_op_saved_regardless_of_save_list(self):
        selective_checkpointing.register_mandatory_save(ops={"aten::topk"})
        for save in (None, [], ["aten::mm"]):
            policy = build_sac_policy(save)
            assert (
                policy(None, torch.ops.aten.topk.default) == CheckpointPolicy.MUST_SAVE
            )
        assert (
            build_sac_policy([])(None, torch.ops.aten.sort.default)
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_mandatory_op_saved_on_recompute_layer_types(self):
        selective_checkpointing.register_mandatory_save(ops={"aten::topk"})
        state = SacPolicyState()
        policy = build_sac_policy(["attention"], state)
        state.current_layer_type = "sliding_attention"
        assert policy(None, torch.ops.aten.topk.default) == CheckpointPolicy.MUST_SAVE
        assert policy(None, self.SDPA_OP) == CheckpointPolicy.PREFER_RECOMPUTE
        assert state.registered_op_names == {"aten::topk"}
        assert not state.saved_op_names

    def test_namespace_save(self):
        selective_checkpointing.register_mandatory_save(
            namespaces={"_c10d_functional", "axolotl"}
        )
        policy = build_sac_policy([])
        assert (
            policy(None, torch.ops._c10d_functional.wait_tensor.default)
            == CheckpointPolicy.MUST_SAVE
        )
        assert (
            policy(None, _FakeOp("axolotl::some_kernel")) == CheckpointPolicy.MUST_SAVE
        )
        assert (
            policy(None, _FakeOp("flash_attn::_flash_attn_forward"))
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_cpu_copy_save(self):
        op = torch.ops.aten._to_copy.default
        assert (
            build_sac_policy([])(None, op, device=torch.device("cpu"))
            == CheckpointPolicy.PREFER_RECOMPUTE
        )
        selective_checkpointing.register_mandatory_save(cpu_copies=True)
        policy = build_sac_policy([])
        assert (
            policy(None, op, device=torch.device("cpu")) == CheckpointPolicy.MUST_SAVE
        )
        assert policy(None, op, device="cpu") == CheckpointPolicy.MUST_SAVE
        assert (
            policy(None, op, device=torch.device("cuda", 0))
            == CheckpointPolicy.PREFER_RECOMPUTE
        )
        assert (
            policy(None, op, dtype=torch.bfloat16) == CheckpointPolicy.PREFER_RECOMPUTE
        )
        assert (
            policy(None, torch.ops.aten.mm.default, device="cpu")
            == CheckpointPolicy.PREFER_RECOMPUTE
        )

    def test_preferred_save_and_mandatory_precedence(self):
        selective_checkpointing.register_preferred_save(
            ops={"axolotl::ep_all_to_all_single"}, namespaces={"mylib"}
        )
        policy = build_sac_policy([])
        assert (
            policy(None, _FakeOp("axolotl::ep_all_to_all_single"))
            == CheckpointPolicy.PREFER_SAVE
        )
        assert policy(None, _FakeOp("mylib::op")) == CheckpointPolicy.PREFER_SAVE
        selective_checkpointing.register_mandatory_save(namespaces={"mylib"})
        assert policy(None, _FakeOp("mylib::op")) == CheckpointPolicy.MUST_SAVE

    def test_clear_registered_saves(self):
        selective_checkpointing.register_mandatory_save(
            ops={"aten::topk"}, cpu_copies=True
        )
        selective_checkpointing.register_preferred_save(namespaces={"axolotl"})
        selective_checkpointing.clear_registered_saves()
        policy = build_sac_policy([])
        assert (
            policy(None, torch.ops.aten.topk.default)
            == CheckpointPolicy.PREFER_RECOMPUTE
        )
        assert (
            selective_checkpointing.registered_save_policy(
                torch.ops.aten._to_copy.default, {"device": "cpu"}
            )
            is None
        )

    def test_explicit_empty_save_skips_attention(self):
        assert build_sac_policy([])(None, self.SDPA_OP) == (
            CheckpointPolicy.PREFER_RECOMPUTE
        )
        assert build_sac_policy(None)(None, self.SDPA_OP) == CheckpointPolicy.MUST_SAVE

    def test_apply_with_empty_save_injects_policy(self):
        model = TestEnableWrap._FakeModel()
        apply_selective_checkpointing(model, save=[])
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
        assert model.seen_kwargs["use_reentrant"] is False
        assert callable(model.seen_kwargs["context_fn"])

    def test_registered_save_precedes_rules(self):
        selective_checkpointing.register_preferred_save(ops={"aten::mm"})
        policy = build_sac_policy([], save_matmul_min_k=1)
        assert (
            policy(None, MM, torch.empty(4, 8), torch.empty(8, 4))
            == CheckpointPolicy.PREFER_SAVE
        )
        assert (
            policy(None, ADDMM, torch.empty(4), torch.empty(4, 8), torch.empty(8, 4))
            == CheckpointPolicy.MUST_SAVE
        )

    def test_empty_save_with_rules_saves_only_rule_matmuls(self):
        policy = build_sac_policy([], save_matmul_min_k=8)
        assert policy(None, SDPA) == CheckpointPolicy.PREFER_RECOMPUTE
        assert (
            policy(None, MM, torch.empty(4, 8), torch.empty(8, 4))
            == CheckpointPolicy.MUST_SAVE
        )

    def test_empty_save_does_not_warn_no_match(self, sac_log):
        context_fn = build_sac_context_fn([])
        for _ in range(_NO_MATCH_WARN_REGIONS):
            context_fn()
        assert not [r for r in sac_log.records if r.levelno == logging.WARNING]

    def test_fsdp_activation_checkpointing_carries_registered_saves(self):
        from axolotl.monkeypatch.accelerate.fsdp2 import (
            _activation_checkpoint_wrapper_fn,
        )

        assert "context_fn" not in _activation_checkpoint_wrapper_fn().keywords
        selective_checkpointing.register_mandatory_save(ops={"aten::topk"})
        assert callable(_activation_checkpoint_wrapper_fn().keywords["context_fn"])


class TestMandatorySaveRecompute:
    """A router that normalises its ``topk`` values in place (Qwen3-MoE, Mixtral) under
    a policy that must save ``topk``: backward must not re-run it and grads must match."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        selective_checkpointing.clear_registered_saves()
        yield
        selective_checkpointing.clear_registered_saves()

    @staticmethod
    def _router_block(x, w_router, w_out):
        probs = torch.softmax(x @ w_router, dim=-1)
        vals, idx = torch.topk(probs, 2, dim=-1)
        vals /= vals.sum(dim=-1, keepdim=True)
        gathered = torch.gather(probs, 1, idx) * vals
        return (gathered.sum(-1, keepdim=True) * (x @ w_out)).sum()

    def _run(self, context_fn):
        from torch.utils._python_dispatch import TorchDispatchMode

        class _CountTopk(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.topk.default:
                    self.count += 1
                return func(*args, **(kwargs or {}))

        gen = torch.Generator().manual_seed(0)
        inputs = [
            torch.randn(shape, generator=gen).requires_grad_(True)
            for shape in ((16, 8), (8, 6), (8, 4))
        ]
        fwd, bwd = _CountTopk(), _CountTopk()
        with fwd:
            if context_fn is None:
                loss = self._router_block(*inputs)
            else:
                loss = checkpoint(
                    self._router_block,
                    *inputs,
                    use_reentrant=False,
                    context_fn=context_fn,
                )
        with bwd:
            loss.backward()
        return fwd.count, bwd.count, [t.grad for t in inputs]

    def test_topk_saved_and_grads_match(self):
        _, _, ref = self._run(None)
        _, replayed, _ = self._run(build_sac_context_fn([]))
        assert replayed == 1

        selective_checkpointing.register_mandatory_save(ops={"aten::topk"})
        for save in ([], None):
            fwd, bwd, grads = self._run(build_sac_context_fn(save))
            assert (fwd, bwd) == (1, 0)
            for got, want in zip(grads, ref, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_offload_engine_topk_saved_and_grads_match(self):
        from axolotl.monkeypatch.selective_checkpointing_offload import (
            build_sac_offload_context_fn,
        )

        _, _, ref = self._run(None)
        selective_checkpointing.register_mandatory_save(ops={"aten::topk"})
        fwd, bwd, grads = self._run(build_sac_offload_context_fn(save=[]))
        assert (fwd, bwd) == (1, 0)
        for got, want in zip(grads, ref, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
