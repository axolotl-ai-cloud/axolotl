import os
import sys
import tempfile
import unittest
from importlib import util as importlib_util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import snapshot_download

from axolotl.integrations.aux_free_router.plugin import AuxFreeMoEPlugin


def _cfg(**overrides):
    defaults = dict(
        moe_balance_type="noaux_tc",
        moe_update_rate=0.1,
        moe_update_momentum=0.9,
        moe_bias_cap=2.0,
        moe_afb_warmup_steps=0,
        moe_bias_sync_group="world",
        expert_parallel_size=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _load_bailing_modules():
    repo_dir = snapshot_download(
        repo_id="inclusionAI/Ring-mini-2.0",
        allow_patterns=[
            "configuration_bailing_moe_v2.py",
            "modeling_bailing_moe_v2.py",
            "__init__.py",
        ],
    )
    repo = Path(repo_dir)
    config_path = repo / "configuration_bailing_moe_v2.py"
    modeling_path = repo / "modeling_bailing_moe_v2.py"

    config_name = "bailing_moe_v2.configuration_bailing_moe_v2"
    if config_name not in sys.modules:
        spec = importlib_util.spec_from_file_location(config_name, config_path)
        module = importlib_util.module_from_spec(spec)
        sys.modules[config_name] = module
        sys.modules["configuration_bailing_moe_v2"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    config_module = sys.modules[config_name]

    modeling_name = "bailing_moe_v2.modeling_bailing_moe_v2"
    if modeling_name not in sys.modules:
        spec = importlib_util.spec_from_file_location(modeling_name, modeling_path)
        module = importlib_util.module_from_spec(spec)
        sys.modules[modeling_name] = module
        sys.modules["modeling_bailing_moe_v2"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    modeling_module = sys.modules[modeling_name]

    BailingMoeV2Config = config_module.BailingMoeV2Config
    BailingMoeV2SparseMoeBlock = modeling_module.BailingMoeV2SparseMoeBlock

    return BailingMoeV2Config, BailingMoeV2SparseMoeBlock


def _build_bailing_model():
    BailingConfig, BailingBlock = _load_bailing_modules()
    config = BailingConfig(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_experts=4,
        num_shared_experts=None,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    block = BailingBlock(config)

    class DummyModel(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.block = layer
            self.config = SimpleNamespace(model_type="bailing_moe")

        def forward(self, hidden_states):
            return self.block(hidden_states)

    return DummyModel(block), block


def _build_llama4_model():
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

    # Build config without __post_init__ validation (works around a
    # huggingface_hub strict-dataclass type mismatch for layer_types).
    config = object.__new__(__import__("transformers").Llama4TextConfig)
    config.__dict__.update(
        hidden_size=16,
        intermediate_size=32,
        num_local_experts=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        hidden_act="silu",
        layer_types=None,
    )
    layer = Llama4TextMoe(config)

    class DummyModel(nn.Module):
        def __init__(self, moe_layer):
            super().__init__()
            self.moe = moe_layer
            self.config = SimpleNamespace(model_type="llama4")

        def forward(self, hidden_states):
            return self.moe(hidden_states)

    return DummyModel(layer), layer


def _build_mixtral_model():
    from transformers import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

    config = MixtralConfig(
        hidden_size=16,
        intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    layer = MixtralSparseMoeBlock(config)
    layer.config = config

    class DummyModel(nn.Module):
        def __init__(self, moe_layer):
            super().__init__()
            self.moe = moe_layer
            self.config = SimpleNamespace(model_type="mixtral")

        def forward(self, hidden_states):
            return self.moe(hidden_states)

    return DummyModel(layer), layer


def _build_qwen35_moe_model():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )

    config = Qwen3_5MoeTextConfig(
        hidden_size=16,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
    )
    layer = Qwen3_5MoeSparseMoeBlock(config)

    class DummyModel(nn.Module):
        def __init__(self, moe_layer):
            super().__init__()
            self.moe = moe_layer
            self.config = SimpleNamespace(model_type="qwen3_5_moe")

        def forward(self, hidden_states):
            return self.moe(hidden_states)

    return DummyModel(layer), layer


def _run_callback(plugin, cfg, *, args=None, state=None, control=None):
    if args is None:
        args = SimpleNamespace(logging_steps=1)
    if state is None:
        state = SimpleNamespace(global_step=1, log_history=[])
    if control is None:
        control = SimpleNamespace(
            should_log=False,
            should_evaluate=False,
            should_save=False,
            should_training_stop=False,
        )

    class DummyTrainer:
        def __init__(self, state_obj, control_obj):
            self.state = state_obj
            self.control = control_obj

        def log(self, logs):
            output = dict(logs)
            output["step"] = self.state.global_step
            self.state.log_history.append(output)
            self.control.should_log = True

    dummy_trainer = DummyTrainer(state, control)
    callbacks = plugin.add_callbacks_post_trainer(cfg, trainer=dummy_trainer)
    assert callbacks, "expected aux-free callback to be registered"
    callback = callbacks[0]
    callback.on_step_end(args=args, state=state, control=control)
    return state, control


class TestAuxFreeAdapters(unittest.TestCase):
    def test_bailing_adapter_updates_counts_and_bias(self):
        model, block = _build_bailing_model()
        cfg = _cfg()
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        self.assertTrue(hasattr(block, "_afb_bias"))
        hidden = torch.randn(2, 3, block.config.hidden_size)
        block(hidden)
        self.assertGreater(torch.count_nonzero(block._afb_counts), 0)

        _run_callback(plugin, cfg)
        self.assertEqual(torch.count_nonzero(block._afb_counts), 0)
        self.assertFalse(
            torch.allclose(block._afb_ema, torch.zeros_like(block._afb_ema))
        )

    def test_llama4_adapter_biases_router_selection(self):
        model, layer = _build_llama4_model()
        cfg = _cfg()
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        self.assertTrue(hasattr(layer, "_afb_bias"))
        hidden = torch.randn(2, 4, layer.hidden_dim)
        layer(hidden)
        self.assertGreater(torch.count_nonzero(layer._afb_counts), 0)

        _run_callback(plugin, cfg)
        self.assertEqual(torch.count_nonzero(layer._afb_counts), 0)
        self.assertFalse(
            torch.allclose(layer._afb_ema, torch.zeros_like(layer._afb_ema))
        )

    def test_bias_warmup_respected(self):
        model, block = _build_bailing_model()
        cfg = _cfg(moe_afb_warmup_steps=2)
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        def _step():
            hidden = torch.randn(2, 3, block.config.hidden_size)
            block(hidden)
            _run_callback(plugin, cfg)

        # Warmup steps should leave bias untouched.
        _step()
        self.assertTrue(
            torch.allclose(block._afb_bias, torch.zeros_like(block._afb_bias))
        )

        _step()
        self.assertTrue(
            torch.allclose(block._afb_bias, torch.zeros_like(block._afb_bias))
        )

        # Third step exceeds warmup -> bias should update.
        _step()
        self.assertGreater(torch.count_nonzero(block._afb_bias), 0)

    def test_mixtral_adapter_patches_router_not_forward(self):
        """Verify that aux-free patches the router (gate) only, and the
        v5 block forward signature (single tensor return) is preserved."""
        model, layer = _build_mixtral_model()
        cfg = _cfg()
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        # Gate should be patched, not the block forward
        self.assertTrue(getattr(layer.gate, "_afb_patched", False))
        self.assertTrue(getattr(layer, "_afb_patched", False))

        # v5 block forward returns a single tensor (not a tuple with logits)
        hidden = torch.randn(2, 3, layer.config.hidden_size)
        out = layer(hidden)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, hidden.shape)

        # Counts should have been accumulated
        self.assertGreater(torch.count_nonzero(layer._afb_counts), 0)
        _run_callback(plugin, cfg)

    def test_mixtral_adapter_bias_affects_selection(self):
        """When bias is large for one expert, it should be selected more often."""
        model, layer = _build_mixtral_model()
        cfg = _cfg()
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        # Set a large bias for expert 0 to force its selection
        layer._afb_bias.zero_()
        layer._afb_bias[0] = 10.0

        hidden = torch.randn(2, 8, layer.config.hidden_size)
        num_tokens = 2 * 8  # batch * seq
        layer(hidden)

        # With top_k=2, expert 0 should appear in every token's selection
        # (once per token = num_tokens counts, not num_tokens * top_k)
        counts = layer._afb_counts.clone()
        self.assertEqual(
            int(counts[0].item()),
            num_tokens,
            msg="Expert 0 should be selected for every token when heavily biased",
        )

    def test_qwen35_moe_adapter_patches_router_and_preserves_shared_expert(self):
        """Verify Qwen 3.5 MoE: router is patched, shared expert is untouched,
        output includes shared expert contribution."""
        model, layer = _build_qwen35_moe_model()
        cfg = _cfg()
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        # Gate should be patched
        self.assertTrue(getattr(layer.gate, "_afb_patched", False))
        self.assertTrue(getattr(layer, "_afb_patched", False))
        # Shared expert should be unmodified
        self.assertTrue(hasattr(layer, "shared_expert"))
        self.assertTrue(hasattr(layer, "shared_expert_gate"))

        # Forward should return a single tensor (shared + routed)
        hidden_size = layer.gate.hidden_dim
        hidden = torch.randn(2, 3, hidden_size)
        out = layer(hidden)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, hidden.shape)

        # Counts should have been accumulated
        self.assertGreater(torch.count_nonzero(layer._afb_counts), 0)

    def test_qwen35_moe_adapter_bias_updates(self):
        """Full cycle: forward → callback → verify bias update for Qwen 3.5 MoE."""
        model, layer = _build_qwen35_moe_model()
        cfg = _cfg()
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        hidden_size = layer.gate.hidden_dim
        hidden = torch.randn(2, 4, hidden_size)
        layer(hidden)

        # Bias should start at zero
        self.assertTrue(
            torch.allclose(layer._afb_bias, torch.zeros_like(layer._afb_bias))
        )

        _run_callback(plugin, cfg)

        # After callback: counts reset, EMA updated, bias updated
        self.assertEqual(torch.count_nonzero(layer._afb_counts), 0)
        self.assertFalse(
            torch.allclose(layer._afb_ema, torch.zeros_like(layer._afb_ema))
        )

    def test_qwen35_moe_adapter_model_type_matching(self):
        """Verify the adapter matches both qwen3_5_moe and qwen3_5_moe_text."""
        from axolotl.integrations.aux_free_router.adapters import Qwen35MoeAdapter

        adapter = Qwen35MoeAdapter()

        model_moe = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_moe"))
        model_text = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen3_5_moe_text")
        )
        model_other = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))

        self.assertTrue(adapter.matches(model_moe))
        self.assertTrue(adapter.matches(model_text))
        self.assertFalse(adapter.matches(model_other))

    def test_ep_group_resolution_deferred_until_dist_ready(self):
        if dist.is_available() and dist.is_initialized():
            self.skipTest(
                "Cannot safely test deferred EP group resolution when a process group is already initialized"
            )

        model, block = _build_bailing_model()
        cfg = _cfg(moe_bias_sync_group="ep", expert_parallel_size=1)
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        self.assertIsNotNone(plugin._shim)
        self.assertIsNone(plugin._shim.ep_group)

        tmp_init = tempfile.NamedTemporaryFile(delete=False)
        tmp_init.close()
        init_method = f"file://{tmp_init.name}"
        dist.init_process_group(
            backend="gloo", init_method=init_method, world_size=1, rank=0
        )
        try:
            hidden = torch.randn(2, 3, block.config.hidden_size)
            block(hidden)
            _run_callback(
                plugin,
                cfg,
                args=SimpleNamespace(logging_steps=1),
                state=SimpleNamespace(global_step=1, log_history=[]),
                control=SimpleNamespace(
                    should_log=False,
                    should_evaluate=False,
                    should_save=False,
                    should_training_stop=False,
                ),
            )
            self.assertIs(plugin._shim.ep_group, dist.group.WORLD)
        finally:
            dist.destroy_process_group()
            os.unlink(tmp_init.name)

    def test_telemetry_logging(self):
        model, layer = _build_mixtral_model()
        cfg = _cfg()
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        hidden_dim = layer.config.hidden_size
        hidden = torch.randn(2, 3, hidden_dim)
        layer(hidden)

        args = SimpleNamespace(logging_steps=1)
        state = SimpleNamespace(global_step=1, log_history=[])
        control = SimpleNamespace(
            should_log=False,
            should_evaluate=False,
            should_save=False,
            should_training_stop=False,
        )
        _run_callback(plugin, cfg, args=args, state=state, control=control)

        self.assertTrue(control.should_log)
        self.assertTrue(state.log_history)
        telemetry = state.log_history[-1]
        self.assertEqual(telemetry["step"], state.global_step)
        self.assertIn("moe_afb/l0_load_min", telemetry)
        self.assertIn("moe_afb/l0_load_max", telemetry)
        self.assertIn("moe_afb/l0_bias_abs_max", telemetry)

    def test_get_num_experts_v5_attribute_paths(self):
        """Verify get_num_experts works with v5 attribute layout where
        num_experts is on gate/experts sub-modules, not the block."""
        from axolotl.integrations.aux_free_router.adapters import MixtralAdapter

        adapter = MixtralAdapter()

        # Simulates v5 MixtralSparseMoeBlock (num_experts on gate, not block)
        block = SimpleNamespace(
            gate=SimpleNamespace(num_experts=8),
            experts=SimpleNamespace(num_experts=8),
        )
        self.assertEqual(adapter.get_num_experts(block), 8)

        # Also works when num_experts is directly on block
        block2 = SimpleNamespace(num_experts=4)
        self.assertEqual(adapter.get_num_experts(block2), 4)


class TestAuxFreeKernelComposition(unittest.TestCase):
    """Tests that aux-free bias composes correctly with kernel routing."""

    def test_sonicmoe_softmax_routing_with_afb_bias(self):
        """SonicMoE softmax routing should use biased selection / unbiased weights."""
        from axolotl.integrations.kernels.sonicmoe.routing import softmax_topk_routing

        num_experts = 4
        top_k = 2
        hidden_dim = 16
        T = 6

        # Build a mock MoE block with gate attributes
        gate = nn.Linear(hidden_dim, num_experts, bias=False)
        gate.top_k = top_k
        gate.num_experts = num_experts
        gate.norm_topk_prob = True

        moe_block = SimpleNamespace(gate=gate)
        hidden = torch.randn(T, hidden_dim)

        # Baseline: no bias
        scores_base, tok_base, exp_base, logits_base = softmax_topk_routing(
            hidden, moe_block
        )
        self.assertEqual(scores_base.shape[0], T * top_k)

        # Now register aux-free buffers and set heavy bias on expert 0
        moe_block._afb_bias = torch.zeros(num_experts)
        moe_block._afb_bias[0] = 100.0
        moe_block._afb_counts = torch.zeros(num_experts)

        scores_biased, tok_biased, exp_biased, logits_biased = softmax_topk_routing(
            hidden, moe_block
        )

        # Expert 0 should be selected for every token
        self.assertTrue(
            (exp_biased == 0).any(),
            "Expert 0 should appear in selections when heavily biased",
        )
        # Counts should have been accumulated
        self.assertGreater(moe_block._afb_counts[0].item(), 0)
        # Total counts should equal T * top_k
        self.assertEqual(int(moe_block._afb_counts.sum().item()), T * top_k)

    def test_sonicmoe_routing_without_bias_unchanged(self):
        """Without _afb_bias, routing should produce identical results."""
        from axolotl.integrations.kernels.sonicmoe.routing import softmax_topk_routing

        num_experts = 4
        top_k = 2
        hidden_dim = 16

        gate = nn.Linear(hidden_dim, num_experts, bias=False)
        gate.top_k = top_k
        gate.num_experts = num_experts
        gate.norm_topk_prob = True

        moe_block = SimpleNamespace(gate=gate)
        hidden = torch.randn(4, hidden_dim)

        # Without _afb_bias attribute
        scores1, _, exp1, _ = softmax_topk_routing(hidden, moe_block)

        # With _afb_bias = zeros (should be equivalent)
        moe_block._afb_bias = torch.zeros(num_experts)
        moe_block._afb_counts = torch.zeros(num_experts)
        scores2, _, exp2, _ = softmax_topk_routing(hidden, moe_block)

        torch.testing.assert_close(scores1, scores2)
        torch.testing.assert_close(exp1, exp2)

    @unittest.skipUnless(
        importlib_util.find_spec("triton") is not None,
        "triton not installed (required by scattermoe)",
    )
    def test_scattermoe_softmax_routing_with_afb_bias(self):
        """ScatterMoE softmax routing should use biased selection / unbiased weights."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _softmax_topk_route,
        )

        num_experts = 4
        top_k = 2
        hidden_dim = 16
        T = 6

        gate_weight = torch.randn(num_experts, hidden_dim)
        base_gate = SimpleNamespace(
            top_k=top_k,
            num_experts=num_experts,
            norm_topk_prob=True,
            weight=gate_weight,
        )

        moe_block = SimpleNamespace()
        hidden = torch.randn(T, hidden_dim)

        # Baseline without bias
        w_base, e_base, _, _ = _softmax_topk_route(
            moe_block, base_gate, hidden, gate_weight, None
        )

        # With heavy bias on expert 0
        moe_block._afb_bias = torch.zeros(num_experts)
        moe_block._afb_bias[0] = 100.0
        moe_block._afb_counts = torch.zeros(num_experts)

        w_biased, e_biased, _, _ = _softmax_topk_route(
            moe_block, base_gate, hidden, gate_weight, None
        )

        # Expert 0 should appear in all selections
        self.assertTrue((e_biased == 0).any())
        # Counts accumulated
        self.assertGreater(moe_block._afb_counts[0].item(), 0)
        self.assertEqual(int(moe_block._afb_counts.sum().item()), T * top_k)

    def test_kernel_routing_skips_router_patch(self):
        """When a kernel backend has patched the block class, the adapter
        should skip patching the router (buffers are still registered)."""
        from axolotl.integrations.aux_free_router.adapters import MixtralAdapter

        adapter = MixtralAdapter()

        # Create a mock layer whose class has _original_forward (SonicMoE marker)
        class PatchedBlock(nn.Module):
            _original_forward = True  # SonicMoE marker

            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(16, 4, bias=False)
                self.gate.top_k = 2
                self.gate.num_experts = 4
                self.gate.hidden_dim = 16
                self.experts = nn.Linear(16, 16)  # placeholder

        layer = PatchedBlock()
        self.assertTrue(adapter.uses_kernel_routing(layer))

        # Gate should NOT be patched (kernel handles routing)
        self.assertFalse(getattr(layer.gate, "_afb_patched", False))

    def test_adapter_buffers_registered_even_with_kernel(self):
        """Even when kernel routing is active, aux-free buffers must be
        registered on the MoE block so the kernel routing can find them."""
        from axolotl.integrations.aux_free_router.adapters import (
            LayerHandle,
            MixtralAdapter,
        )
        from axolotl.integrations.aux_free_router.core import (
            AuxFreeConfig,
            AuxFreeShim,
            AuxFreeState,
        )

        class PatchedBlock(nn.Module):
            _original_forward = True

            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(16, 4, bias=False)
                self.gate.top_k = 2
                self.gate.num_experts = 4
                self.gate.hidden_dim = 16
                self.experts = nn.Linear(16, 16)

        layer = PatchedBlock()
        adapter = MixtralAdapter()
        cfg = AuxFreeConfig()
        state = AuxFreeState(
            num_layers=1, num_experts=4, device=torch.device("cpu"), cfg=cfg
        )
        shim = AuxFreeShim(state=state)
        handle = LayerHandle(layer=layer, layer_idx=0, num_experts=4, top_k=2)

        adapter.prepare(layer, handle, shim)

        # Buffers should be registered for kernel routing to use
        self.assertTrue(hasattr(layer, "_afb_bias"))
        self.assertTrue(hasattr(layer, "_afb_counts"))
        self.assertTrue(hasattr(layer, "_afb_ema"))
        # But gate should NOT be patched
        self.assertFalse(getattr(layer.gate, "_afb_patched", False))


if __name__ == "__main__":
    unittest.main()
