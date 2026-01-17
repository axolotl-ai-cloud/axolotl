import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from importlib import util as importlib_util
from pathlib import Path

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
    from transformers import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

    config = Llama4TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_local_experts=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts_per_tok=2,
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
        self.assertFalse(torch.allclose(block._afb_ema, torch.zeros_like(block._afb_ema)))

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
        self.assertFalse(torch.allclose(layer._afb_ema, torch.zeros_like(layer._afb_ema)))

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
        self.assertTrue(torch.allclose(block._afb_bias, torch.zeros_like(block._afb_bias)))

        _step()
        self.assertTrue(torch.allclose(block._afb_bias, torch.zeros_like(block._afb_bias)))

        # Third step exceeds warmup -> bias should update.
        _step()
        self.assertGreater(torch.count_nonzero(block._afb_bias), 0)

    def test_mixtral_adapter_respects_native_forward(self):
        model, layer = _build_mixtral_model()
        layer.jitter_noise = 0.0  # avoid stochasticity for comparison

        hidden_dim = layer.config.hidden_size
        hidden = torch.randn(2, 3, hidden_dim)
        baseline_out, baseline_logits = layer(hidden.clone())

        cfg = _cfg()
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        patched_out, patched_logits = layer(hidden.clone())
        self.assertTrue(torch.allclose(baseline_out, patched_out))
        self.assertTrue(torch.allclose(baseline_logits, patched_logits))
        self.assertGreater(torch.count_nonzero(layer._afb_counts), 0)
        _run_callback(plugin, cfg)

    def test_ep_group_resolution_deferred_until_dist_ready(self):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

        model, block = _build_bailing_model()
        cfg = _cfg(moe_bias_sync_group="ep", expert_parallel_size=1)
        plugin = AuxFreeMoEPlugin()
        plugin.post_model_build(cfg, model)

        self.assertIsNotNone(plugin._shim)
        self.assertIsNone(plugin._shim.ep_group)

        tmp_init = tempfile.NamedTemporaryFile(delete=False)
        tmp_init.close()
        init_method = f"file://{tmp_init.name}"
        dist.init_process_group(backend="gloo", init_method=init_method, world_size=1, rank=0)
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
        layer.jitter_noise = 0.0
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


if __name__ == "__main__":
    unittest.main()
