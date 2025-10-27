import sys
import unittest
from types import SimpleNamespace

import torch
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


def _run_callback(plugin, cfg):
    callbacks = plugin.add_callbacks_post_trainer(cfg, trainer=SimpleNamespace())
    assert callbacks, "expected aux-free callback to be registered"
    callback = callbacks[0]
    dummy = SimpleNamespace()
    callback.on_step_end(args=dummy, state=dummy, control=dummy)


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


if __name__ == "__main__":
    unittest.main()
