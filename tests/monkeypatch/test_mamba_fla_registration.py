"""Optional FLA imports must not replace the HF models used for Mamba packing."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Mamba2Config, MambaConfig

from axolotl.loaders.utils import load_model_config
from axolotl.utils.dict import DictDefault


@pytest.mark.parametrize("config_cls", [MambaConfig, Mamba2Config])
def test_native_mamba_save_reload_after_fla_import(config_cls, tmp_path):
    pytest.importorskip("fla")
    config = config_cls(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        state_size=4,
        expand=2,
        num_heads=4,
        head_dim=16,
        n_groups=1,
        tie_word_embeddings=True,
    )
    config.save_pretrained(tmp_path)
    loaded = load_model_config(DictDefault(base_model=str(tmp_path)))
    assert type(loaded) is config_cls
    model = AutoModelForCausalLM.from_config(loaded)
    assert type(model).__module__.startswith("transformers.models.")
    assert model.lm_head.weight is model.backbone.embeddings.weight
    model.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path, config=loaded)
    assert restored.lm_head.weight is restored.backbone.embeddings.weight
    torch.testing.assert_close(restored.lm_head.weight, model.lm_head.weight)
    # Keep FLA's global registrations available to its own callers.
    assert type(AutoConfig.from_pretrained(tmp_path)).__module__.startswith("fla.")
