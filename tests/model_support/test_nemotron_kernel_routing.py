"""Native Nemotron attention routes projections through attached LoRA kernels."""

import glob
import types

import pytest
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from axolotl.model_support.nemotron_diffusion.compat import resolve_nemotron_model_class

from tests.native_source_fixtures import (
    native_source_fixture_path,
    validate_native_source_fixture,
)

_CACHED_SNAPSHOT = (
    "/mnt/data/hf_cache/hub/models--nvidia--Nemotron-Labs-Diffusion-3B/snapshots/"
    "0d51902da1f8869f83413ce642fab402fa5641e0"
)


def _source():
    source = native_source_fixture_path("nemotron")
    if source is not None:
        return source
    for candidate in glob.glob(_CACHED_SNAPSHOT):
        try:
            return validate_native_source_fixture("nemotron", candidate)
        except (FileNotFoundError, ValueError):
            continue
    pytest.skip("native Nemotron source fixture unavailable")


@pytest.fixture
def model():
    source = _source()
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        str(source),
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        use_cache=False,
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    return resolve_nemotron_model_class(str(source))(config).eval()


def _attach_counting_kernels(model):
    counts = []
    for layer in model.encoder.layers:
        count = {"qkv": 0, "o": 0}

        def apply_qkv(self, hidden_states, count=count):
            count["qkv"] += 1
            return (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            )

        def apply_o(self, hidden_states, count=count):
            count["o"] += 1
            return self.o_proj(hidden_states)

        attn = layer.self_attn
        attn.apply_qkv = types.MethodType(apply_qkv, attn)
        attn.apply_o = types.MethodType(apply_o, attn)
        counts.append(count)
    return counts


def _forward(model):
    ids = torch.tensor([[7, 8, 9, 10, 11]])
    mask = torch.ones(1, 1, ids.shape[1], ids.shape[1], dtype=torch.bool)
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask).logits


def test_attached_qkv_and_o_kernels_are_invoked_once_per_layer(model):
    assert all(
        type(layer.self_attn).__name__ == "MaskAwareAttention"
        for layer in model.encoder.layers
    )
    counts = _attach_counting_kernels(model)
    routed = _forward(model)
    assert counts == [{"qkv": 1, "o": 1}] * model.config.num_hidden_layers
    for layer in model.encoder.layers:
        del layer.self_attn.apply_qkv
        del layer.self_attn.apply_o
    torch.testing.assert_close(routed, _forward(model), atol=1e-6, rtol=0)


def test_plain_projection_path_without_kernels(model):
    assert not any(
        hasattr(layer.self_attn, "apply_qkv") for layer in model.encoder.layers
    )
    logits = _forward(model)
    assert logits.shape == (1, 5, model.config.vocab_size)
    assert torch.isfinite(logits).all()
