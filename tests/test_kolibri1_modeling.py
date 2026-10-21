"""Kolibri-1 (`kolibri1`): registration, checkpoint layout, and routing."""

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

from axolotl.model_support import get_model_support, get_model_support_for_cfg
from axolotl.model_support.kolibri1 import _weight_conversions, register_kolibri1
from axolotl.utils.dict import DictDefault
from axolotl.utils.weight_conversions import register_weight_conversions

CONFIG_KWARGS = dict(
    vocab_size=64,
    hidden_size=32,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    sliding_window=3,
)


@pytest.fixture(autouse=True)
def register():
    register_kolibri1()
    for key, entries in _weight_conversions().items():
        register_weight_conversions(key, list(entries))


@pytest.fixture(name="model")
def model_fixture():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        AutoConfig.for_model("kolibri1", **CONFIG_KWARGS)
    )
    for layer in model.model.layers:
        layer.mlp.gate.e_score_correction_bias.normal_()
    return model.eval()


def published_layout(model):
    """Per-expert linears and the `moe.router.expert_bias` key the Hub checkpoint ships."""
    checkpoint = {}
    for name, tensor in model.state_dict().items():
        prefix = name.rsplit(".", 1)[0]
        if name.endswith("mlp.experts.gate_up_proj"):
            gate, up = tensor.chunk(2, dim=1)
            for expert in range(tensor.shape[0]):
                checkpoint[f"{prefix}.{expert}.gate_proj.weight"] = gate[expert].clone()
                checkpoint[f"{prefix}.{expert}.up_proj.weight"] = up[expert].clone()
        elif name.endswith("mlp.experts.down_proj"):
            for expert in range(tensor.shape[0]):
                checkpoint[f"{prefix}.{expert}.down_proj.weight"] = tensor[
                    expert
                ].clone()
        elif name.endswith("mlp.gate.e_score_correction_bias"):
            bias_name = name.replace(
                "mlp.gate.e_score_correction_bias", "moe.router.expert_bias"
            )
            checkpoint[bias_name] = tensor.clone()
        else:
            checkpoint[name] = tensor.clone()
    return checkpoint


def test_matched_by_checkpoint_model_type(tmp_path):
    support = get_model_support("kolibri1")
    finetune = tmp_path / "my-finetune"
    AutoConfig.for_model("kolibri1", **CONFIG_KWARGS).save_pretrained(finetune)
    lookalike = tmp_path / "kolibri-lookalike"
    AutoConfig.for_model("llama").save_pretrained(lookalike)

    assert get_model_support_for_cfg(DictDefault(base_model=str(finetune))) is support
    assert get_model_support_for_cfg(DictDefault(base_model=str(lookalike))) is None


def test_experts_implementation_is_settable(model):
    assert model.config._experts_implementation == "grouped_mm"


def test_router_aux_loss_rejected():
    assert AutoConfig.for_model("kolibri1").router_aux_loss_coef == 0.0
    with pytest.raises(ValueError, match="output_router_logits"):
        AutoConfig.for_model("kolibri1", output_router_logits=True)


def test_rope_only_on_sliding_layers(model):
    assert [layer.self_attn.use_rope for layer in model.model.layers] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert model.model.layers[4].self_attn.sliding_window is None


def test_published_checkpoint_round_trips(model, tmp_path):
    save_file(published_layout(model), tmp_path / "model.safetensors")
    model.config.save_pretrained(tmp_path)

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    expected = model.state_dict()
    assert reloaded.state_dict().keys() == expected.keys()
    for name, tensor in reloaded.state_dict().items():
        assert torch.equal(tensor, expected[name]), name

    out_dir = tmp_path / "saved"
    reloaded.save_pretrained(out_dir)
    assert (
        load_file(out_dir / "model.safetensors").keys()
        == published_layout(model).keys()
    )


def test_bias_steers_selection_but_not_weights(model):
    gate = model.model.layers[0].mlp.gate
    gate.e_score_correction_bias.zero_()
    gate.e_score_correction_bias[0] = 1e4
    hidden = torch.randn(6, CONFIG_KWARGS["hidden_size"])

    logits, weights, indices = gate(hidden)

    assert (indices == 0).any(dim=-1).all()
    assert torch.allclose(weights, torch.sigmoid(logits.gather(1, indices)))


def test_forward_backward(model):
    model.train()
    input_ids = torch.randint(0, CONFIG_KWARGS["vocab_size"], (2, 12))

    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()

    assert torch.isfinite(loss)
    assert model.model.layers[0].mlp.gate.weight.grad.abs().sum() > 0
