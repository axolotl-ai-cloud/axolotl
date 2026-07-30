"""M5 TernaryExperts: fused 3-D expert stacks quantize through a parametrization."""

import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig

from axolotl.integrations.ternary.modules import TernaryExperts, iter_quantized_modules
from axolotl.integrations.ternary.swap import convert_model
from axolotl.utils.dict import DictDefault


def _moe():
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
    )
    return AutoModelForCausalLM.from_config(config)


def _cfg():
    return DictDefault(
        {
            "ternary": {
                "weight_scale": "learnable",
                "lambda_schedule": "none",
                "strict_enumeration": False,
                "target_modules": [".*_proj$"],
            }
        }
    )


def _expert_stacks(model):
    return [
        (name, attr, param)
        for name, module in model.named_modules()
        if "expert" in f"{name}.{type(module).__name__}".lower()
        for attr, param in module.named_parameters(recurse=False)
        if param.ndim == 3
    ]


def test_conversion_parametrizes_every_stack_and_manifests_them():
    model = _moe()
    stacks_before = _expert_stacks(model)
    if not stacks_before:
        import pytest

        pytest.skip("this transformers Qwen3Moe keeps experts unfused")

    manifest = convert_model(model, _cfg())
    experts = [e for e in manifest.entries if e.kind == "experts"]
    assert len(experts) == len(stacks_before)
    for entry in experts:
        assert entry.parameter_key() == entry.name
    quantizers = [m for _, m in model.named_modules() if isinstance(m, TernaryExperts)]
    assert len(quantizers) == len(stacks_before)


def test_the_read_is_quantized_and_lambda_reaches_the_quantizer():
    model = _moe()
    if not _expert_stacks(model):
        import pytest

        pytest.skip("unfused experts")
    convert_model(model, _cfg())
    name, quantizer = next(
        (n, m)
        for n, m in iter_quantized_modules(model)
        if isinstance(m, TernaryExperts)
    )
    holder_name = name.rsplit(".parametrizations", 1)[0]
    holder = model.get_submodule(holder_name)
    # read through the parametrization: values sit on a per-expert ternary grid
    quantizer.set_lambda(1.0)
    original = holder.parametrizations[list(holder.parametrizations)[0]].original
    quantized = getattr(holder, list(holder.parametrizations)[0])
    scales = quantizer.scale.exp()
    per_slice_values = (quantized / scales.to(quantized.dtype)).round().unique()
    assert set(per_slice_values.tolist()) <= {-1.0, 0.0, 1.0}
    assert not torch.equal(original, quantized)


def test_forward_and_backward_flow_through_experts():
    model = _moe()
    if not _expert_stacks(model):
        import pytest

        pytest.skip("unfused experts")
    convert_model(model, _cfg())
    ids = torch.randint(0, 256, (2, 8))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    quantizer = next(
        m for _, m in model.named_modules() if isinstance(m, TernaryExperts)
    )
    assert quantizer.scale.grad is not None
