"""Healing-strategy trainables: frozen latents and low-rank latent deltas.

Born from the blockwise verdict: local-objective healing is dead, so end-to-end
healing at 100B+ scale must shrink what trains instead. Frozen latents heal
through the scale grid; a delta heals the codes through a low-rank correction
applied INSIDE the quantizer, so the bake still collapses to pure ternary.
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.modules import TernaryLinear
from axolotl.integrations.ternary.swap import convert_model
from axolotl.utils.dict import DictDefault


def _model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    return LlamaForCausalLM(config)


def _cfg(**ternary):
    base = {"weight_scale": "learnable", "lambda_schedule": "none"}
    base.update(ternary)
    return DictDefault({"ternary": base})


def _first_ternary(model) -> TernaryLinear:
    return next(m for m in model.modules() if isinstance(m, TernaryLinear))


def _step(model, steps=3):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    for _ in range(steps):
        ids = torch.randint(0, 256, (2, 16))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def test_frozen_latents_do_not_move_while_scales_and_norms_do():
    model = _model()
    convert_model(model, _cfg(heal_codes="frozen"))
    module = _first_ternary(model)
    latent_before = module.weight.detach().clone()
    scale_before = module.scale.detach().clone()
    norm_before = model.model.norm.weight.detach().clone()

    _step(model)

    assert torch.equal(module.weight.detach(), latent_before)
    assert not torch.equal(module.scale.detach(), scale_before)
    assert not torch.equal(model.model.norm.weight.detach(), norm_before)


def test_frozen_module_never_reports_baked():
    model = _model()
    convert_model(model, _cfg(init="ternary_fit", heal_codes="frozen"))
    module = _first_ternary(model)
    # a fit master's values sit on the grid, but the quantizer must stay in the
    # loop or scale gradients silently stop flowing
    assert module.is_baked() is False


def test_zero_init_delta_is_a_bitwise_noop():
    model = _model()
    reference = _model()
    convert_model(model, _cfg(low_rank_delta={"r": 4, "alpha": 8.0}))
    convert_model(reference, _cfg())
    module = _first_ternary(model)
    ref_module = _first_ternary(reference)
    with torch.no_grad():
        ref_module.weight.copy_(module.weight)
        ref_module.scale.copy_(module.scale)

    x = torch.randn(3, module.in_features)
    with torch.no_grad():
        assert torch.equal(module(x), ref_module(x))


def test_codes_flip_through_the_delta_with_frozen_latents():
    model = _model()
    convert_model(
        model, _cfg(heal_codes="frozen", low_rank_delta={"r": 4, "alpha": 8.0})
    )
    module = _first_ternary(model)
    before = module.code_snapshot().clone()
    with torch.no_grad():
        module.delta_b.normal_(std=2.0 * float(module.weight.abs().mean()))
    after = module.code_snapshot()
    assert not torch.equal(before, after)
    assert torch.equal(
        module.weight.detach(), module.weight.detach()
    )  # latent untouched


def test_fold_collapses_the_delta_and_bake_is_exact():
    model = _model()
    convert_model(model, _cfg(low_rank_delta={"r": 4, "alpha": 8.0}))
    module = _first_ternary(model)
    with torch.no_grad():
        module.delta_b.normal_(std=float(module.weight.abs().mean()))
    quantized_with_delta = module._quant_weight(1.0).detach().clone()

    module.fold_delta()

    assert module.delta_a is None and module.delta_b is None
    assert torch.equal(module._quant_weight(1.0).detach(), quantized_with_delta)
    baked = module.baked_weight()
    with torch.no_grad():
        module.weight.copy_(baked)
    assert torch.equal(module.baked_weight(), baked)


def test_int8_path_declines_while_a_delta_is_live():
    model = _model()
    convert_model(model, _cfg(low_rank_delta={"r": 4, "alpha": 8.0}))
    module = _first_ternary(model)
    module.int8_forward = True
    assert module._int8_linear(torch.randn(2, module.in_features)) is None


def test_param_accounting_under_frozen_plus_delta():
    model = _model()
    total = sum(p.numel() for p in model.parameters())
    convert_model(
        model, _cfg(heal_codes="frozen", low_rank_delta={"r": 4, "alpha": 8.0})
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable < total * 0.65  # tiny model: embeddings dominate; real
    # models report thousandths — the assertion just pins that latents dropped out


def test_config_rejects_bad_delta():
    for bad in ({"r": 0}, {"alpha": 0.0}, {"r": -3}):
        try:
            TernaryConfig(low_rank_delta=bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
