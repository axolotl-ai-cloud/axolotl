"""CPU-only tests for `TernaryLinear`: fake-quant forward, STE, bake, state dict."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary.modules import (
    TernaryLinear,
    iter_ternary_modules,
    set_act_quant_sharing,
)
from axolotl.integrations.ternary.quant import (
    ACT_QMAX,
    SCALE_EPS,
    ternary_codes,
    unpack_codes,
)
from axolotl.integrations.ternary.swap import convert_model
from axolotl.utils.dict import DictDefault

IN_FEATURES = 16
OUT_FEATURES = 8


def _tiny_llama(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _linear(seed: int = 0, in_features: int = IN_FEATURES) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(in_features, OUT_FEATURES, bias=False)


def _oracle_weight(
    weight: torch.Tensor, lambda_: float = 1.0, group_size: int | None = None
) -> torch.Tensor:
    """Independent restatement of the pinned weight numerics (design.md)."""
    values = weight.detach().float()
    if group_size is not None:
        values = values.reshape(values.shape[0], -1, group_size)
    scale = values.abs().mean(-1 if group_size else None).clamp_min(SCALE_EPS)
    scale16 = scale.half().float()
    if group_size is not None:
        scale, scale16 = scale.unsqueeze(-1), scale16.unsqueeze(-1)
    codes = torch.round(values / scale).clamp(-1.0, 1.0)
    quantized = (codes * scale16).reshape(weight.shape)
    return (values.reshape(weight.shape) * (1 - lambda_) + quantized * lambda_).to(
        weight.dtype
    )


def _oracle_act(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Independent restatement of the pinned per-token activation numerics."""
    values = x.detach().float()
    scale = (values.abs().amax(-1, keepdim=True) / ACT_QMAX).clamp_min(SCALE_EPS)
    quantized = torch.round(values / scale).clamp(-ACT_QMAX, ACT_QMAX) * scale
    return (values * (1 - lambda_) + quantized * lambda_).to(x.dtype)


def test_from_linear_adopts_the_latent_weight():
    linear = _linear()
    weight = linear.weight

    module = TernaryLinear.from_linear(linear)

    assert module.weight is weight
    assert module.in_features == IN_FEATURES
    assert module.out_features == OUT_FEATURES
    assert module.scale is None
    assert module.lambda_ == 1.0
    assert module.baked is False
    assert module.weight.requires_grad


def test_from_linear_rejects_bias():
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)

    with pytest.raises(ValueError, match="biased Linear"):
        TernaryLinear.from_linear(linear)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"weight_scale": "group"}, "requires a group_size"),
        ({"weight_scale": "group", "group_size": 5}, "does not divide"),
        ({"group_size": 8}, "only valid with weight_scale"),
        ({"activation_bits": 4}, "activation_bits must be 8 or None"),
        ({"weight_scale": "int4"}, "unknown ternary weight_scale"),
        ({"int8_forward": "yes"}, "int8_forward must be"),
    ],
)
def test_init_validates_quantizer_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TernaryLinear(IN_FEATURES, OUT_FEATURES, **kwargs)


def test_forward_at_lambda_zero_is_the_fp_linear():
    linear = _linear()
    x = torch.randn(3, IN_FEATURES)
    expected = linear(x)

    module = TernaryLinear.from_linear(linear)
    module.set_lambda(0.0)

    assert torch.equal(module(x), expected)


def test_model_forward_at_lambda_zero_matches_the_fp_model():
    model = _tiny_llama()
    converted = copy.deepcopy(model)
    convert_model(converted, DictDefault({"ternary": {}}))
    for _, module in iter_ternary_modules(converted):
        module.set_lambda(0.0)

    input_ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        expected = model(input_ids).logits
        actual = converted(input_ids).logits

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("lambda_", [0.25, 0.5, 1.0])
def test_forward_matches_the_eager_oracle(lambda_):
    linear = _linear()
    x = torch.randn(2, 3, IN_FEATURES)
    module = TernaryLinear.from_linear(linear)
    module.set_lambda(lambda_)

    expected = F.linear(_oracle_act(x, lambda_), _oracle_weight(linear.weight, lambda_))

    torch.testing.assert_close(module(x), expected, rtol=1e-6, atol=1e-6)


def test_forward_without_activation_quantization():
    linear = _linear()
    x = torch.randn(2, IN_FEATURES)
    module = TernaryLinear.from_linear(linear, activation_bits=None)

    expected = F.linear(x, _oracle_weight(linear.weight))

    torch.testing.assert_close(module(x), expected, rtol=1e-6, atol=1e-6)


def test_forward_with_group_scale():
    linear = _linear(in_features=16)
    x = torch.randn(2, 16)
    module = TernaryLinear.from_linear(linear, weight_scale="group", group_size=8)

    expected = F.linear(_oracle_act(x), _oracle_weight(linear.weight, group_size=8))

    torch.testing.assert_close(module(x), expected, rtol=1e-6, atol=1e-6)


def test_set_lambda_clamps_to_unit_interval():
    module = TernaryLinear(IN_FEATURES, OUT_FEATURES)

    module.set_lambda(-1.0)
    assert module.lambda_ == 0.0
    module.set_lambda(3.0)
    assert module.lambda_ == 1.0


def test_straight_through_gradients_reach_the_latent_weight():
    linear = _linear()
    module = TernaryLinear.from_linear(linear)
    x = torch.randn(4, IN_FEATURES, requires_grad=True)

    module(x).sum().backward()

    x_eff = _oracle_act(x)
    w_eff = _oracle_weight(linear.weight)
    torch.testing.assert_close(
        module.weight.grad, x_eff.sum(0).expand(OUT_FEATURES, IN_FEATURES)
    )
    torch.testing.assert_close(x.grad, w_eff.sum(0).expand_as(x))


def test_baked_weight_is_exactly_ternary():
    linear = _linear()
    module = TernaryLinear.from_linear(linear)

    baked = module.baked_weight()

    scale16 = linear.weight.detach().float().abs().mean().half().float().item()
    assert set(torch.unique(baked).tolist()) <= {-scale16, 0.0, scale16}
    torch.testing.assert_close(baked, _oracle_weight(linear.weight), rtol=0, atol=0)


def test_post_training_bakes_in_place():
    linear = _linear()
    module = TernaryLinear.from_linear(linear)
    weight = module.weight
    expected = module.baked_weight()
    x = torch.randn(2, IN_FEATURES)

    module._post_training(None, "proj")

    assert module.baked is True
    assert module.weight is weight
    assert torch.equal(module.weight.data, expected)
    # a baked module must not re-quantize its already-ternary weight
    torch.testing.assert_close(module(x), F.linear(_oracle_act(x), expected))

    module._post_training(None, "proj")
    assert torch.equal(module.weight.data, expected)


class _SwallowingParameter(nn.Parameter):
    """A parameter whose in-place write silently does nothing, as a `.data` rebind did."""

    def copy_(self, other, non_blocking=False):
        return self


def test_post_training_refuses_a_write_that_did_not_land():
    """A silently unbaked parameter would ship as an FP checkpoint under a manifest."""
    module = TernaryLinear.from_linear(_linear())
    module.weight = _SwallowingParameter(module.weight.detach())

    with pytest.raises(RuntimeError, match="did not land"):
        module._post_training(None, "proj")

    assert not module.is_baked()


def test_post_training_warns_and_records_when_lambda_never_reached_one(caplog):
    model = _tiny_llama()
    manifest = convert_model(model, DictDefault({"ternary": {}}))
    for _, module in iter_ternary_modules(model):
        module.set_lambda(0.5)

    with caplog.at_level("WARNING"):
        for name, module in model.named_modules():
            if hasattr(module, "_post_training"):
                module._post_training(model, name)

    assert "lambda=0.5" in caplog.text
    assert manifest.final_lambda == 0.5
    # one warning for the whole model, not one per swapped linear
    assert caplog.text.count("The saved master") == 1


def test_post_training_records_a_completed_schedule():
    model = _tiny_llama()
    manifest = convert_model(model, DictDefault({"ternary": {}}))

    for name, module in model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(model, name)

    assert manifest.final_lambda == 1.0


def test_reloading_a_baked_master_is_a_fixed_point():
    """The master is the interchange artifact, so loading it back must not rescale it."""
    module = TernaryLinear.from_linear(_linear())
    module._post_training(None, "proj")
    x = torch.randn(4, IN_FEATURES)
    with torch.no_grad():
        expected = module(x)

    fresh = TernaryLinear(IN_FEATURES, OUT_FEATURES)
    fresh.load_state_dict(module.state_dict())

    assert fresh.baked is True and fresh.is_baked()
    assert torch.equal(fresh.weight.detach(), module.weight.detach())
    with torch.no_grad():
        torch.testing.assert_close(fresh(x), expected, rtol=0, atol=0)
    # the absmean of ternary values is smaller than their amax by the non-zero
    # fraction, which is exactly the rescaling this guards against
    assert module.weight.detach().abs().mean() < module.weight.detach().abs().max()


def test_swapping_a_baked_linear_keeps_its_scale():
    """`base_model:` pointing at a saved master goes through `from_linear`, not a load."""
    module = TernaryLinear.from_linear(_linear())
    module._post_training(None, "proj")
    baked = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    with torch.no_grad():
        baked.weight.copy_(module.weight.detach())

    reswapped = TernaryLinear.from_linear(baked)

    assert reswapped.is_baked()
    x = torch.randn(4, IN_FEATURES)
    with torch.no_grad():
        torch.testing.assert_close(reswapped(x), module(x), rtol=0, atol=0)


def test_a_learnable_scale_reloaded_from_a_master_starts_at_its_amax():
    module = TernaryLinear.from_linear(_linear())
    module._post_training(None, "proj")
    baked = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    with torch.no_grad():
        baked.weight.copy_(module.weight.detach())

    reswapped = TernaryLinear.from_linear(baked, weight_scale="learnable")

    torch.testing.assert_close(
        reswapped.scale.detach().exp(),
        module.weight.detach().abs().max().float().reshape(1),
    )


def test_baked_lapses_once_the_latent_moves_again():
    """Resumed training re-quantizes; a stale flag would train in full precision."""
    module = TernaryLinear.from_linear(_linear())
    module._post_training(None, "proj")
    assert module.is_baked()

    with torch.no_grad():
        module.weight.add_(0.01)

    assert module.baked is True
    assert not module.is_baked()
    x = torch.randn(2, IN_FEATURES)
    with torch.no_grad():
        assert torch.equal(module(x), F.linear(_oracle_act(x), module.baked_weight()))


def test_a_learnable_scale_below_the_floor_does_not_kill_the_layer():
    """An f16-subnormal scale rounds to zero, and the parity gate certifies the corpse."""
    module = TernaryLinear.from_linear(_linear(), weight_scale="learnable")
    with torch.no_grad():
        module.scale.fill_(torch.tensor(1e-9).log())

    baked = module.baked_weight()

    assert float(baked.abs().max()) > 0.0
    assert float(module._scale().detach()) == pytest.approx(SCALE_EPS)


def test_post_training_hook_bakes_a_converted_model():
    model = _tiny_llama()
    convert_model(model, DictDefault({"ternary": {}}))

    for name, module in model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(model, name)

    for _, module in iter_ternary_modules(model):
        assert module.baked is True
        assert torch.unique(module.weight).numel() <= 3

    with torch.no_grad():
        logits = model(torch.randint(0, 128, (1, 4))).logits
    assert torch.isfinite(logits).all()


def test_swapped_state_dict_matches_the_fp_layout(tmp_path):
    reference = _tiny_llama()
    model = _tiny_llama(seed=1)
    convert_model(model, DictDefault({"ternary": {}}))

    assert set(model.state_dict()) == set(reference.state_dict())

    torch.save(model.state_dict(), tmp_path / "master.pt")
    reloaded = _tiny_llama(seed=2)
    convert_model(reloaded, DictDefault({"ternary": {}}))
    reloaded.load_state_dict(torch.load(tmp_path / "master.pt", weights_only=True))

    input_ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        torch.testing.assert_close(
            reloaded(input_ids).logits, model(input_ids).logits, rtol=0, atol=0
        )


def test_code_snapshot_packs_the_current_codes():
    linear = _linear()
    module = TernaryLinear.from_linear(linear)

    packed = module.code_snapshot()

    assert packed.dtype == torch.uint8
    assert packed.numel() == (linear.weight.numel() + 3) // 4
    scale = linear.weight.detach().float().abs().mean()
    codes = ternary_codes(linear.weight.detach(), scale)
    assert torch.equal(unpack_codes(packed, tuple(codes.shape)), codes)


def test_activation_memo_is_opt_in():
    modules = nn.ModuleList([TernaryLinear.from_linear(_linear(i)) for i in range(2)])
    first, second = modules
    x = torch.randn(2, IN_FEATURES)

    assert first._quant_act(x, 1.0) is not second._quant_act(x, 1.0)

    assert set_act_quant_sharing(modules) == 2
    shared = first._quant_act(x, 1.0)
    assert second._quant_act(x, 1.0) is shared
    torch.testing.assert_close(shared, _oracle_act(x), rtol=0, atol=0)

    assert first._quant_act(x, 0.5) is not shared
    x.add_(1.0)
    assert first._quant_act(x, 1.0) is not shared

    assert set_act_quant_sharing(modules, enabled=False) == 2
    assert first._quant_act(x, 1.0) is not second._quant_act(x, 1.0)


def test_learnable_scale_is_trained_and_folded_at_bake():
    linear = _linear()
    module = TernaryLinear.from_linear(linear, weight_scale="learnable")

    expected_log = (
        linear.weight.detach().float().abs().mean().clamp_min(SCALE_EPS).log()
    )
    # shape (1,), not a scalar: FSDP2 refuses to shard 0-dim parameters
    assert module.scale.shape == (1,)
    torch.testing.assert_close(module.scale, expected_log.reshape(1))

    module(torch.randn(2, IN_FEATURES)).sum().backward()
    assert module.scale.grad is not None
    assert module.scale.grad.abs().item() > 0.0

    baked = module.baked_weight()
    module._post_training(None, "proj")
    assert module.scale is None
    assert torch.equal(module.weight.data, baked)


def test_extra_repr_reports_the_quantizer():
    module = TernaryLinear(
        IN_FEATURES, OUT_FEATURES, weight_scale="group", group_size=8
    )

    text = repr(module)

    assert "in_features=16" in text
    assert "weight_scale=group" in text
    assert "group_size=8" in text
    assert "activation_bits=8" in text


def test_int8_forward_falls_back_off_gpu():
    linear = _linear()
    module = TernaryLinear.from_linear(linear, int8_forward="auto")
    x = torch.randn(2, IN_FEATURES)

    expected = F.linear(_oracle_act(x), _oracle_weight(linear.weight))

    torch.testing.assert_close(module(x), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_int8_forward_matches_the_fake_quant_forward():
    linear = nn.Linear(512, 512, bias=False).cuda().to(torch.bfloat16)
    int8 = TernaryLinear.from_linear(linear, int8_forward="auto")
    eager = TernaryLinear.from_linear(copy.deepcopy(linear))
    x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)

    out = int8(x)
    reference = eager(x)

    assert (out - reference).norm() / reference.norm() < 5e-3

    # below λ == 1 the int8 path is skipped, so the two modules agree exactly
    int8.set_lambda(0.5)
    eager.set_lambda(0.5)
    torch.testing.assert_close(int8(x), eager(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_forward_matches_the_eager_oracle():
    linear = _linear().cuda()
    module = TernaryLinear.from_linear(linear, fused=False)
    x = torch.randn(4, IN_FEATURES, device="cuda")

    expected = F.linear(_oracle_act(x), _oracle_weight(linear.weight))

    torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_fused_dispatch_matches_the_eager_path():
    linear = _linear().cuda()
    fused = TernaryLinear.from_linear(linear, fused=True)
    eager = TernaryLinear.from_linear(copy.deepcopy(linear), fused=False)
    x = torch.randn(4, IN_FEATURES, device="cuda", requires_grad=True)

    fused(x).sum().backward()
    fused_grads = (fused.weight.grad.clone(), x.grad.clone())
    x.grad = None
    eager(x).sum().backward()

    torch.testing.assert_close(fused(x), eager(x), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_grads[0], eager.weight.grad)
    torch.testing.assert_close(fused_grads[1], x.grad)
