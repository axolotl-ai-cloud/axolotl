"""BF16 merge-aware LoRA forward (``lora_merge_aware: bf16``)."""

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn

from axolotl.monkeypatch.lora_merge_aware import (
    configure_lora_merge_aware,
    install_lora_merge_aware_linears,
)
from axolotl.utils.dict import DictDefault


class _Toy(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.q = nn.Linear(64, 48, bias=bias)
        self.o = nn.Linear(48, 64, bias=False)

    def forward(self, x):
        return self.o(torch.relu(self.q(x)))


def _peft(base_dtype=torch.bfloat16, dropout=0.0, seed=0, **lora_kwargs):
    torch.manual_seed(seed)
    model = get_peft_model(
        _Toy().to(base_dtype),
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=dropout,
            target_modules=["q", "o"],
            **lora_kwargs,
        ),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 3e-3)
    return model


def _lora(module):
    return (
        module.base_layer.weight,
        module.base_layer.bias,
        module.lora_A["default"].weight,
        module.lora_B["default"].weight,
        module.scaling["default"],
    )


def _assert_rel_close(got, expected, rel=1e-2):
    err = (got.float() - expected.float()).norm() / expected.float().norm()
    assert err < rel, err


def _reference_forward(module, x):
    """Autograd straight through the cast: F.linear(x, bf16(W + s * B @ A))."""
    weight, bias, lora_a, lora_b, scaling = _lora(module)
    merged = (weight.float() + (lora_b @ lora_a) * scaling).to(torch.bfloat16)
    return F.linear(x, merged.to(x.dtype), None if bias is None else bias.to(x.dtype))


@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_forward_and_grads_match_straight_through_reference(base_dtype):
    model = _peft(base_dtype)
    q = model.base_model.model.q
    assert install_lora_merge_aware_linears(model) == (2, [])

    x = torch.randn(3, 5, 64, dtype=base_dtype, requires_grad=True)
    out = q(x)
    grad = torch.randn_like(out)
    out.backward(grad)
    got = (x.grad.clone(), q.lora_A["default"].weight.grad.clone())
    got_b = q.lora_B["default"].weight.grad.clone()
    for param in (x, q.lora_A["default"].weight, q.lora_B["default"].weight):
        param.grad = None

    ref = _reference_forward(q, x)
    assert torch.equal(out, ref)
    # The reference rounds the dense dW through the BF16 cast; the low-rank
    # backward does not, so the two agree to BF16 precision.
    ref.backward(grad)
    _assert_rel_close(got[0], x.grad)
    _assert_rel_close(got[1], q.lora_A["default"].weight.grad)
    _assert_rel_close(got_b, q.lora_B["default"].weight.grad)


def test_forward_is_bitwise_the_peft_merged_bf16_model():
    model = _peft()
    install_lora_merge_aware_linears(model)
    model.eval()
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        merge_aware = model(x)
        q = model.base_model.model.q
        weight, _, lora_a, lora_b, scaling = _lora(q)
        expected_q = (weight.float() + (lora_b @ lora_a) * scaling).to(torch.bfloat16)
        model.merge_adapter()
        assert torch.equal(q.base_layer.weight, expected_q)
        merged = model(x)
        model.unmerge_adapter()
    assert torch.equal(merge_aware, merged)


def test_ordinary_lora_differs_from_merged_model():
    model = _peft()
    model.eval()
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        unmerged = model(x)
        model.merge_adapter()
        merged = model(x)
        model.unmerge_adapter()
    assert not torch.equal(unmerged, merged)


def test_autocast_with_fp32_params_serves_bf16_weights():
    model = _peft(torch.float32)
    install_lora_merge_aware_linears(model)
    q = model.base_model.model.q
    x = torch.randn(2, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = q(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, _reference_forward(q, x.to(torch.bfloat16)))


def test_disabled_adapter_and_merged_state_fall_back():
    model = _peft()
    install_lora_merge_aware_linears(model)
    q = model.base_model.model.q
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    with torch.no_grad(), model.disable_adapter():
        assert torch.equal(q(x), q.base_layer(x))
    q.merge()
    with torch.no_grad():
        assert torch.equal(q(x), q.base_layer(x))
    q.unmerge()


def test_dropout_adds_low_rank_residual_and_is_exact_in_eval():
    model = _peft(dropout=0.5)
    install_lora_merge_aware_linears(model)
    q = model.base_model.model.q
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    torch.manual_seed(1)
    out = q(x)
    torch.manual_seed(1)
    dropped = q.lora_dropout["default"](x)
    _, _, lora_a, lora_b, scaling = _lora(q)
    residual = ((dropped - x).float() @ lora_a.t() @ lora_b.t()) * scaling
    _assert_rel_close(out, _reference_forward(q, x).float() + residual)
    model.eval()
    with torch.no_grad():
        assert torch.equal(q(x), _reference_forward(q, x))


def test_trainable_base_bias_gets_gradient():
    model = _peft()
    install_lora_merge_aware_linears(model)
    q = model.base_model.model.q
    q.base_layer.bias.requires_grad_(True)
    out = q(torch.randn(3, 64, dtype=torch.bfloat16))
    out.float().sum().backward()
    assert q.base_layer.bias.grad.dtype == q.base_layer.bias.dtype
    torch.testing.assert_close(
        q.base_layer.bias.grad.float(), torch.full((48,), 3.0), rtol=0, atol=0
    )


def test_install_is_idempotent_and_skips_unsupported_modules():
    model = _peft(use_dora=True)
    installed, skipped = install_lora_merge_aware_linears(model)
    assert installed == 0
    assert len(skipped) == 2 and all("DoRA" in reason for reason in skipped)

    model = _peft()
    model.base_model.model.o.base_layer._nvfp4_pts = torch.ones(())
    assert install_lora_merge_aware_linears(model)[0] == 1
    installed, skipped = install_lora_merge_aware_linears(model)
    assert installed == 1
    assert "NVFP4" in skipped[0]


def test_install_leaves_a_forward_owned_by_another_integration():
    model = _peft()
    q = model.base_model.model.q
    q.forward = q.forward  # instance-level override, as other integrations install
    installed, skipped = install_lora_merge_aware_linears(model)
    assert installed == 1
    assert skipped[0].startswith("base_model.model.q ")


def test_configure_raises_when_nothing_is_merge_aware():
    model = _peft(use_dora=True)
    with pytest.raises(ValueError, match="no supported LoRA linear"):
        configure_lora_merge_aware(DictDefault({"lora_merge_aware": "bf16"}), model)


def test_configure_is_a_noop_when_unset():
    model = _peft()
    configure_lora_merge_aware(DictDefault({}), model)
    assert "forward" not in model.base_model.model.q.__dict__


def test_training_step_reaches_merge_aware_weights():
    """Gradients from the merge-aware loss move the adapter."""
    model = _peft()
    install_lora_merge_aware_linears(model)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=1.0)
    before = [p.detach().clone() for p in params]
    loss = model(torch.randn(4, 64, dtype=torch.bfloat16)).float().square().mean()
    loss.backward()
    opt.step()
    assert all(not torch.equal(b, p) for b, p in zip(before, params, strict=True))
