"""CPU-only tests for the per-row and five-value dual scale modes.

The oracles are restated independently of `quant`, so a change to the pinned numerics
has to be made twice before these agree with it.
"""

import copy
import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, TrainerControl, TrainerState

from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.callbacks import TernaryMonitorCallback
from axolotl.integrations.ternary.modules import (
    FUSED_SCALE_MODES,
    INT8_FORWARD_SCALE_MODES,
    UNIMPLEMENTED_SCALE_MODES,
    TernaryLinear,
    as_local,
    iter_ternary_modules,
)
from axolotl.integrations.ternary.quant import (
    ACT_QMAX,
    DUAL_PLANES,
    SCALE_EPS,
    baked_dual_codes_and_scales,
    dual_absmean_scales,
    dual_codes,
    dual_state_planes,
    flip_count,
    unpack_codes,
    zero_fraction,
)
from axolotl.integrations.ternary.swap import convert_model
from axolotl.utils.dict import DictDefault

IN_FEATURES = 16
OUT_FEATURES = 8

ROW_MODES = ("learnable_row", "dual")


def _linear(seed: int = 0, in_features: int = IN_FEATURES) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(in_features, OUT_FEATURES, bias=False)


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


def _oracle_act(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    values = x.detach().float()
    scale = (values.abs().amax(-1, keepdim=True) / ACT_QMAX).clamp_min(SCALE_EPS)
    quantized = torch.round(values / scale).clamp(-ACT_QMAX, ACT_QMAX) * scale
    return (values * (1 - lambda_) + quantized * lambda_).to(x.dtype)


def _oracle_row_weight(
    weight: torch.Tensor, scale: torch.Tensor | None = None, lambda_: float = 1.0
) -> torch.Tensor:
    """Per-row restatement: one scale per output channel, f16-rounded on dequant."""
    values = weight.detach().float()
    if scale is None:
        scale = values.abs().mean(-1, keepdim=True)
    scale = scale.detach().float().reshape(-1, 1).clamp_min(SCALE_EPS)
    codes = torch.round(values / scale).clamp(-1.0, 1.0)
    quantized = codes * scale.half().float()
    return (values * (1 - lambda_) + quantized * lambda_).to(weight.dtype)


def _oracle_dual_weight(
    weight: torch.Tensor,
    scale_lo: torch.Tensor,
    scale_hi: torch.Tensor,
    lambda_: float = 1.0,
) -> torch.Tensor:
    """Five-value restatement: nearest of `{0, ±s_lo, ±s_hi}`, ties to the lower state."""
    values = weight.detach().float()
    low = scale_lo.detach().float().reshape(-1, 1).clamp_min(SCALE_EPS)
    high = scale_hi.detach().float().reshape(-1, 1).clamp_min(SCALE_EPS)
    magnitude = values.abs()
    level = (magnitude > 0.5 * low).float() + (magnitude > 0.5 * (low + high)).float()
    picked = torch.where(level > 1, high.half().float(), low.half().float())
    quantized = torch.sign(values) * picked * (level > 0)
    return (values * (1 - lambda_) + quantized * lambda_).to(weight.dtype)


class TestRowScaleOracle:
    def test_row_scale_golden(self):
        weight = torch.tensor([[1.0, 2.0, 3.0, 6.0], [0.1, 0.1, 0.1, 0.1]])
        module = TernaryLinear.from_linear(
            nn.Linear(4, 2, bias=False), weight_scale="learnable_row"
        )
        with torch.no_grad():
            module.weight.copy_(weight)
            module.refresh_scale_from_weight()

        baked = module.baked_weight()

        # row absmeans are 3.0 and 0.1; f16(0.1) is 0.0999755859375
        assert baked.tolist() == [
            [0.0, 3.0, 3.0, 3.0],
            [0.0999755859375] * 4,
        ]

    def test_a_row_scale_is_not_the_tensor_scale(self):
        weight = torch.tensor([[1.0, 2.0, 3.0, 6.0], [0.1, 0.1, 0.1, 0.1]])
        per_tensor = TernaryLinear.from_linear(nn.Linear(4, 2, bias=False))
        with torch.no_grad():
            per_tensor.weight.copy_(weight)

        # the small row rounds to zero under the shared absmean of 1.55
        assert per_tensor.baked_weight()[1].abs().sum() == 0.0

    def test_scale_is_seeded_from_the_row_absmean(self):
        linear = _linear()
        module = TernaryLinear.from_linear(linear, weight_scale="learnable_row")

        assert module.scale.shape == (OUT_FEATURES, 1)
        torch.testing.assert_close(
            module.scale.detach().exp(),
            linear.weight.detach().abs().mean(-1, keepdim=True),
        )

    @pytest.mark.parametrize("lambda_", [0.0, 0.25, 1.0])
    def test_forward_matches_the_row_oracle(self, lambda_):
        linear = _linear()
        module = TernaryLinear.from_linear(linear, weight_scale="learnable_row")
        module.set_lambda(lambda_)
        x = torch.randn(2, 3, IN_FEATURES)

        expected = F.linear(
            _oracle_act(x, lambda_), _oracle_row_weight(linear.weight, lambda_=lambda_)
        )

        torch.testing.assert_close(module(x), expected, rtol=1e-6, atol=1e-6)

    def test_lambda_zero_is_the_fp_linear(self):
        linear = _linear()
        module = TernaryLinear.from_linear(linear, weight_scale="learnable_row")
        module.set_lambda(0.0)
        x = torch.randn(3, IN_FEATURES)

        assert torch.equal(module(x), linear(x))

    def test_the_scale_trains_and_folds_into_the_bake(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="learnable_row")

        module(torch.randn(4, IN_FEATURES)).sum().backward()

        assert module.scale.grad.shape == (OUT_FEATURES, 1)
        assert module.scale.grad.abs().sum() > 0.0

        with torch.no_grad():
            module.scale.mul_(1.1)
        baked = module.baked_weight()
        module._post_training(None, "q_proj")

        assert module.scale is None
        assert torch.equal(module.weight.data, baked)

    def test_baked_rows_carry_their_own_magnitude(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="learnable_row")

        baked = module.baked_weight()

        torch.testing.assert_close(
            baked, _oracle_row_weight(module.weight), rtol=0, atol=0
        )
        for row in baked:
            assert torch.unique(row.abs()).numel() <= 2
        # distinct rows, or a per-tensor scale would have done
        assert torch.unique(baked.abs().amax(-1)).numel() > 1

    def test_reswapping_a_baked_master_recovers_the_row_amax(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="learnable_row")
        module._post_training(None, "q_proj")
        master = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        with torch.no_grad():
            master.weight.copy_(module.weight.detach())

        reswapped = TernaryLinear.from_linear(master, weight_scale="learnable_row")

        assert reswapped.is_baked()
        torch.testing.assert_close(
            reswapped.scale.detach().exp(),
            module.weight.detach().abs().amax(-1, keepdim=True),
        )
        x = torch.randn(4, IN_FEATURES)
        with torch.no_grad():
            torch.testing.assert_close(reswapped(x), module(x), rtol=0, atol=0)

    def test_reloading_a_baked_master_is_a_fixed_point(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="learnable_row")
        module._post_training(None, "q_proj")
        x = torch.randn(4, IN_FEATURES)
        with torch.no_grad():
            expected = module(x)

        fresh = TernaryLinear(IN_FEATURES, OUT_FEATURES, weight_scale="learnable_row")
        fresh.load_state_dict(module.state_dict(), strict=False)

        assert fresh.is_baked()
        torch.testing.assert_close(
            fresh.scale.detach().exp(),
            module.weight.detach().abs().amax(-1, keepdim=True),
        )
        with torch.no_grad():
            torch.testing.assert_close(fresh(x), expected, rtol=0, atol=0)

    def test_a_shipped_scale_is_not_overwritten_on_load(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="learnable_row")
        with torch.no_grad():
            module.scale.fill_(0.5)

        fresh = TernaryLinear(IN_FEATURES, OUT_FEATURES, weight_scale="learnable_row")
        fresh.load_state_dict(module.state_dict())

        torch.testing.assert_close(fresh.scale.detach(), module.scale.detach())

    def test_degenerate_rows_survive_the_round_trip(self):
        module = TernaryLinear.from_linear(
            nn.Linear(4, 3, bias=False), weight_scale="learnable_row"
        )
        with torch.no_grad():
            module.weight.copy_(
                torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0], [0.5, 0.0, -0.5, 0.25]]
                )
            )
            module.refresh_scale_from_weight()

        baked = module.baked_weight()
        module._post_training(None, "q_proj")
        reswapped = TernaryLinear.from_linear(
            _linear_holding(baked), weight_scale="learnable_row"
        )

        assert baked[0].abs().sum() == 0.0
        assert baked[1].tolist() == [2.0, 2.0, 2.0, 2.0]
        assert reswapped.is_baked()
        assert torch.equal(reswapped.baked_weight(), baked)


class TestDualOracle:
    def test_state_assignment_golden(self):
        weight = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.5]])
        low = torch.tensor([[1.0]])
        high = torch.tensor([[3.0]])

        codes = dual_codes(weight, low, high)

        # boundaries at 0.5 and 2.0; both ties take the lower state
        assert codes.dtype is torch.int8
        assert codes.tolist() == [[0, 1, 1, 2, 0]]

    def test_signs_are_symmetric(self):
        weight = torch.tensor([[-0.0, -1.0, -2.0, -3.0, -0.5]])

        codes = dual_codes(weight, torch.tensor([[1.0]]), torch.tensor([[3.0]]))

        assert codes.tolist() == [[0, -1, -1, -2, 0]]

    def test_initial_scales_golden(self):
        weight = torch.tensor([[0.5, 1.0, 2.0, 4.5]])

        low, high = dual_absmean_scales(weight)

        # row absmean 2.0: |w| > 3.0 sets s_hi, the rest above 1.0 sets s_lo
        assert low.shape == (1, 1)
        assert low.tolist() == [[2.0]]
        assert high.tolist() == [[4.5]]

    def test_initial_scales_fall_back_when_a_level_is_empty(self):
        weight = torch.tensor([[1.0, 2.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])

        low, high = dual_absmean_scales(weight)

        assert low[0].item() == pytest.approx(7 / 3)
        assert high[0].item() == pytest.approx(7 / 3)
        assert low[1].item() == pytest.approx(SCALE_EPS)
        assert high[1].item() == pytest.approx(SCALE_EPS)

    def test_scales_are_ordered_however_they_are_passed(self):
        weight = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.5]])
        low = torch.tensor([[1.0]])
        high = torch.tensor([[3.0]])

        assert torch.equal(dual_codes(weight, high, low), dual_codes(weight, low, high))

    def test_recovery_from_a_baked_row(self):
        weight = torch.tensor([[0.0, 1.0, -1.0, 3.0, -3.0]])

        codes, low, high = baked_dual_codes_and_scales(weight)

        assert codes.tolist() == [[0, 1, -1, 2, -2]]
        assert low.tolist() == [[1.0]]
        assert high.tolist() == [[3.0]]

    def test_recovery_declines_a_latent_weight(self):
        torch.manual_seed(0)

        assert baked_dual_codes_and_scales(torch.randn(8, 16)) is None

    def test_recovery_of_degenerate_rows(self):
        weight = torch.tensor([[0.0, 0.0], [2.0, -2.0]])

        codes, low, high = baked_dual_codes_and_scales(weight)

        expected = torch.tensor([[SCALE_EPS], [2.0]])
        assert codes.tolist() == [[0, 0], [1, -1]]
        torch.testing.assert_close(low, expected)
        torch.testing.assert_close(high, expected)

    def test_state_planes_are_ternary_and_agree_on_zero(self):
        codes = torch.tensor([[0, 1, -1, 2, -2]], dtype=torch.int8)

        planes = dual_state_planes(codes)

        assert planes.shape == (DUAL_PLANES, 1, 5)
        assert planes[0].tolist() == [[0, 1, -1, 1, -1]]
        assert planes[1].tolist() == [[0, -1, -1, 1, 1]]
        assert torch.equal((planes == 0).all(0), codes == 0)


class TestDualModule:
    def test_two_ordered_scale_vectors(self):
        linear = _linear()
        module = TernaryLinear.from_linear(linear, weight_scale="dual")

        assert module.scale.shape == (OUT_FEATURES, 1)
        assert module.scale_lo.shape == (OUT_FEATURES, 1)
        low, high = dual_absmean_scales(linear.weight.detach())
        torch.testing.assert_close(module.scale_lo.detach().exp(), low)
        torch.testing.assert_close(module.scale.detach().exp(), high)
        assert bool((module.scale_lo <= module.scale).all())

    @pytest.mark.parametrize("lambda_", [0.0, 0.5, 1.0])
    def test_forward_matches_the_dual_oracle(self, lambda_):
        linear = _linear()
        module = TernaryLinear.from_linear(linear, weight_scale="dual")
        module.set_lambda(lambda_)
        x = torch.randn(2, 3, IN_FEATURES)

        low, high = dual_absmean_scales(linear.weight.detach())
        expected = F.linear(
            _oracle_act(x, lambda_),
            _oracle_dual_weight(linear.weight, low, high, lambda_),
        )

        torch.testing.assert_close(module(x), expected, rtol=1e-6, atol=1e-6)

    def test_an_inverted_pair_is_read_in_order(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="dual")
        x = torch.randn(2, IN_FEATURES)
        with torch.no_grad():
            expected = module(x)
            swapped = module.scale.detach().clone()
            module.scale.copy_(module.scale_lo.detach())
            module.scale_lo.copy_(swapped)

        with torch.no_grad():
            assert torch.equal(module(x), expected)

    def test_bake_writes_five_values_per_row(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="dual")

        baked = module.baked_weight()

        low, high = module._dual_scales(module.weight)
        torch.testing.assert_close(
            baked,
            _oracle_dual_weight(module.weight, low, high),
            rtol=0,
            atol=0,
        )
        for row in baked:
            assert torch.unique(row.abs()).numel() <= 3
        assert torch.unique(baked).numel() > 3

    def test_reswapping_a_baked_master_recovers_both_scales(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="dual")
        module._post_training(None, "down_proj")
        assert module.scale is None and module.scale_lo is None

        reswapped = TernaryLinear.from_linear(
            _linear_holding(module.weight.detach()), weight_scale="dual"
        )

        assert reswapped.is_baked()
        magnitude = module.weight.detach().abs()
        expected_hi = magnitude.amax(-1, keepdim=True)
        expected_lo = magnitude.masked_fill(magnitude == 0, float("inf")).amin(
            -1, keepdim=True
        )
        torch.testing.assert_close(reswapped.scale.detach().exp(), expected_hi)
        torch.testing.assert_close(reswapped.scale_lo.detach().exp(), expected_lo)
        x = torch.randn(4, IN_FEATURES)
        with torch.no_grad():
            torch.testing.assert_close(reswapped(x), module(x), rtol=0, atol=0)

    def test_reloading_a_baked_master_is_a_fixed_point(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="dual")
        module._post_training(None, "down_proj")
        x = torch.randn(4, IN_FEATURES)
        with torch.no_grad():
            expected = module(x)

        fresh = TernaryLinear(IN_FEATURES, OUT_FEATURES, weight_scale="dual")
        fresh.load_state_dict(module.state_dict(), strict=False)

        assert fresh.is_baked()
        with torch.no_grad():
            torch.testing.assert_close(fresh(x), expected, rtol=0, atol=0)
        assert torch.equal(fresh.baked_weight(), module.weight.detach())

    def test_degenerate_rows_survive_the_round_trip(self):
        module = TernaryLinear.from_linear(
            nn.Linear(4, 3, bias=False), weight_scale="dual"
        )
        with torch.no_grad():
            module.weight.copy_(
                torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0], [0.5, 0.0, 2.0, 4.0]]
                )
            )
            module.refresh_scale_from_weight()

        baked = module.baked_weight()
        reswapped = TernaryLinear.from_linear(
            _linear_holding(baked), weight_scale="dual"
        )

        assert baked[0].abs().sum() == 0.0
        # a single-magnitude row recovers s_lo == s_hi and stays put
        assert baked[1].tolist() == [1.0, -1.0, 1.0, -1.0]
        assert reswapped.is_baked()
        assert torch.equal(reswapped.baked_weight(), baked)
        torch.testing.assert_close(
            reswapped.scale_lo.detach().exp()[1], reswapped.scale.detach().exp()[1]
        )

    def test_straight_through_gradients(self):
        linear = _linear()
        module = TernaryLinear.from_linear(linear, weight_scale="dual")
        x = torch.randn(4, IN_FEATURES, requires_grad=True)

        module(x).sum().backward()

        x_eff = _oracle_act(x)
        torch.testing.assert_close(
            module.weight.grad, x_eff.sum(0).expand(OUT_FEATURES, IN_FEATURES)
        )
        low, high = module._dual_scales(module.weight)
        codes = dual_codes(linear.weight.detach(), low, high)
        upstream = x_eff.sum(0).expand(OUT_FEATURES, IN_FEATURES)
        signs = torch.sign(codes.float())
        # log-scale parameters, so the chain rule multiplies by the scale itself
        torch.testing.assert_close(
            module.scale_lo.grad,
            (upstream * signs * (codes.abs() == 1)).sum(-1, keepdim=True) * low,
        )
        torch.testing.assert_close(
            module.scale.grad,
            (upstream * signs * (codes.abs() > 1)).sum(-1, keepdim=True) * high,
        )

    def test_the_public_ste_follows_the_ordering_its_forward_applied(self):
        """The forward reads the pair as (min, max); backward must undo that too."""
        from axolotl.integrations.ternary.quant import fake_quant_weight_dual_ste

        weight = _linear().weight.detach()
        low, high = dual_absmean_scales(weight)

        def _grads(first: torch.Tensor, second: torch.Tensor):
            first = first.clone().requires_grad_(True)
            second = second.clone().requires_grad_(True)
            out = fake_quant_weight_dual_ste(weight, 1.0, first, second)
            out.backward(torch.ones_like(out))
            return first.grad, second.grad

        ordered = _grads(low, high)
        inverted = _grads(high, low)

        assert not torch.equal(ordered[0], ordered[1])
        torch.testing.assert_close(inverted[0], ordered[1])
        torch.testing.assert_close(inverted[1], ordered[0])

    def test_a_scale_below_the_floor_does_not_kill_the_layer(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="dual")
        with torch.no_grad():
            module.scale_lo.fill_(torch.tensor(1e-9).log())
            module.scale.fill_(torch.tensor(1e-9).log())

        baked = module.baked_weight()

        assert float(baked.abs().max()) > 0.0
        low, high = module._dual_scales(module.weight)
        assert float(low.detach().min()) == pytest.approx(SCALE_EPS)
        assert float(high.detach().min()) == pytest.approx(SCALE_EPS)


class TestMonitoring:
    def test_row_snapshot_packs_the_row_codes(self):
        linear = _linear()
        module = TernaryLinear.from_linear(linear, weight_scale="learnable_row")

        packed = module.code_snapshot()

        assert module.code_count() == linear.weight.numel()
        scale = linear.weight.detach().abs().mean(-1, keepdim=True)
        codes = torch.round(linear.weight.detach() / scale).clamp(-1, 1).to(torch.int8)
        assert torch.equal(unpack_codes(packed, tuple(codes.shape)), codes)

    def test_dual_snapshot_covers_both_planes(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="dual")

        packed = module.code_snapshot()

        assert module.code_count() == module.weight.numel() * DUAL_PLANES
        assert packed.numel() == (module.code_count() + 3) // 4
        low, high = module._dual_scales(module.weight)
        codes = dual_codes(module.weight.detach(), low, high)
        planes = unpack_codes(packed, (DUAL_PLANES, OUT_FEATURES, IN_FEATURES))
        assert torch.equal(planes, dual_state_planes(codes))

    def test_dual_zero_fraction_is_the_share_of_zeroed_weights(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="dual")
        low, high = module._dual_scales(module.weight)
        codes = dual_codes(module.weight.detach(), low, high)

        fraction = zero_fraction(module.code_snapshot(), module.code_count())

        assert 0.0 < float(fraction) < 1.0
        assert float(fraction) == pytest.approx(
            (codes == 0).to(torch.float32).mean().item()
        )

    def test_dual_flip_count_sees_a_state_change(self):
        module = TernaryLinear.from_linear(_linear(), weight_scale="dual")
        before = module.code_snapshot()

        with torch.no_grad():
            module.scale_lo.mul_(2.0)

        flips = flip_count(before, module.code_snapshot())

        assert int(flips) > 0

    @pytest.mark.parametrize("mode", ROW_MODES)
    def test_the_monitor_callback_runs(self, mode):
        model = _tiny_llama()
        convert_model(model, DictDefault({"ternary": _master_only(mode)}))
        callback = TernaryMonitorCallback(model, every_n_steps=1)

        callback.on_step_end(None, TrainerState(global_step=1), TrainerControl())

        assert 0.0 <= callback.metrics["ternary/zero_frac"] <= 1.0
        assert "ternary/zero_frac/q_proj" in callback.metrics


class TestDispatch:
    def test_every_schema_mode_is_implemented(self):
        assert UNIMPLEMENTED_SCALE_MODES == frozenset()

    def test_unknown_modes_are_still_rejected(self):
        with pytest.raises(ValueError, match="unknown ternary weight_scale"):
            TernaryLinear(IN_FEATURES, OUT_FEATURES, weight_scale="int4")

    @pytest.mark.parametrize("mode", ROW_MODES)
    def test_a_group_size_is_rejected(self, mode):
        with pytest.raises(ValueError, match="only valid with weight_scale"):
            TernaryLinear(IN_FEATURES, OUT_FEATURES, weight_scale=mode, group_size=8)

    @pytest.mark.parametrize("mode", ["absmean", "group", *ROW_MODES])
    def test_the_fused_weight_kernels_only_take_the_grids_they_implement(self, mode):
        sentinel = object()
        module = TernaryLinear(
            IN_FEATURES,
            OUT_FEATURES,
            weight_scale=mode,
            group_size=8 if mode == "group" else None,
        )
        module._ops = lambda tensor: sentinel

        taken = module._weight_ops(torch.zeros(1))

        assert (taken is sentinel) == (mode in FUSED_SCALE_MODES)

    @pytest.mark.parametrize("mode", ["absmean", *ROW_MODES])
    def test_the_int8_forward_only_takes_a_per_tensor_scale(self, mode, monkeypatch):
        marker = torch.zeros(1)
        module = TernaryLinear.from_linear(_linear(), weight_scale=mode)
        monkeypatch.setattr(
            "axolotl.integrations.ternary.modules._int8_ops",
            lambda: type(
                "_Ops",
                (),
                {"int8_linear_forward": staticmethod(lambda module, x: marker)},
            ),
        )

        out = module._int8_linear(torch.zeros(2, IN_FEATURES))

        assert (out is marker) == (mode in INT8_FORWARD_SCALE_MODES)

    def test_a_per_tensor_module_is_untouched_by_the_new_modes(self):
        """The two implemented grids must not perturb the default path."""
        linear = _linear()
        module = TernaryLinear.from_linear(copy.deepcopy(linear))
        x = torch.randn(2, IN_FEATURES)

        scale = linear.weight.detach().float().abs().mean()
        codes = torch.round(linear.weight.detach() / scale).clamp(-1.0, 1.0)
        expected = F.linear(_oracle_act(x), codes * scale.half().float())

        assert module.scale is None and module.scale_lo is None
        assert module.code_count() == linear.weight.numel()
        torch.testing.assert_close(module(x), expected, rtol=1e-6, atol=1e-6)


class TestSwapAndExport:
    @pytest.mark.parametrize("mode", ROW_MODES)
    def test_convert_model_records_the_mode(self, mode):
        model = _tiny_llama()

        manifest = convert_model(model, DictDefault({"ternary": _master_only(mode)}))

        assert manifest.weight_scale == mode
        assert manifest.group_size is None
        assert {entry.weight_scale for entry in manifest.entries} == {mode}
        modules = dict(iter_ternary_modules(model))
        assert modules
        assert all(module.weight_scale == mode for module in modules.values())

    @pytest.mark.parametrize("mode", ROW_MODES)
    def test_a_converted_model_trains_and_bakes(self, mode):
        model = _tiny_llama()
        convert_model(model, DictDefault({"ternary": _master_only(mode)}))
        input_ids = torch.randint(0, 128, (2, 8))

        model(input_ids, labels=input_ids).loss.backward()
        for name, module in list(iter_ternary_modules(model)):
            assert module.weight.grad is not None
            module._post_training(model, name)

        with torch.no_grad():
            logits = model(input_ids).logits
        assert torch.isfinite(logits).all()
        for _, module in iter_ternary_modules(model):
            assert module.is_baked()
            assert module.scale is None and module.scale_lo is None

    @pytest.mark.parametrize("mode", ROW_MODES)
    def test_only_the_master_export_accepts_the_mode(self, mode):
        assert (
            TernaryConfig(
                weight_scale=mode, export={"formats": ["master_bf16"]}
            ).weight_scale
            == mode
        )

        with pytest.raises(ValueError, match="cannot be represented"):
            TernaryConfig(weight_scale=mode, export={"formats": ["hf_bitnet"]})


@pytest.fixture(name="mesh", scope="module")
def fixture_mesh():
    """A single-rank gloo device mesh; the process group is torn down after."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29734")
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


class TestSharded:
    """Single-rank gloo mesh: the row scales must survive `full_tensor`/`to_local`."""

    @pytest.mark.parametrize("mode", ROW_MODES)
    def test_a_sharded_row_scale_bakes_like_the_unsharded_one(self, mesh, mode):
        from torch.distributed.tensor import Shard, distribute_tensor

        plain = TernaryLinear.from_linear(_linear(), weight_scale=mode)
        sharded = TernaryLinear.from_linear(_linear(), weight_scale=mode)
        for attr in ("weight", "scale", "scale_lo"):
            parameter = getattr(sharded, attr)
            if parameter is not None:
                setattr(
                    sharded,
                    attr,
                    nn.Parameter(
                        distribute_tensor(parameter.detach(), mesh, [Shard(0)])
                    ),
                )

        assert torch.equal(sharded.code_snapshot(), plain.code_snapshot())
        assert sharded.code_count() == plain.code_count()

        sharded._post_training(None, "q_proj")
        plain._post_training(None, "q_proj")

        assert torch.equal(as_local(sharded.weight.detach()), plain.weight.detach())
        assert sharded.is_baked()


@pytest.mark.parametrize("mode", ROW_MODES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_forward_matches_the_cpu_oracle(mode):
    linear = _linear()
    module = TernaryLinear.from_linear(copy.deepcopy(linear), weight_scale=mode).cuda()
    reference = TernaryLinear.from_linear(linear, weight_scale=mode)
    x = torch.randn(4, IN_FEATURES)

    out = module(x.cuda()).cpu()

    torch.testing.assert_close(out, reference(x), rtol=1e-5, atol=1e-5)


def _master_only(mode: str) -> dict:
    """A ternary block in `mode`; the packed formats cannot carry these grids."""
    return {"weight_scale": mode, "export": {"formats": ["master_bf16"]}}


def _linear_holding(weight: torch.Tensor) -> nn.Linear:
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    return linear
