"""CPU-only golden tests for the eager ternary quantizer (the semantic oracle)."""

import pytest
import torch

from axolotl.integrations.ternary.quant import (
    ACT_QMAX,
    CODES_PER_BYTE,
    SCALE_EPS,
    absmean_scale,
    act_quant,
    act_quant_int8,
    act_quant_ste,
    act_scale,
    baked_codes_and_scale,
    dequantize_codes,
    f16_round_scale,
    fake_quant_weight,
    fake_quant_weight_ste,
    flip_count,
    pack_codes,
    ternary_codes,
    unpack_codes,
    zero_fraction,
)

# absmean = 1.0 exactly, so w/s hits the round-half-even ties 0.5 and 1.5
TIE_WEIGHT = torch.tensor([[0.5, 1.5, -0.5, -1.5]])


class TestWeightScale:
    def test_absmean_golden(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        scale = absmean_scale(weight)

        assert scale.dtype is torch.float32
        assert scale.shape == torch.Size([])
        assert scale.item() == 2.5

    def test_absmean_accumulates_in_fp32(self):
        weight = torch.full((2, 4), 0.1, dtype=torch.bfloat16)

        scale = absmean_scale(weight)

        assert scale.dtype is torch.float32
        assert scale.item() == pytest.approx(
            weight.to(torch.float32).abs().mean().item()
        )

    def test_clamp_floor_on_all_zero_weight(self):
        weight = torch.zeros(3, 4)

        scale = absmean_scale(weight)

        assert scale.item() == pytest.approx(SCALE_EPS)
        assert torch.equal(
            ternary_codes(weight, scale), torch.zeros(3, 4, dtype=torch.int8)
        )
        assert torch.equal(fake_quant_weight(weight), torch.zeros(3, 4))

    def test_group_scale_shape_and_values(self):
        weight = torch.tensor([[1.0, 3.0, 0.5, 0.5]])

        scale = absmean_scale(weight, group_size=2)

        assert scale.shape == (1, 2)
        assert scale.tolist() == [[2.0, 0.5]]

    def test_group_size_must_divide(self):
        with pytest.raises(ValueError, match="does not divide"):
            absmean_scale(torch.zeros(2, 6), group_size=4)

    def test_f16_round_scale_golden(self):
        scale = torch.tensor(0.1)

        rounded = f16_round_scale(scale)

        assert rounded.item() == 0.0999755859375
        assert rounded.dtype is torch.float32

    def test_f16_round_scale_keeps_eps_nonzero(self):
        assert f16_round_scale(torch.tensor(SCALE_EPS)).item() > 0.0


class TestTernaryCodes:
    def test_round_half_even_ties(self):
        scale = absmean_scale(TIE_WEIGHT)
        assert scale.item() == 1.0

        codes = ternary_codes(TIE_WEIGHT, scale)

        # 0.5 -> 0 (half to even), 1.5 -> 2 -> clipped to 1
        assert codes.dtype is torch.int8
        assert codes.tolist() == [[0, 1, 0, -1]]

    def test_clipping_to_ternary(self):
        weight = torch.tensor([[3.0, -3.0, 0.0]])

        codes = ternary_codes(weight, torch.tensor(1.0))

        assert codes.tolist() == [[1, -1, 0]]

    def test_explicit_scale_is_clamped(self):
        weight = torch.tensor([[1.0, -1.0]])

        codes = ternary_codes(weight, torch.tensor(0.0))

        assert codes.tolist() == [[1, -1]]

    def test_group_scale_codes(self):
        weight = torch.tensor([[4.0, 4.0, 0.1, 1.0]])
        scale = absmean_scale(weight, group_size=2)

        codes = ternary_codes(weight, scale)

        assert codes.tolist() == [[1, 1, 0, 1]]

    def test_dequantize_uses_f16_scale_and_dtype(self):
        codes = torch.tensor([[1, 0, -1]], dtype=torch.int8)
        scale_f16 = f16_round_scale(torch.tensor(0.1))

        values = dequantize_codes(codes, scale_f16, torch.float32)

        assert values.dtype is torch.float32
        assert values.tolist() == [[0.0999755859375, 0.0, -0.0999755859375]]


class TestRequantizationInvariant:
    def test_codes_survive_a_round_trip_through_the_baked_master(self):
        torch.manual_seed(0)
        weight = torch.randn(32, 64)
        scale = absmean_scale(weight)
        codes = ternary_codes(weight, scale)

        master = dequantize_codes(codes, f16_round_scale(scale), torch.bfloat16)
        rederived = ternary_codes(master, absmean_scale(master))

        assert torch.equal(rederived, codes)

    def test_baked_values_are_exactly_representable_in_f16(self):
        torch.manual_seed(0)
        weight = torch.randn(8, 16)
        scale = absmean_scale(weight)
        codes = ternary_codes(weight, scale)

        master = dequantize_codes(codes, f16_round_scale(scale), torch.float32)

        assert torch.equal(master.to(torch.float16).to(torch.float32), master)

    def test_master_dequantizes_exactly_from_its_recovered_scale(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 32)
        scale = absmean_scale(weight)
        codes = ternary_codes(weight, scale)
        master = dequantize_codes(codes, f16_round_scale(scale), torch.bfloat16)

        recovered = f16_round_scale(master.to(torch.float32).abs().amax())

        assert torch.equal(dequantize_codes(codes, recovered, torch.bfloat16), master)


class TestFakeQuantWeight:
    def test_lambda_one_is_pure_fake_quant(self):
        torch.manual_seed(0)
        weight = torch.randn(4, 8, dtype=torch.bfloat16)
        scale = absmean_scale(weight)
        expected = dequantize_codes(
            ternary_codes(weight, scale), f16_round_scale(scale), torch.bfloat16
        )

        out = fake_quant_weight(weight, lambda_=1.0)

        assert out.dtype is torch.bfloat16
        assert torch.equal(out, expected)

    def test_lambda_zero_is_identity(self):
        torch.manual_seed(0)
        weight = torch.randn(4, 8)

        assert torch.equal(fake_quant_weight(weight, lambda_=0.0), weight)

    def test_lambda_midpoint_interpolates(self):
        out = fake_quant_weight(TIE_WEIGHT, lambda_=0.5)

        # codes [0, 1, 0, -1] * s16 1.0, halfway from [0.5, 1.5, -0.5, -1.5]
        assert out.tolist() == [[0.25, 1.25, -0.25, -1.25]]

    def test_explicit_scale_overrides_absmean(self):
        weight = torch.tensor([[2.0, 2.0, 2.0, 2.0]])

        out = fake_quant_weight(weight, scale=torch.tensor(4.0))

        assert out.tolist() == [[0.0, 0.0, 0.0, 0.0]]

    def test_group_mode_matches_per_group_reference(self):
        torch.manual_seed(0)
        weight = torch.randn(4, 8)

        out = fake_quant_weight(weight, group_size=4)

        expected = torch.stack(
            [
                torch.cat(
                    [
                        fake_quant_weight(row[None, :4]),
                        fake_quant_weight(row[None, 4:]),
                    ],
                    dim=-1,
                )[0]
                for row in weight
            ]
        )
        assert torch.equal(out, expected)


class TestBakedDetection:
    def test_a_latent_weight_is_not_baked(self):
        torch.manual_seed(0)
        assert baked_codes_and_scale(torch.randn(8, 16)) is None

    def test_a_weight_baked_beyond_the_probe_window_is_still_detected(self):
        """The cheap probe only rules bakedness out; it must never rule it in."""
        from axolotl.integrations.ternary.quant import BAKED_PROBE_ELEMENTS

        torch.manual_seed(0)
        baked = fake_quant_weight(torch.randn(4, BAKED_PROBE_ELEMENTS * 4), 1.0)

        assert baked_codes_and_scale(baked) is not None

        latent = baked.clone()
        latent[-1, -1] += 1e-3
        assert baked_codes_and_scale(latent) is None

    def test_a_baked_weight_yields_its_amax_not_its_absmean(self):
        torch.manual_seed(0)
        baked = fake_quant_weight(torch.randn(8, 16) * 0.02, 1.0)

        codes, scale = baked_codes_and_scale(baked)

        assert set(codes.unique().tolist()) <= {-1, 0, 1}
        assert float(scale) == pytest.approx(float(baked.abs().max()))
        assert float(scale) > float(baked.abs().mean())
        assert torch.equal(dequantize_codes(codes, scale, baked.dtype), baked)

    def test_a_baked_group_weight_recovers_one_scale_per_group(self):
        torch.manual_seed(0)
        baked = fake_quant_weight(torch.randn(4, 32), 1.0, 8)

        codes, scale = baked_codes_and_scale(baked, 8)

        assert scale.shape == (4, 4)
        assert torch.equal(dequantize_codes(codes, scale, baked.dtype), baked)

    def test_detection_is_a_fixed_point(self):
        torch.manual_seed(0)
        baked = fake_quant_weight(torch.randn(8, 16) * 0.02, 1.0)

        codes, scale = baked_codes_and_scale(baked)
        again = dequantize_codes(codes, scale, baked.dtype)

        assert torch.equal(again, baked)
        assert baked_codes_and_scale(again) is not None


class TestActivationQuant:
    def test_scale_shape_and_value(self):
        x = torch.tensor([[-4.0, 4.0, 0.0, 1.0]])

        scale = act_scale(x)

        assert scale.shape == (1, 1)
        assert scale.item() == pytest.approx(4.0 / ACT_QMAX)

    def test_symmetric_extremes_hit_plus_minus_qmax(self):
        x = torch.tensor([[-4.0, 4.0, 0.0, 1.0]])

        codes, scale = act_quant_int8(x)

        assert codes.dtype is torch.int8
        assert codes.tolist() == [[-ACT_QMAX, ACT_QMAX, 0, 32]]
        assert scale.item() == pytest.approx(4.0 / ACT_QMAX)

    def test_scale_floor_on_all_zero_activation(self):
        x = torch.zeros(2, 4)

        codes, scale = act_quant_int8(x)

        assert scale.unique().tolist() == pytest.approx([SCALE_EPS])
        assert codes.abs().sum().item() == 0

    def test_per_token_scales_are_independent(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 100.0]])

        codes, _ = act_quant_int8(x)

        assert codes.tolist() == [[ACT_QMAX, 0], [0, ACT_QMAX]]

    def test_lambda_endpoints(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        codes, scale = act_quant_int8(x)

        assert torch.equal(act_quant(x, lambda_=0.0), x)
        assert torch.equal(
            act_quant(x, lambda_=1.0), (codes.to(torch.float32) * scale).to(x.dtype)
        )

    def test_lambda_midpoint_interpolates(self):
        x = torch.tensor([[-4.0, 4.0, 0.0, 1.0]])
        codes, scale = act_quant_int8(x)

        out = act_quant(x, lambda_=0.5)

        assert torch.allclose(out, x + 0.5 * (codes.to(torch.float32) * scale - x))


class TestPacking:
    def test_byte_layout_golden(self):
        codes = torch.tensor([-1, 0, 1, 0], dtype=torch.int8)

        packed = pack_codes(codes)

        assert packed.dtype is torch.uint8
        # (code + 1) in little-endian 2-bit lanes: 0 | 1<<2 | 2<<4 | 1<<6
        assert packed.tolist() == [0 | (1 << 2) | (2 << 4) | (1 << 6)]

    def test_roundtrip_with_padded_tail(self):
        torch.manual_seed(0)
        codes = torch.randint(-1, 2, (5, 7), dtype=torch.int8)

        packed = pack_codes(codes)

        assert packed.numel() == (codes.numel() + CODES_PER_BYTE - 1) // CODES_PER_BYTE
        assert torch.equal(unpack_codes(packed, tuple(codes.shape)), codes)

    def test_roundtrip_exact_multiple(self):
        codes = torch.tensor([[1, -1, 0, 1], [0, 0, -1, 1]], dtype=torch.int8)

        assert torch.equal(unpack_codes(pack_codes(codes), (2, 4)), codes)

    def test_flip_count(self):
        torch.manual_seed(0)
        prev = torch.randint(-1, 2, (3, 9), dtype=torch.int8)
        now = prev.clone()
        now[0, 0] = -now[0, 0] if now[0, 0] != 0 else 1
        now[2, 8] = 1 - now[2, 8]

        count = flip_count(pack_codes(prev), pack_codes(now))

        assert count.dtype is torch.int64
        assert count.item() == int((prev != now).sum())

    def test_flip_count_ignores_padding(self):
        codes = torch.randint(-1, 2, (7,), dtype=torch.int8)
        packed = pack_codes(codes)

        assert flip_count(packed, packed).item() == 0

    def test_flip_count_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            flip_count(
                pack_codes(torch.zeros(4, dtype=torch.int8)),
                pack_codes(torch.zeros(8, dtype=torch.int8)),
            )

    def test_zero_fraction(self):
        torch.manual_seed(0)
        codes = torch.randint(-1, 2, (6, 7), dtype=torch.int8)

        fraction = zero_fraction(pack_codes(codes), codes.numel())

        assert fraction.item() == pytest.approx(
            (codes == 0).to(torch.float32).mean().item()
        )

    def test_zero_fraction_all_zero_codes(self):
        codes = torch.zeros(9, dtype=torch.int8)

        assert zero_fraction(pack_codes(codes), codes.numel()).item() == 1.0


class TestStraightThrough:
    @pytest.mark.parametrize("lambda_", [0.0, 0.5, 1.0])
    def test_weight_gradient_is_the_upstream_gradient(self, lambda_):
        torch.manual_seed(0)
        weight = torch.randn(4, 8, requires_grad=True)
        grad = torch.randn(4, 8)

        fake_quant_weight_ste(weight, lambda_).backward(grad)

        assert torch.equal(weight.grad, grad)

    @pytest.mark.parametrize("lambda_", [0.0, 0.5, 1.0])
    def test_activation_gradient_is_the_upstream_gradient(self, lambda_):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 8, requires_grad=True)
        grad = torch.randn(2, 3, 8)

        act_quant_ste(x, lambda_).backward(grad)

        assert torch.equal(x.grad, grad)

    def test_ste_forward_matches_the_eager_oracle(self):
        torch.manual_seed(0)
        weight = torch.randn(4, 8, requires_grad=True)
        x = torch.randn(2, 8, requires_grad=True)

        assert torch.equal(
            fake_quant_weight_ste(weight, 0.3),
            fake_quant_weight(weight.detach(), 0.3),
        )
        assert torch.equal(act_quant_ste(x, 0.3), act_quant(x.detach(), 0.3))

    def test_learnable_scale_gradient(self):
        torch.manual_seed(0)
        weight = torch.randn(4, 8)
        scale = torch.tensor(0.7, requires_grad=True)
        grad = torch.randn(4, 8)

        fake_quant_weight_ste(weight, 0.5, None, scale).backward(grad)

        codes = ternary_codes(weight, scale.detach())
        assert scale.grad.item() == pytest.approx(
            (grad * codes.to(torch.float32) * 0.5).sum().item(), rel=1e-5
        )

    def test_an_explicit_scale_is_floored_before_the_f16_dequant(self):
        """An f16-subnormal scale would round to exactly zero and kill the layer."""
        torch.manual_seed(0)
        weight = torch.randn(4, 8)

        quantized = fake_quant_weight(weight, 1.0, None, torch.tensor(1e-9))

        assert float(quantized.abs().max()) > 0.0
        # the codes and the dequant scale share the one floored value
        floored = fake_quant_weight(weight, 1.0, None, torch.tensor(SCALE_EPS))
        assert torch.equal(quantized, floored)

    def test_learnable_group_scale_gradient_shape(self):
        torch.manual_seed(0)
        weight = torch.randn(4, 8)
        scale = absmean_scale(weight, group_size=4).clone().requires_grad_(True)

        fake_quant_weight_ste(weight, 1.0, 4, scale).backward(torch.randn(4, 8))

        assert scale.grad.shape == scale.shape
