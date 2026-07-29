"""CPU-only tests for `codebook: binary` — the `{-s, +s}` sign plane, end to end.

The binary codebook is the ternary one with the zero state removed, which changes three
things and nothing else: the scale is the *exact* least-squares optimum rather than a
threshold heuristic, the fit converges in a single pass because the assignment does not
depend on the scale, and a baked tensor holds one magnitude instead of two. Everything
downstream — the f16-rounded dequant scale, the exact-requantization fixed point, the
bake-after-training invariant — is inherited unchanged, and these tests say so.
"""

from contextlib import contextmanager

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.export import bake
from axolotl.integrations.ternary.modules import (
    TernaryLinear,
    assert_codebook_applied,
    iter_ternary_modules,
)
from axolotl.integrations.ternary.ptq.calibrate import aga_refit, gptq_compensate
from axolotl.integrations.ternary.ptq.ternary_fit import fit_binary, fit_ternary
from axolotl.integrations.ternary.swap import SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

SINGLE_PLANE_MODES = ("absmean", "learnable", "learnable_row", "group")


@contextmanager
def binary_swap():
    """Make `convert_model` build binary modules whether or not the swap plumbs it.

    Inert now that `swap.convert_model` passes `ternary.codebook` itself: `setdefault`
    leaves the real argument in charge. Kept so these tests still pin the module-level
    behaviour if the swap's plumbing is ever the thing under repair.
    """
    original = TernaryLinear.from_linear

    def from_linear(linear, **kwargs):
        kwargs.setdefault("codebook", "binary")
        return original(linear, **kwargs)

    TernaryLinear.from_linear = from_linear  # type: ignore[method-assign]
    try:
        yield
    finally:
        TernaryLinear.from_linear = original  # type: ignore[method-assign]


def write_binary_master(directory, **export) -> SwapManifest:
    """Convert, bake and save a `codebook: binary` master, as a healed run does."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).to(torch.bfloat16)
    with binary_swap():
        manifest = convert_model(
            model,
            DictDefault(
                {
                    "output_dir": str(directory),
                    "ternary": {"codebook": "binary", "export": export},
                }
            ),
        )
    for name, module in iter_ternary_modules(model):
        module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    bake.write_quantizer_metadata(directory, manifest)
    return manifest


@pytest.fixture(name="converted_binary")
def fixture_converted_binary():
    with binary_swap():
        yield


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
    )


def _frobenius(weight: torch.Tensor, scale: float | torch.Tensor) -> float:
    codes = quant.binary_codes(weight, torch.tensor(1.0)).to(torch.float32)
    return float((weight - codes * scale).norm())


# ------------------------------------------------------------------ the quantizer


class TestBinaryScale:
    def test_the_scale_is_the_closed_form_absmean(self):
        weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]])

        assert float(quant.binary_scale(weight)) == 2.5

    def test_it_is_the_least_squares_optimum_not_a_heuristic(self):
        """Perturbing the scale in either direction only grows the reconstruction error."""
        torch.manual_seed(0)
        weight = torch.randn(16, 32)
        best = float(quant.binary_scale(weight))

        error = _frobenius(weight, best)

        for factor in (0.5, 0.9, 0.99, 1.01, 1.1, 2.0):
            assert _frobenius(weight, best * factor) > error

    def test_the_floor_survives_an_all_zero_tensor(self):
        scale = quant.binary_scale(torch.zeros(3, 4))

        assert float(scale) == pytest.approx(quant.SCALE_EPS)

    def test_group_mode_scales_each_block(self):
        weight = torch.tensor([[1.0, 3.0, 0.5, 0.5]])

        assert quant.binary_scale(weight, group_size=2).tolist() == [[2.0, 0.5]]


class TestBinaryCodes:
    def test_every_weight_takes_a_sign_and_zero_takes_the_positive_one(self):
        weight = torch.tensor([[1e-9, -1e-9, 0.0, -0.0, 5.0, -5.0]])

        codes = quant.binary_codes(weight, quant.binary_scale(weight))

        assert codes.dtype is torch.int8
        assert codes.tolist() == [[1, -1, 1, 1, 1, -1]]

    def test_the_assignment_does_not_depend_on_the_scale(self):
        torch.manual_seed(0)
        weight = torch.randn(8, 8)

        for scale in (1e-4, 1.0, 1e4):
            assert torch.equal(
                quant.binary_codes(weight, torch.tensor(scale)),
                quant.binary_codes(weight, quant.binary_scale(weight)),
            )


class TestBinaryFakeQuant:
    def test_the_values_are_the_f16_rounded_scale_times_the_signs(self):
        weight = torch.tensor([[0.1, -0.3]])
        scale = quant.f16_round_scale(quant.binary_scale(weight))

        quantized = quant.fake_quant_weight_binary(weight)

        assert quantized.tolist() == [[float(scale), -float(scale)]]

    def test_no_value_is_ever_zero(self):
        torch.manual_seed(0)
        weight = torch.randn(32, 32)
        weight[3, 4] = 0.0

        assert not bool((quant.fake_quant_weight_binary(weight) == 0).any())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_requantizing_a_quantized_tensor_is_a_fixed_point(self, dtype):
        torch.manual_seed(0)
        weight = torch.randn(16, 24, dtype=dtype)

        once = quant.fake_quant_weight_binary(weight)
        twice = quant.fake_quant_weight_binary(once)

        assert twice.dtype is dtype
        assert torch.equal(once, twice)

    def test_lambda_interpolates_towards_the_grid(self):
        weight = torch.tensor([[1.0, -3.0]])

        half = quant.fake_quant_weight_binary(weight, 0.5)

        assert torch.equal(quant.fake_quant_weight_binary(weight, 0.0), weight)
        assert torch.allclose(
            half, 0.5 * (weight + quant.fake_quant_weight_binary(weight))
        )

    def test_an_explicit_scale_is_floored_like_the_derived_one(self):
        weight = torch.tensor([[1.0, -1.0]])
        floor = quant.f16_round_scale(torch.tensor(quant.SCALE_EPS))

        quantized = quant.fake_quant_weight_binary(weight, scale=torch.tensor(0.0))

        assert float(quantized[0, 0]) == float(floor)


class TestBakedBinaryProbe:
    def test_a_latent_tensor_is_not_baked(self):
        torch.manual_seed(0)

        assert quant.baked_binary_codes_and_scale(torch.randn(8, 8)) is None

    def test_a_quantized_tensor_recovers_its_codes_and_scale(self):
        torch.manual_seed(0)
        weight = torch.randn(8, 8)
        baked = quant.fake_quant_weight_binary(weight)

        codes, scale = quant.baked_binary_codes_and_scale(baked)

        assert torch.equal(codes, quant.binary_codes(weight, scale))
        assert float(scale) == float(baked.abs().amax())

    def test_a_row_of_one_sign_is_still_baked(self):
        """The counter-sign is not evidence: a degenerate tensor is on the grid too."""
        baked = torch.full((4, 6), 0.25)

        recovered = quant.baked_binary_codes_and_scale(baked)

        assert recovered is not None
        assert bool((recovered[0] == 1).all())
        assert float(recovered[1]) == 0.25

    def test_a_single_zero_makes_the_tensor_latent(self):
        """A zero is not on the binary grid, so a ternary master can never pass here."""
        baked = quant.fake_quant_weight_binary(torch.randn(8, 8))
        baked[2, 3] = 0.0

        assert quant.baked_binary_codes_and_scale(baked) is None

    def test_a_ternary_master_is_rejected(self):
        torch.manual_seed(0)
        ternary = quant.fake_quant_weight(torch.randn(32, 32))

        assert bool((ternary == 0).any())
        assert quant.baked_binary_codes_and_scale(ternary) is None

    def test_group_mode_recovers_one_scale_per_block(self):
        torch.manual_seed(0)
        baked = quant.fake_quant_weight_binary(torch.randn(4, 16), group_size=8)

        codes, scale = quant.baked_binary_codes_and_scale(baked, group_size=8)

        assert codes.shape == (4, 16)
        assert scale.shape == (4, 2)
        assert torch.equal(quant.dequantize_codes(codes, scale, baked.dtype), baked)


class TestBinarySTE:
    def test_the_gradient_reaches_the_latent_unchanged(self):
        torch.manual_seed(0)
        weight = torch.randn(4, 8, requires_grad=True)

        quant.fake_quant_weight_binary_ste(weight).mul(
            torch.arange(8.0)
        ).sum().backward()

        assert torch.equal(weight.grad, torch.arange(8.0).expand(4, 8))

    def test_a_learnable_scale_gets_the_true_gradient(self):
        torch.manual_seed(0)
        weight = torch.randn(4, 8)
        scale = torch.tensor(0.3, requires_grad=True)
        codes = quant.binary_codes(weight, scale).to(torch.float32)

        quant.fake_quant_weight_binary_ste(weight, 1.0, None, scale).sum().backward()

        assert float(scale.grad) == pytest.approx(float(codes.sum()))


# ---------------------------------------------------------------------- the module


class TestBinaryModule:
    @pytest.mark.parametrize("weight_scale", SINGLE_PLANE_MODES)
    def test_the_forward_quantizes_onto_the_sign_grid(self, weight_scale):
        torch.manual_seed(0)
        group_size = 8 if weight_scale == "group" else None
        module = TernaryLinear(
            16, 8, weight_scale=weight_scale, codebook="binary", group_size=group_size
        )
        nn.init.normal_(module.weight)
        module.refresh_scale_from_weight()

        effective = module._quant_weight(1.0)

        assert not bool((effective == 0).any())
        assert (
            quant.baked_binary_codes_and_scale(effective, module._scale_group_size())
            is not None
        )

    @pytest.mark.parametrize("weight_scale", ["dual", "trit_planes"])
    def test_the_two_plane_grids_are_refused(self, weight_scale):
        with pytest.raises(ValueError, match="cannot be combined"):
            TernaryLinear(16, 8, weight_scale=weight_scale, codebook="binary")

    def test_an_unknown_codebook_is_refused(self):
        with pytest.raises(ValueError, match="unknown ternary codebook"):
            TernaryLinear(16, 8, codebook="quaternary")

    def test_the_repr_names_the_codebook_only_when_it_is_not_the_default(self):
        assert "codebook=binary" in repr(TernaryLinear(4, 2, codebook="binary"))
        assert "codebook" not in repr(TernaryLinear(4, 2))

    def test_the_bake_lands_on_the_sign_grid_and_lights_the_flag(self):
        torch.manual_seed(0)
        linear = nn.Linear(16, 8, bias=False)
        module = TernaryLinear.from_linear(linear, codebook="binary")

        baked = module.baked_weight()
        with torch.no_grad():
            module.weight.copy_(baked)
        module._detect_baked()

        assert module.is_baked()
        assert quant.baked_binary_codes_and_scale(baked) is not None

    def test_a_reloaded_binary_master_is_detected_as_baked(self):
        torch.manual_seed(0)
        module = TernaryLinear.from_linear(
            nn.Linear(16, 8, bias=False), weight_scale="learnable", codebook="binary"
        )
        with torch.no_grad():
            module.weight.copy_(module.baked_weight())
        module._detect_baked()
        module.scale = None
        module.refresh_scale_from_weight()

        # the scale a binary master carries is recoverable: every magnitude *is* the
        # scale, where a ternary reconstruction's absmean is the scale times its density
        assert float(module.quantization_scale().detach()) == pytest.approx(
            float(module.weight.detach().abs().amax()), rel=1e-6
        )

    def test_the_statistic_mode_re_derives_a_binary_master_exactly(self):
        """`absmean` of a binary master *is* its scale; of a ternary one it is s·density.

        Which is why the schema's fitting-init restriction to the learned scale modes
        is a ternary constraint rather than a universal one.
        """
        torch.manual_seed(0)
        module = TernaryLinear.from_linear(
            nn.Linear(16, 8, bias=False), codebook="binary"
        )
        with torch.no_grad():
            module.weight.copy_(module.baked_weight())

        assert torch.equal(module._quant_weight(1.0), module.weight.detach())

    def test_the_snapshot_reports_a_zero_fraction_of_zero(self):
        torch.manual_seed(0)
        module = TernaryLinear.from_linear(
            nn.Linear(16, 8, bias=False), codebook="binary"
        )

        packed = module.code_snapshot()

        assert float(quant.zero_fraction(packed, module.code_count())) == 0.0

    def test_the_snapshot_counts_two_state_flips(self):
        torch.manual_seed(0)
        module = TernaryLinear.from_linear(
            nn.Linear(16, 8, bias=False), codebook="binary"
        )
        before = module.code_snapshot()
        with torch.no_grad():
            module.weight[0, :4] = module.weight[0, :4].abs().add(1.0).neg()
            module.weight[1, :2] = module.weight[1, :2].abs().add(1.0)

        flips = int(quant.flip_count(before, module.code_snapshot()))

        # only the weights whose sign was forced can have moved, and a two-state code
        # moves at most once
        assert 0 < flips <= 6

    def test_the_int8_forward_and_the_fused_kernels_stay_ternary(self):
        module = TernaryLinear.from_linear(
            nn.Linear(16, 8, bias=False), codebook="binary", int8_forward=True
        )

        assert module._weight_ops(module.weight) is None
        assert module._int8_linear(torch.zeros(1, 16)) is None


# ------------------------------------------------------------------------ the fit


class TestBinaryFit:
    @pytest.mark.parametrize("weight_scale", SINGLE_PLANE_MODES)
    def test_the_fit_is_a_fixed_point_of_the_quantizer(self, weight_scale):
        torch.manual_seed(0)
        weight = torch.randn(8, 16)
        group_size = 8 if weight_scale == "group" else None
        # a per-row scale is one group spanning every input feature
        grid = 16 if weight_scale == "learnable_row" else group_size

        codes, scale = fit_binary(weight, weight_scale, group_size)
        reconstruction = quant.dequantize_codes(
            codes, quant.f16_round_scale(scale), torch.float32
        )

        assert torch.equal(codes, quant.binary_codes(weight, scale))
        assert torch.equal(
            quant.fake_quant_weight_binary(reconstruction, group_size=grid),
            reconstruction,
        )

    def test_the_solved_scale_is_the_frobenius_optimum(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 32)

        _, scale = fit_binary(weight)

        error = _frobenius(weight, float(scale))
        for factor in (0.8, 0.95, 1.05, 1.2):
            assert _frobenius(weight, float(scale) * factor) > error

    def test_the_ternary_fit_collapses_onto_it_when_no_weight_is_near_zero(self):
        """The zero state costs nothing and buys nothing here: the two optima coincide."""
        torch.manual_seed(0)
        weight = torch.randn(16, 64).sign() * (1.0 + torch.rand(16, 64))

        binary = fit_binary(weight, "learnable_row")
        ternary = fit_ternary(weight, "learnable_row")

        assert not bool((ternary[0] == 0).any())
        assert torch.equal(binary[0], ternary[0])
        assert self._error(weight, *binary) == self._error(weight, *ternary)

    def test_the_zero_state_is_what_the_ternary_fit_buys_on_a_gaussian(self):
        """And on a real weight distribution it is worth having — that is the trade."""
        torch.manual_seed(0)
        weight = torch.randn(16, 64)

        binary = fit_binary(weight, "learnable_row")
        ternary = fit_ternary(weight, "learnable_row")

        assert self._error(weight, *ternary) < self._error(weight, *binary)

    @staticmethod
    def _error(weight: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor) -> float:
        return float((weight - codes * quant.f16_round_scale(scale)).norm())

    def test_the_two_plane_modes_are_refused(self):
        with pytest.raises(ValueError, match="has one plane"):
            fit_binary(torch.randn(4, 8), "dual")

    def test_the_row_fit_solves_each_row_independently(self):
        weight = torch.tensor([[1.0, -1.0], [10.0, -10.0]])

        codes, scale = fit_binary(weight, "learnable_row")

        assert codes.tolist() == [[1, -1], [1, -1]]
        assert scale.reshape(-1).tolist() == [1.0, 10.0]


class TestBinaryCalibratedFit:
    def _gram(self, in_features: int, seed: int = 1) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        activations = torch.randn(256, in_features, generator=generator)
        activations[:, 0] *= 8.0
        return activations.T @ activations

    def test_the_aga_refit_keeps_the_sign_codes_and_moves_the_scale(self):
        torch.manual_seed(0)
        weight = torch.randn(8, 16)
        gram = self._gram(16)
        codes, data_free = fit_binary(weight, "learnable")

        refit = aga_refit(weight, codes, gram, "learnable")

        assert torch.equal(codes, quant.binary_codes(weight, refit))
        assert float(refit) != float(data_free)

    def test_the_refit_minimizes_the_activation_weighted_objective(self):
        torch.manual_seed(0)
        weight = torch.randn(8, 16)
        gram = self._gram(16)
        codes, _ = fit_binary(weight, "learnable")

        refit = float(aga_refit(weight, codes, gram, "learnable"))

        def objective(scale: float) -> float:
            residual = weight - codes.to(torch.float32) * scale
            return float((residual @ gram * residual).sum())

        best = objective(refit)
        for factor in (0.9, 0.98, 1.02, 1.1):
            assert objective(refit * factor) > best

    def test_the_compensated_fit_beats_the_data_free_one_on_its_own_objective(self):
        torch.manual_seed(0)
        weight = torch.randn(16, 64)
        gram = self._gram(64, seed=2)

        codes, scale = gptq_compensate(weight, gram, "learnable", codebook="binary")
        free_codes, free_scale = fit_binary(weight, "learnable")

        assert set(codes.reshape(-1).tolist()) <= {-1, 1}

        def error(module_codes, module_scale) -> float:
            residual = weight - module_codes.to(torch.float32) * float(module_scale)
            return float((residual @ gram * residual).sum())

        assert error(codes, scale) < error(free_codes, free_scale)


# ------------------------------------------------------------------- the conversion


def test_the_swap_applies_the_configured_codebook_unstubbed():
    """No `binary_swap()`: the real `convert_model` has to carry the codebook itself."""
    model = _tiny_llama()

    manifest = convert_model(model, DictDefault({"ternary": {"codebook": "binary"}}))

    assert {module.codebook for _, module in iter_ternary_modules(model)} == {"binary"}
    assert manifest.codebook == "binary"
    assert {entry.codebook for entry in manifest.entries} == {"binary"}


def test_the_swap_stamps_the_binary_quantizer_identity():
    """The stamp is what a packer reads when it has only the checkpoint on disk."""
    manifest = convert_model(
        _tiny_llama(), DictDefault({"ternary": {"codebook": "binary"}})
    )

    assert manifest.quantizer["codebook"] == "binary"
    assert manifest.quantizer["scheme"] == "binary"
    assert manifest.quantizer["codes"] == [-1, 1]


def test_a_ternary_master_reloaded_as_binary_is_refused(tmp_path):
    """Both grids share {-s, +s}, so the baked probe lights and nothing fails loudly."""
    master = tmp_path / "ternary_master"
    convert_model(
        _tiny_llama(), DictDefault({"output_dir": str(master), "ternary": {}})
    )

    with pytest.raises(ValueError, match="fitted on the ternary codebook"):
        convert_model(
            _tiny_llama(),
            DictDefault(
                {
                    "base_model": str(master),
                    "ternary": {"codebook": "binary"},
                }
            ),
        )


def test_a_binary_conversion_quantizes_every_target(converted_binary, tmp_path):
    model = _tiny_llama()

    manifest = convert_model(
        model,
        DictDefault({"output_dir": str(tmp_path), "ternary": {"codebook": "binary"}}),
    )

    assert len(manifest.entries) == 7
    for _, module in iter_ternary_modules(model):
        assert module.codebook == "binary"
        assert not bool((module._quant_weight(1.0) == 0).any())


def test_a_binary_fit_init_lands_on_the_grid_and_is_baked(converted_binary, tmp_path):
    model = _tiny_llama()

    convert_model(
        model,
        DictDefault(
            {
                "output_dir": str(tmp_path),
                "ternary": {
                    "codebook": "binary",
                    "init": "ternary_fit",
                    "weight_scale": "learnable",
                    "lambda_schedule": "none",
                },
            }
        ),
    )

    for _, module in iter_ternary_modules(model):
        assert module.is_baked()
        weight = module.weight.detach()
        assert quant.baked_binary_codes_and_scale(weight) is not None
        # the scale a fit solved survives into the parameter, exactly
        assert float(module.quantization_scale().detach()) == pytest.approx(
            float(weight.abs().amax()), rel=1e-6
        )


def test_a_binary_model_trains_and_keeps_its_codebook(converted_binary, tmp_path):
    model = _tiny_llama()
    convert_model(
        model,
        DictDefault(
            {
                "output_dir": str(tmp_path),
                "ternary": {"codebook": "binary", "lambda_schedule": "none"},
            }
        ),
    )
    ids = torch.randint(0, 64, (2, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(3):
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for _, module in iter_ternary_modules(model):
        assert not module.is_baked()
        assert not bool((module._quant_weight(1.0) == 0).any())


def test_the_codebook_assertion_names_the_seam_that_has_to_pass_it():
    model = _tiny_llama()
    convert_model(model, DictDefault({"ternary": {}}))

    with pytest.raises(RuntimeError, match="from_linear"):
        assert_codebook_applied(model, "binary")
