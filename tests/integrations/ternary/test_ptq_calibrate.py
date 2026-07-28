"""CPU-only tests for the activation-aware fit behind `init: ternary_fit_calibrated`."""

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.ptq import calibrate as calibrate_module
from axolotl.integrations.ternary.ptq.calibrate import (
    accumulate_gram,
    aga_refit,
    calibrate_model_latents,
    collect_layer_inputs,
    decoder_layers,
    gptq_compensate,
)
from axolotl.integrations.ternary.ptq.ternary_fit import dequantize_fit, fit_ternary
from axolotl.integrations.ternary.swap import convert_model, get_manifest
from axolotl.utils.dict import DictDefault

SCALE_MODES = [
    ("absmean", None),
    ("learnable_row", None),
    ("group", 32),
    ("dual", None),
]

CALIBRATION_SEQUENCES = 6
CALIBRATION_LEN = 24


def _correlated(rows: int = 24, cols: int = 64, samples: int = 256, seed: int = 0):
    """A weight plus activations whose channels differ in scale by two decades."""
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=gen)
    basis = torch.randn(cols, cols, generator=gen) / cols**0.5
    channel = torch.logspace(-1.5, 1.5, cols)[torch.randperm(cols, generator=gen)]
    activations = (torch.randn(samples, cols, generator=gen) @ basis) * channel
    return weight, activations


def _output_error(
    weight: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor,
    activations: torch.Tensor,
) -> float:
    return float(((weight - dequantize_fit(codes, scale)) @ activations.T).norm())


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
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


def _tokens(rows: int = CALIBRATION_SEQUENCES, length: int = CALIBRATION_LEN):
    return torch.randint(
        0, 128, (rows, length), generator=torch.Generator().manual_seed(4)
    )


def _calibrated_cfg() -> DictDefault:
    return DictDefault(
        {
            "ternary": {
                "init": "ternary_fit_calibrated",
                # a fitting init needs a scale mode that keeps the scale it solved
                "weight_scale": "learnable",
                "init_calibration": {
                    "num_sequences": CALIBRATION_SEQUENCES,
                    "sequence_len": CALIBRATION_LEN,
                },
            }
        }
    )


@pytest.fixture(name="fixed_tokens")
def fixed_tokens_fixture(monkeypatch):
    """Serve the calibration harness a fixed token tensor instead of a dataset."""
    tokens = _tokens()
    original = calibrate_module._token_batches
    monkeypatch.setattr(
        calibrate_module,
        "_token_batches",
        lambda cfg, num_sequences, sequence_len, dataset, input_ids: original(
            cfg, num_sequences, sequence_len, dataset, tokens
        ),
    )
    return tokens


# ------------------------------------------------------------------ gram


def test_accumulate_gram_is_the_summed_second_moment():
    first = torch.randn(4, 8, 16)
    second = torch.randn(2, 8, 16)

    gram = accumulate_gram([first, second])

    flat = torch.cat([first.reshape(-1, 16), second.reshape(-1, 16)])
    assert gram.shape == (16, 16)
    assert gram.dtype == torch.float32
    assert torch.allclose(gram, flat.T @ flat, atol=1e-3)


def test_accumulate_gram_promotes_low_precision_activations():
    activations = torch.randn(3, 8, 16).to(torch.bfloat16)

    gram = accumulate_gram([activations])

    assert gram.dtype == torch.float32


def test_accumulate_gram_needs_an_activation():
    with pytest.raises(ValueError, match="at least one"):
        accumulate_gram([])


# ------------------------------------------------------------------- AGA


@pytest.mark.parametrize("scale_mode,group_size", SCALE_MODES)
def test_aga_refit_lowers_the_output_space_error(scale_mode, group_size):
    weight, activations = _correlated(seed=1)
    gram = accumulate_gram([activations])
    codes, scale = fit_ternary(weight, scale_mode, group_size)

    refit = aga_refit(weight, codes, gram, scale_mode, group_size)

    assert refit.shape == scale.shape
    assert _output_error(weight, codes, refit, activations) <= _output_error(
        weight, codes, scale, activations
    )


@pytest.mark.parametrize(
    "scale_mode,group_size", [("learnable_row", None), ("group", 32)]
)
def test_aga_refit_is_worth_more_than_a_rounding_wobble(scale_mode, group_size):
    weight, activations = _correlated(seed=2)
    gram = accumulate_gram([activations])
    codes, scale = fit_ternary(weight, scale_mode, group_size)

    refit = aga_refit(weight, codes, gram, scale_mode, group_size)

    plain = _output_error(weight, codes, scale, activations)
    assert _output_error(weight, codes, refit, activations) < 0.99 * plain


def test_aga_refit_matches_the_brute_force_per_row_optimum():
    weight, activations = _correlated(rows=4, cols=16, samples=64, seed=3)
    gram = accumulate_gram([activations])
    codes, _ = fit_ternary(weight, "learnable_row")

    refit = aga_refit(weight, codes, gram, "learnable_row", damping=0.0)

    plane = codes.to(torch.float32)
    for row in range(weight.shape[0]):
        residual = weight[row] - float(refit[row]) * plane[row]
        gradient = float(plane[row] @ gram @ residual)
        assert abs(gradient) < 1e-2 * float(plane[row] @ gram @ plane[row])


def test_aga_refit_keeps_the_data_free_scale_when_the_codes_are_empty():
    weight, activations = _correlated(seed=5)
    gram = accumulate_gram([activations])
    codes = torch.zeros_like(weight, dtype=torch.int8)

    refit = aga_refit(weight, codes, gram, "learnable_row")

    assert torch.allclose(refit, quant.absmean_scale(weight, weight.shape[1]))


def _rank_one_gram(cols: int = 64, seed: int = 1) -> torch.Tensor:
    """The degenerate case: every calibration token points the same way."""
    gen = torch.Generator().manual_seed(seed)
    direction = torch.randn(cols, 1, generator=gen) @ torch.randn(
        1, cols, generator=gen
    )
    return accumulate_gram([direction])


def test_aga_refit_keeps_the_dual_pair_matched_to_its_planes():
    """A per-component fallback would put the high plane's scale on the low plane."""
    weight = torch.randn(4, 64, generator=torch.Generator().manual_seed(1)) * 0.02
    gram = _rank_one_gram()
    codes, data_free = fit_ternary(weight, "dual")

    refit = aga_refit(weight, codes, gram, "dual")

    assert bool((refit[:, 0] <= refit[:, 1]).all())
    free_error = float((weight - dequantize_fit(codes, data_free)).norm())
    assert float((weight - dequantize_fit(codes, refit)).norm()) < 2 * free_error


@pytest.mark.parametrize("scale_mode,group_size", SCALE_MODES)
def test_the_damping_keeps_an_unconstrained_direction_from_running_away(
    scale_mode, group_size
):
    """A rank-deficient S says nothing about the scale along its null space."""
    weight = torch.randn(4, 64, generator=torch.Generator().manual_seed(1)) * 0.02
    gram = _rank_one_gram()
    codes, data_free = fit_ternary(weight, scale_mode, group_size)

    refit = aga_refit(weight, codes, gram, scale_mode, group_size)
    undamped = aga_refit(weight, codes, gram, scale_mode, group_size, damping=0.0)

    row_max = float(weight.abs().amax())
    assert float(refit.max()) <= row_max
    assert float(undamped.max()) > float(refit.max())
    assert bool((refit > 0).all())
    assert float(refit.max()) < 20 * float(data_free.max())


def test_aga_refit_rejects_a_gram_of_the_wrong_width():
    weight, _ = _correlated()
    codes, _ = fit_ternary(weight)

    with pytest.raises(ValueError, match="gram must be"):
        aga_refit(weight, codes, torch.eye(8))


# ------------------------------------------------------------------ GPTQ


@pytest.mark.parametrize("scale_mode,group_size", SCALE_MODES)
def test_gptq_compensation_returns_the_canonical_layout(scale_mode, group_size):
    weight, activations = _correlated(seed=6)
    gram = accumulate_gram([activations])
    reference = fit_ternary(weight, scale_mode, group_size)

    codes, scale = gptq_compensate(weight, gram, scale_mode, group_size, block_size=16)

    assert codes.shape == reference[0].shape
    assert codes.dtype == torch.int8
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    assert scale.shape == reference[1].shape


@pytest.mark.parametrize(
    "scale_mode,group_size", [("absmean", None), ("learnable_row", None), ("group", 32)]
)
def test_gptq_compensation_lowers_the_end_to_end_block_error(scale_mode, group_size):
    weight, activations = _correlated(seed=7)
    gram = accumulate_gram([activations])
    codes, _ = fit_ternary(weight, scale_mode, group_size)
    refit = aga_refit(weight, codes, gram, scale_mode, group_size)

    compensated = gptq_compensate(weight, gram, scale_mode, group_size, block_size=16)

    assert _output_error(weight, *compensated, activations) < _output_error(
        weight, codes, refit, activations
    )


def test_gptq_compensation_leaves_the_five_value_grid_to_the_scale_refit():
    weight, activations = _correlated(seed=8)
    gram = accumulate_gram([activations])
    codes, _ = fit_ternary(weight, "dual")

    compensated = gptq_compensate(weight, gram, "dual")

    assert torch.equal(compensated[0], codes)
    assert torch.equal(compensated[1], aga_refit(weight, codes, gram, "dual"))


def test_gptq_compensation_survives_a_rank_deficient_gram():
    weight, _ = _correlated(cols=32, seed=9)
    activations = torch.randn(4, 32, generator=torch.Generator().manual_seed(9))
    gram = accumulate_gram([activations])

    codes, scale = gptq_compensate(weight, gram, block_size=16)

    assert bool(torch.isfinite(dequantize_fit(codes, scale)).all())


def test_gptq_compensation_survives_a_dead_input_channel():
    weight, activations = _correlated(cols=32, seed=10)
    activations[:, ::4] = 0.0
    gram = accumulate_gram([activations])

    codes, scale = gptq_compensate(weight, gram, block_size=16)

    assert bool(torch.isfinite(dequantize_fit(codes, scale)).all())


# --------------------------------------------------------------- the harness


def test_decoder_layers_finds_the_block_list():
    model = _tiny_llama()
    convert_model(model, DictDefault({"ternary": {}}))

    layers = decoder_layers(model)

    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 2


def test_decoder_layers_rejects_a_model_without_blocks():
    with pytest.raises(ValueError, match="no ModuleList of blocks"):
        decoder_layers(nn.Sequential(nn.Linear(4, 4)))


def test_collect_layer_inputs_captures_replayable_hidden_states():
    model = _tiny_llama()
    convert_model(model, DictDefault({"ternary": {}}))
    tokens = _tokens()

    inputs, kwargs = collect_layer_inputs(
        model,
        DictDefault({}),
        CALIBRATION_SEQUENCES,
        CALIBRATION_LEN,
        input_ids=tokens,
    )

    assert len(inputs) == CALIBRATION_SEQUENCES
    assert inputs[0].shape == (1, CALIBRATION_LEN, model.config.hidden_size)
    assert "position_embeddings" in kwargs
    replayed = decoder_layers(model)[0](inputs[0], **kwargs)
    assert (replayed[0] if isinstance(replayed, tuple) else replayed).shape == inputs[
        0
    ].shape


def test_collect_layer_inputs_restores_the_first_block():
    model = _tiny_llama()
    convert_model(model, DictDefault({"ternary": {}}))
    first = decoder_layers(model)[0]

    collect_layer_inputs(
        model, DictDefault({}), 2, CALIBRATION_LEN, input_ids=_tokens()
    )

    assert decoder_layers(model)[0] is first


def test_collect_layer_inputs_needs_a_full_length_sequence():
    model = _tiny_llama()
    convert_model(model, DictDefault({"ternary": {}}))

    with pytest.raises(ValueError, match="no sequence of"):
        collect_layer_inputs(model, DictDefault({}), 4, 4096, input_ids=_tokens())


def test_one_gram_is_shared_by_the_linears_that_read_the_same_input():
    model = _tiny_llama()
    convert_model(model, DictDefault({"ternary": {}}))
    inputs, kwargs = collect_layer_inputs(
        model, DictDefault({}), 2, CALIBRATION_LEN, input_ids=_tokens()
    )
    layer = decoder_layers(model)[0]
    modules = {
        name.rpartition(".")[2]: module for name, module in iter_ternary_modules(layer)
    }

    grams = calibrate_module._layer_grams(list(modules.values()), layer, inputs, kwargs)

    keyed = {name: grams[id(module)] for name, module in modules.items()}
    assert keyed["q_proj"] is keyed["k_proj"] is keyed["v_proj"]
    assert keyed["gate_proj"] is keyed["up_proj"]
    assert keyed["o_proj"] is not keyed["q_proj"]
    assert keyed["down_proj"].shape == (model.config.intermediate_size,) * 2


def test_a_target_its_own_layer_never_calls_falls_back_to_the_data_free_fit(
    fixed_tokens, caplog
):
    """A conditionally-routed expert has no activations; that is a warning, not a KeyError."""
    model = _tiny_llama()
    layer = model.model.layers[0]
    layer.unrouted = nn.Linear(model.config.hidden_size, 8, bias=False)
    cfg = _calibrated_cfg()
    cfg["ternary"]["target_modules"] = [
        r".*\.self_attn\.(q|k|v|o)_proj",
        r".*\.mlp\.(gate|up|down)_proj",
        r".*\.unrouted",
    ]

    with caplog.at_level("WARNING"):
        convert_model(model, cfg)

    assert "was not called by its own decoder layer" in caplog.text
    assert model.get_submodule("model.layers.0.unrouted").is_baked()


def test_an_explicit_calibration_dataset_overrides_a_pretraining_corpus(monkeypatch):
    """`prepare_datasets` hands `pretraining_dataset` the whole job if it is left set."""
    seen = {}

    def _prepare(cfg, _tokenizer):
        seen["cfg"] = cfg
        return ([{"input_ids": [1, 2, 3]}], None, 1, [])

    monkeypatch.setattr("axolotl.loaders.load_tokenizer", lambda _cfg: object())
    monkeypatch.setattr("axolotl.utils.data.sft.prepare_datasets", _prepare)
    cfg = DictDefault(
        {
            "pretraining_dataset": [{"path": "training/corpus"}],
            "streaming": True,
        }
    )

    tokens = list(calibrate_module._dataset_tokens(cfg, "calibration/corpus"))

    assert tokens == [[1, 2, 3]]
    assert seen["cfg"]["datasets"] == [
        {"path": "calibration/corpus", "type": "completion"}
    ]
    assert seen["cfg"]["pretraining_dataset"] is None
    assert seen["cfg"]["streaming"] is False


# ----------------------------------------------------------- calibrated init


def test_the_calibrated_init_writes_ternary_latents(fixed_tokens):
    model = _tiny_llama()

    manifest = convert_model(model, _calibrated_cfg())

    assert manifest.init == "ternary_fit_calibrated"
    for _, module in iter_ternary_modules(model):
        assert module.is_baked()
        assert quant.baked_codes_and_scale(module.weight.detach()) is not None


def test_the_calibrated_init_restores_lambda_and_the_training_mode(fixed_tokens):
    model = _tiny_llama()
    model.train()

    convert_model(model, _calibrated_cfg())

    assert model.training
    assert all(module.lambda_ == 1.0 for _, module in iter_ternary_modules(model))


def test_the_calibrated_init_beats_the_data_free_fit_end_to_end(fixed_tokens):
    reference = _tiny_llama()
    with torch.no_grad():
        target = reference(input_ids=fixed_tokens).logits

    errors = {}
    for init in ("ternary_fit", "ternary_fit_calibrated"):
        model = _tiny_llama()
        convert_model(
            model,
            _calibrated_cfg()
            if init.endswith("calibrated")
            else DictDefault({"ternary": {"init": init, "weight_scale": "learnable"}}),
        )
        with torch.no_grad():
            logits = model(input_ids=fixed_tokens).logits
        assert bool(torch.isfinite(logits).all())
        errors[init] = float((logits - target).norm())

    assert errors["ternary_fit_calibrated"] < errors["ternary_fit"]


def test_the_calibrated_init_skips_modules_that_are_already_baked(fixed_tokens):
    model = _tiny_llama()
    manifest = convert_model(model, _calibrated_cfg())
    calibrated = {
        name: module.weight.detach().clone()
        for name, module in iter_ternary_modules(model)
    }

    calibrate_model_latents(model, manifest, _calibrated_cfg())

    for name, module in iter_ternary_modules(model):
        assert torch.equal(module.weight.detach(), calibrated[name])


def test_the_calibrated_init_records_the_same_manifest_shape(fixed_tokens):
    model = _tiny_llama()

    convert_model(model, _calibrated_cfg())

    manifest = get_manifest(model)
    assert manifest is not None
    assert {entry.weight_scale for entry in manifest.entries} == {"learnable"}
    assert manifest.init == "ternary_fit_calibrated"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_gram_accumulates_on_cuda():
    activations = torch.randn(2, 8, 16).cuda()

    gram = accumulate_gram([activations])

    assert gram.is_cuda
    assert gram.dtype == torch.float32
