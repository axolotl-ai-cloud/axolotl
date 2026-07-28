"""`ternary.init_jitter`: breaking the cell-centre degeneracy a fit init leaves behind.

A fit writes the solver's own reconstruction into the latent, so every weight sits
exactly at a cell centre — half a quantization step from the nearest assignment
boundary. No stable optimizer step moves a weight that far, so the codes stay frozen at
the solver's output for the whole heal: the scales and the norms train, the assignment
does not. These tests pin the pathology, the cure, and that the cure stays inside the
init seam.
"""

import pydantic
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import TernaryPlugin
from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.ptq import apply_init_jitter
from axolotl.integrations.ternary.quant import flip_count
from axolotl.utils.dict import DictDefault

LEARNED_SCALE_MODES = ("learnable", "learnable_row", "dual", "trit_planes")
PROBE = "model.layers.0.mlp.down_proj"


def _cfg(tmp_path, jitter: float, seed: int = 42, **ternary) -> DictDefault:
    return DictDefault(
        {
            "output_dir": str(tmp_path),
            "seed": seed,
            "ternary": {
                "init": "ternary_fit",
                "weight_scale": "learnable",
                "lambda_schedule": "none",
                "init_jitter": jitter,
                "export": {"formats": ["master_bf16"]},
                **ternary,
            },
        }
    )


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    )


def _converted(tmp_path, jitter: float, seed: int = 42, **ternary):
    model = _tiny_llama()
    TernaryPlugin().post_model_build(_cfg(tmp_path, jitter, seed, **ternary), model)
    return model


def _batch() -> dict[str, torch.Tensor]:
    ids = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(1))
    return {"input_ids": ids, "labels": ids}


def _train_and_count_flips(model, steps: int = 8, lr: float = 1e-4) -> tuple[int, int]:
    before = {
        name: mod.code_snapshot().clone() for name, mod in iter_ternary_modules(model)
    }
    batch = _batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        model(**batch).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    flips = sum(
        int(flip_count(before[name], mod.code_snapshot()))
        for name, mod in iter_ternary_modules(model)
    )
    total = sum(mod.code_count() for _, mod in iter_ternary_modules(model))
    return flips, total


# ------------------------------------------------------------ the whole point


def test_a_fit_init_alone_freezes_every_code(tmp_path):
    """The pathology: a solved latent cannot reach a boundary at a stable step size."""
    model = _converted(tmp_path, 0.0)

    flips, total = _train_and_count_flips(model)

    assert total > 0
    assert flips == 0


def test_jitter_unfreezes_the_assignment(tmp_path):
    model = _converted(tmp_path, 0.25)

    flips, total = _train_and_count_flips(model)

    assert flips > 0
    # a nudge, not a reshuffle: the solver's assignment still dominates
    assert flips < total // 4


# --------------------------------------------------------------- determinism


@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_the_same_seed_reproduces_the_same_latents(tmp_path, weight_scale):
    first = _converted(tmp_path, 0.25, seed=42, weight_scale=weight_scale)
    second = _converted(tmp_path, 0.25, seed=42, weight_scale=weight_scale)

    for (name, left), (_, right) in zip(
        iter_ternary_modules(first), iter_ternary_modules(second), strict=True
    ):
        assert torch.equal(left.weight, right.weight), name


def test_a_different_seed_draws_different_noise(tmp_path):
    first = _converted(tmp_path, 0.25, seed=42)
    second = _converted(tmp_path, 0.25, seed=7)

    left = dict(iter_ternary_modules(first))[PROBE].weight
    right = dict(iter_ternary_modules(second))[PROBE].weight
    assert not torch.equal(left, right)


def test_every_module_draws_its_own_noise(tmp_path):
    """Mixing the module name in stops one draw being replayed across the model."""
    model = _converted(tmp_path, 0.25)
    modules = dict(iter_ternary_modules(model))
    same_shape = [
        modules[name].weight
        for name in (
            "model.layers.0.self_attn.q_proj",
            "model.layers.1.self_attn.q_proj",
        )
    ]

    assert not torch.equal(*same_shape)


# ------------------------------------------------------------------ magnitude


@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
@pytest.mark.parametrize("jitter", [0.1, 0.25])
def test_the_noise_is_scale_relative(tmp_path, weight_scale, jitter):
    """Measured against the grid step, so it means the same thing in every mode."""
    clean = dict(
        iter_ternary_modules(_converted(tmp_path, 0.0, weight_scale=weight_scale))
    )
    noisy = dict(
        iter_ternary_modules(_converted(tmp_path, jitter, weight_scale=weight_scale))
    )

    module = clean[PROBE]
    step = module.quantization_scale().detach().to(torch.float32)
    delta = (noisy[PROBE].weight - module.weight).detach().to(torch.float32)
    ratio = float((delta / step).std())

    assert ratio == pytest.approx(jitter, rel=0.15)


def test_the_two_plane_modes_reference_the_finer_scale(tmp_path):
    """Jitter measured against the coarse plane would swamp the fine one."""
    for weight_scale in ("dual", "trit_planes"):
        module = dict(
            iter_ternary_modules(_converted(tmp_path, 0.0, weight_scale=weight_scale))
        )[PROBE]
        weight = module.weight.detach()
        pair = (
            module._trit_plane_scales(weight)
            if weight_scale == "trit_planes"
            else module._dual_scales(weight)
        )
        finer = min(float(pair[0].detach().mean()), float(pair[1].detach().mean()))
        assert float(module.quantization_scale().detach().mean()) == pytest.approx(
            finer
        )


# -------------------------------------------------------- the baked-flag seam


@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_a_jittered_latent_is_not_baked(tmp_path, weight_scale):
    """It is deliberately off-grid; a stale flag would skip the quantizer entirely."""
    jittered = _converted(tmp_path, 0.25, weight_scale=weight_scale)
    exact = _converted(tmp_path, 0.0, weight_scale=weight_scale)

    assert not any(module.baked for _, module in iter_ternary_modules(jittered))
    assert all(module.baked for _, module in iter_ternary_modules(exact))


def test_a_jittered_module_quantizes_in_the_forward(tmp_path):
    model = _converted(tmp_path, 0.25)
    module = dict(iter_ternary_modules(model))[PROBE]

    effective = module._quant_weight(1.0).detach()

    assert not torch.equal(effective, module.weight.detach())
    assert len(torch.unique(effective.abs())) <= 2


# ------------------------------------------------------------- bake stays exact


@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_jitter_does_not_leak_into_the_master(tmp_path, weight_scale):
    """Noise belongs to the latent; the saved master is still an exact grid."""
    from axolotl.integrations.ternary import quant

    model = _converted(tmp_path, 0.25, weight_scale=weight_scale)
    _train_and_count_flips(model, steps=4)

    for name, module in model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(model, name)

    for _, module in iter_ternary_modules(model):
        values = module.weight.detach()
        if weight_scale == "trit_planes":
            assert quant.baked_trit_plane_codes_and_scales(values) is not None
        elif weight_scale == "dual":
            assert quant.baked_dual_codes_and_scales(values) is not None
        else:
            group = module._scale_group_size()
            assert quant.baked_codes_and_scale(values, group) is not None


# ----------------------------------------------------------------- the seam API


def test_apply_init_jitter_is_a_no_op_at_zero(tmp_path):
    model = _tiny_llama()
    cfg = _cfg(tmp_path, 0.0)
    plugin = TernaryPlugin()
    plugin.post_model_build(cfg, model)
    before = {
        name: mod.weight.detach().clone() for name, mod in iter_ternary_modules(model)
    }

    assert apply_init_jitter(model, plugin.manifest, cfg) == 0

    for name, module in iter_ternary_modules(model):
        assert torch.equal(module.weight.detach(), before[name])


def test_apply_init_jitter_reports_what_it_touched(tmp_path):
    model = _tiny_llama()
    cfg = _cfg(tmp_path, 0.25)
    plugin = TernaryPlugin()
    plugin.post_model_build(cfg, model)

    # already applied once by the init seam; running it again perturbs again
    perturbed = apply_init_jitter(model, plugin.manifest, cfg)

    assert perturbed == len(plugin.manifest.entries)


# ------------------------------------------------------------------- the schema


def test_jitter_defaults_to_off():
    assert TernaryConfig().init_jitter == 0.0


@pytest.mark.parametrize("init", ["ternary_fit", "ternary_fit_calibrated"])
def test_jitter_is_accepted_with_a_fitting_init(init):
    config = TernaryConfig(
        init=init,
        weight_scale="learnable",
        lambda_schedule="none",
        init_jitter=0.25,
        export={"formats": ["master_bf16"]},
    )

    assert config.init_jitter == 0.25


def test_jitter_is_rejected_without_a_fitting_init():
    with pytest.raises(pydantic.ValidationError, match="leaves the full-precision"):
        TernaryConfig(init="absmean", init_jitter=0.25)


def test_zero_jitter_is_allowed_with_any_init():
    assert TernaryConfig(init="absmean", init_jitter=0.0).init_jitter == 0.0


@pytest.mark.parametrize("jitter", [-0.1, 1.5])
def test_jitter_is_bounded(jitter):
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(
            init="ternary_fit",
            weight_scale="learnable",
            lambda_schedule="none",
            init_jitter=jitter,
            export={"formats": ["master_bf16"]},
        )
