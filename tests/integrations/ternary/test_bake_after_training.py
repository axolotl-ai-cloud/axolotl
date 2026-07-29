"""The bake-after-TRAINING invariant: a saved master computes the trained function.

Baking at *init* is a fixed point for free — the fit already wrote the quantizer's own
output. The invariant that matters is the one after real optimizer steps have moved the
latents, and it is the one that broke: `is_baked()` used to trust the weight's autograd
version counter, which `torch._fused_adamw_` (`optimizer: adamw_torch_fused`) never
bumps, so a fit-initialized module still claimed to be baked after a whole run.
`_post_training` then skipped the bake, left the `.scale` parameters in the checkpoint,
and the export baker re-quantized the trained latents onto an absmean grid the model had
never used.
"""

import pytest
import torch
from safetensors import safe_open
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import TernaryPlugin, quant
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.utils.dict import DictDefault

from .test_binary_codebook import binary_swap

LEARNED_SCALE_MODES = ("learnable", "learnable_row", "dual", "trit_planes")
# the sign codebook has no two-plane grid, so its learned modes are the single-plane ones
BINARY_SCALE_MODES = ("learnable", "learnable_row")
STEPS = 8


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


def _batch() -> dict[str, torch.Tensor]:
    ids = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(1))
    return {"input_ids": ids, "labels": ids}


def _cfg(
    output_dir, init: str, weight_scale: str, codebook: str = "ternary"
) -> DictDefault:
    return DictDefault(
        {
            "output_dir": str(output_dir),
            "ternary": {
                "init": init,
                "weight_scale": weight_scale,
                "codebook": codebook,
                "lambda_schedule": "none",
                "export": {"formats": ["master_bf16"]},
            },
        }
    )


def _train(model, steps: int = STEPS, fused: bool = False) -> None:
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in _batch().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=fused)
    for _ in range(steps):
        model(**batch).loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def _bake_all(model) -> None:
    for name, module in model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(model, name)


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).norm() / want.norm())


@pytest.mark.parametrize("init", ["absmean", "ternary_fit"])
@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_the_trained_function_survives_bake_and_reload(tmp_path, init, weight_scale):
    """Train for real, bake, save, reload through the plugin: same logits."""
    model = _tiny_llama()
    cfg = _cfg(tmp_path, init, weight_scale)
    TernaryPlugin().post_model_build(cfg, model)
    _train(model)

    model.eval()
    batch = _batch()
    with torch.no_grad():
        trained = model(**batch).logits.clone()

    _bake_all(model)
    with torch.no_grad():
        after_bake = model(**batch).logits.clone()
    model.save_pretrained(tmp_path)

    reloaded = LlamaForCausalLM.from_pretrained(tmp_path)
    TernaryPlugin().post_model_build(cfg, reloaded)
    reloaded.eval()
    with torch.no_grad():
        after_reload = reloaded(**batch).logits

    # the bake is the identity on the function the module already computed at λ=1
    assert _rel(after_bake, trained) == 0.0
    assert _rel(after_reload, trained) < 1e-5


@pytest.mark.parametrize("init", ["absmean", "ternary_fit"])
@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_the_master_never_ships_scale_parameters(tmp_path, init, weight_scale):
    """A leaked `.scale` reloads as an UNEXPECTED key and a different function."""
    model = _tiny_llama()
    TernaryPlugin().post_model_build(_cfg(tmp_path, init, weight_scale), model)
    _train(model)
    _bake_all(model)
    model.save_pretrained(tmp_path)

    with safe_open(tmp_path / "model.safetensors", framework="pt") as shard:
        keys = list(shard.keys())

    assert not [key for key in keys if key.split(".")[-1] in ("scale", "scale_lo")]
    assert all(module.scale is None for _, module in iter_ternary_modules(model))


@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_an_untrained_fit_still_drops_its_scale_parameters(tmp_path, weight_scale):
    """The fit leaves the latent already on its grid, so the bake is skipped."""
    model = _tiny_llama()
    TernaryPlugin().post_model_build(_cfg(tmp_path, "ternary_fit", weight_scale), model)
    assert all(module.baked for _, module in iter_ternary_modules(model))

    _bake_all(model)
    model.save_pretrained(tmp_path)

    with safe_open(tmp_path / "model.safetensors", framework="pt") as shard:
        keys = list(shard.keys())
    assert not [key for key in keys if key.split(".")[-1] in ("scale", "scale_lo")]


@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_a_gradient_lapses_the_baked_flag(tmp_path, weight_scale):
    """The flag has to follow the values, not the autograd version counter."""
    model = _tiny_llama()
    TernaryPlugin().post_model_build(_cfg(tmp_path, "ternary_fit", weight_scale), model)
    modules = [module for _, module in iter_ternary_modules(model)]
    assert all(module.is_baked() for module in modules)

    model(**_batch()).loss.backward()

    assert not any(module.is_baked() for module in modules)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")
@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_a_fused_optimizer_still_lapses_the_baked_flag(tmp_path, weight_scale):
    """`torch._fused_adamw_` updates parameters without bumping `_version`."""
    model = _tiny_llama().to("cuda")
    cfg = _cfg(tmp_path, "ternary_fit", weight_scale)
    TernaryPlugin().post_model_build(cfg, model)
    modules = [module for _, module in iter_ternary_modules(model)]
    versions = [module.weight._version for module in modules]

    batch = {key: value.to("cuda") for key, value in _batch().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    for _ in range(4):
        model(**batch).loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # the version counter is unmoved — which is exactly why it cannot be the signal
    assert [module.weight._version for module in modules] == versions
    assert not any(module.is_baked() for module in modules)


@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_the_export_baker_refuses_a_latent_under_a_learned_scale(weight_scale):
    """Its grid lives in a trained parameter, so absmean here is silent corruption."""
    from axolotl.integrations.ternary.export import bake

    torch.manual_seed(0)
    latent = torch.randn(8, 16)

    with pytest.raises(ValueError, match="reached the export baker as a latent"):
        bake.bake_weight(latent, None, weight_scale)


def test_the_export_baker_still_bakes_statistic_modes():
    from axolotl.integrations.ternary.export import bake

    torch.manual_seed(0)
    latent = torch.randn(8, 16)

    baked = bake.bake_weight(latent, None, "absmean")

    assert len(torch.unique(baked.abs())) <= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")
@pytest.mark.parametrize("weight_scale", LEARNED_SCALE_MODES)
def test_a_fused_run_saves_the_trained_function(tmp_path, weight_scale):
    """The ablation's shape end to end: fit init + a fused optimizer + bake + reload."""
    model = _tiny_llama().to("cuda")
    cfg = _cfg(tmp_path, "ternary_fit", weight_scale)
    TernaryPlugin().post_model_build(cfg, model)
    _train(model.to("cuda"), fused=True)

    model.eval()
    batch = {key: value.to("cuda") for key, value in _batch().items()}
    with torch.no_grad():
        trained = model(**batch).logits.clone().cpu()

    _bake_all(model)
    model.save_pretrained(tmp_path)

    with safe_open(tmp_path / "model.safetensors", framework="pt") as shard:
        keys = list(shard.keys())
    assert not [key for key in keys if key.split(".")[-1] in ("scale", "scale_lo")]

    reloaded = LlamaForCausalLM.from_pretrained(tmp_path)
    TernaryPlugin().post_model_build(cfg, reloaded)
    reloaded.eval()
    with torch.no_grad():
        after_reload = reloaded(**{k: v.cpu() for k, v in batch.items()}).logits

    assert _rel(after_reload, trained) < 1e-5


# ------------------------------------------------------- the same rows, codebook: binary


@pytest.mark.parametrize("init", ["absmean", "ternary_fit"])
@pytest.mark.parametrize("weight_scale", BINARY_SCALE_MODES)
def test_a_binary_trained_function_survives_bake_and_reload(
    tmp_path, init, weight_scale
):
    """The sign grid has no zero state, so a shrunken absmean would be visible at once."""
    model = _tiny_llama()
    cfg = _cfg(tmp_path, init, weight_scale, codebook="binary")
    with binary_swap():
        TernaryPlugin().post_model_build(cfg, model)
    _train(model)

    model.eval()
    batch = _batch()
    with torch.no_grad():
        trained = model(**batch).logits.clone()

    _bake_all(model)
    with torch.no_grad():
        after_bake = model(**batch).logits.clone()
    model.save_pretrained(tmp_path)

    reloaded = LlamaForCausalLM.from_pretrained(tmp_path)
    with binary_swap():
        TernaryPlugin().post_model_build(cfg, reloaded)
    reloaded.eval()
    with torch.no_grad():
        after_reload = reloaded(**batch).logits

    assert _rel(after_bake, trained) == 0.0
    assert _rel(after_reload, trained) < 1e-5


@pytest.mark.parametrize("weight_scale", BINARY_SCALE_MODES)
def test_a_binary_master_holds_one_magnitude_and_no_scales(tmp_path, weight_scale):
    model = _tiny_llama()
    cfg = _cfg(tmp_path, "ternary_fit", weight_scale, codebook="binary")
    with binary_swap():
        TernaryPlugin().post_model_build(cfg, model)
    _train(model)
    _bake_all(model)
    model.save_pretrained(tmp_path)

    with safe_open(tmp_path / "model.safetensors", framework="pt") as shard:
        keys = list(shard.keys())
        weights = {
            key: shard.get_tensor(key) for key in keys if key.endswith(".weight")
        }

    assert not [key for key in keys if key.split(".")[-1] in ("scale", "scale_lo")]
    for name, module in iter_ternary_modules(model):
        weight = weights[f"{name}.weight"]
        assert (
            quant.baked_binary_codes_and_scale(weight, module._scale_group_size())
            is not None
        )


@pytest.mark.parametrize("weight_scale", BINARY_SCALE_MODES)
def test_a_gradient_lapses_the_baked_flag_under_binary(tmp_path, weight_scale):
    model = _tiny_llama()
    cfg = _cfg(tmp_path, "ternary_fit", weight_scale, codebook="binary")
    with binary_swap():
        TernaryPlugin().post_model_build(cfg, model)
    modules = [module for _, module in iter_ternary_modules(model)]
    assert all(module.is_baked() for module in modules)

    model(**_batch()).loss.backward()

    assert not any(module.is_baked() for module in modules)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")
@pytest.mark.parametrize("weight_scale", BINARY_SCALE_MODES)
def test_a_fused_binary_run_saves_the_trained_function(tmp_path, weight_scale):
    model = _tiny_llama().to("cuda")
    cfg = _cfg(tmp_path, "ternary_fit", weight_scale, codebook="binary")
    with binary_swap():
        TernaryPlugin().post_model_build(cfg, model)
    _train(model, fused=True)

    model.eval()
    batch = {key: value.to("cuda") for key, value in _batch().items()}
    with torch.no_grad():
        trained = model(**batch).logits.clone().cpu()

    _bake_all(model)
    model.save_pretrained(tmp_path)

    reloaded = LlamaForCausalLM.from_pretrained(tmp_path)
    with binary_swap():
        TernaryPlugin().post_model_build(cfg, reloaded)
    reloaded.eval()
    with torch.no_grad():
        after_reload = reloaded(**{k: v.cpu() for k, v in batch.items()}).logits

    assert _rel(after_reload, trained) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")
def test_a_fused_binary_run_still_lapses_the_baked_flag(tmp_path):
    """`torch._fused_adamw_` moves the latent without bumping `_version`."""
    model = _tiny_llama().to("cuda")
    cfg = _cfg(tmp_path, "ternary_fit", "learnable", codebook="binary")
    with binary_swap():
        TernaryPlugin().post_model_build(cfg, model)
    modules = [module for _, module in iter_ternary_modules(model)]
    versions = [module.weight._version for module in modules]

    _train(model, steps=4, fused=True)

    assert [module.weight._version for module in modules] == versions
    assert not any(module.is_baked() for module in modules)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")
def test_a_fused_run_keeps_quantizing_after_a_reload(tmp_path):
    """A refinetuned master must not silently train in full precision."""
    model = _tiny_llama().to("cuda")
    cfg = _cfg(tmp_path, "ternary_fit", "learnable")
    TernaryPlugin().post_model_build(cfg, model)
    _train(model, fused=True)
    _bake_all(model)
    model.save_pretrained(tmp_path)

    resumed = LlamaForCausalLM.from_pretrained(tmp_path).to("cuda")
    TernaryPlugin().post_model_build(cfg, resumed)
    modules = [module for _, module in iter_ternary_modules(resumed)]
    assert all(module.is_baked() for module in modules)

    _train(resumed, steps=2, fused=True)

    assert not any(module.is_baked() for module in modules)
    for module in modules:
        latent = module.weight.detach()
        effective = module._quant_weight(1.0)
        assert not torch.equal(effective, latent)
