"""Fitting a ternary master straight from an fp8 source checkpoint.

`mistralai/Mistral-Medium-3.5-128B` is the driving case: per-tensor `float8_e4m3`
weights with a companion scale, static activation scales beside them, and embeddings,
norms, the head, the vision tower and the projector all still bf16 in the same shards.
The scales must be consumed on read and must never reach the master — a `weight_scale`
sitting next to a ternary tensor tells the next loader to dequantize it again.
"""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.ptq import fp8
from axolotl.integrations.ternary.ptq.stream import classify_tensor, stream_fit
from axolotl.integrations.ternary.swap import SwapManifest, resolve_preset
from axolotl.utils.dict import DictDefault

E4M3 = torch.float8_e4m3fn

# the three spellings of the same pair that are in the wild, two of them shipped by
# the Mistral repo itself in its two layouts of the same weights
SCALE_CONVENTIONS = [
    ("weight_scale_inv", "activation_scale"),
    ("weight_scale", "input_scale"),
    ("qscale_weight", "qscale_act"),
]


def _quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor fp8, the way a `weight_block_size: null` checkpoint stores it."""
    scale = weight.abs().amax().to(torch.float32) / 448.0
    return (weight.to(torch.float32) / scale).to(E4M3), scale.to(torch.bfloat16)


def _fp8_checkpoint(
    directory,
    weight_suffix="weight_scale_inv",
    act_suffix="activation_scale",
    layers=2,
    hidden=32,
    seed=0,
):
    """A mixed-dtype shard: fp8 projections, bf16 embeddings, norms and head."""
    torch.manual_seed(seed)
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": torch.randn(
            64, hidden, dtype=torch.bfloat16
        ),
        "model.language_model.norm.weight": torch.randn(hidden, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(64, hidden, dtype=torch.bfloat16),
        "model.vision_tower.transformer.layers.0.attention.q_proj.weight": torch.randn(
            hidden, hidden, dtype=torch.bfloat16
        ),
        "model.multi_modal_projector.linear_1.weight": torch.randn(
            hidden, hidden, dtype=torch.bfloat16
        ),
    }
    reference: dict[str, torch.Tensor] = {}
    for layer in range(layers):
        stem = f"model.language_model.layers.{layer}"
        tensors[f"{stem}.input_layernorm.weight"] = torch.randn(
            hidden, dtype=torch.bfloat16
        )
        for name in ("self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj"):
            weight = torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.05
            payload, scale = _quantize(weight)
            key = f"{stem}.{name}"
            tensors[f"{key}.weight"] = payload
            tensors[f"{key}.{weight_suffix}"] = scale
            tensors[f"{key}.{act_suffix}"] = torch.tensor(0.5, dtype=torch.bfloat16)
            reference[f"{key}.weight"] = fp8.dequantize(payload, scale)
    save_file(tensors, directory / "model.safetensors")
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Mistral3ForConditionalGeneration"],
                "model_type": "mistral3",
                "torch_dtype": "bfloat16",
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": None,
                    "activation_scheme": "static",
                    "modules_to_not_convert": [
                        "model.vision_tower",
                        "model.multi_modal_projector",
                        "lm_head",
                    ],
                },
            }
        )
    )
    return directory, reference


def _cfg(**ternary):
    preset = resolve_preset("mistral3")
    return DictDefault(
        {
            "ternary": {
                "init": "ternary_fit",
                "weight_scale": "learnable_row",
                "lambda_schedule": "none",
                "target_modules": list(preset.target_modules),
                "keep_fp_modules": list(preset.keep_fp_modules),
                "export": {"formats": ["master_bf16"]},
                **ternary,
            }
        }
    )


# ------------------------------------------------------------------- dequant


def test_dequantize_multiplies_through_fp32_and_rounds_once():
    """The reference loader computes `(q * s)` in fp32 and casts once; so must we."""
    weight = torch.randn(16, 8, dtype=torch.bfloat16) * 0.05
    payload, scale = _quantize(weight)

    got = fp8.dequantize(payload, scale)

    expected = (payload.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
    assert torch.equal(got, expected)
    assert got.dtype == torch.bfloat16


def test_dequantize_recovers_the_original_magnitudes():
    """A direction check: dividing instead of multiplying is off by ~scale squared."""
    weight = torch.randn(32, 32, dtype=torch.bfloat16) * 0.05
    payload, scale = _quantize(weight)

    got = fp8.dequantize(payload, scale)

    assert float(got.abs().max()) == pytest.approx(float(weight.abs().max()), rel=0.02)
    assert float((got.float() - weight.float()).abs().max()) < 0.01


def test_dequantize_accepts_a_per_output_channel_scale():
    payload = torch.randn(8, 4).to(E4M3)
    scale = torch.rand(8, dtype=torch.bfloat16) + 0.5

    got = fp8.dequantize(payload, scale)

    expected = payload.to(torch.float32) * scale.to(torch.float32).reshape(-1, 1)
    assert torch.equal(got, expected.to(torch.bfloat16))


def test_dequantize_rejects_a_block_scale_grid_it_cannot_expand():
    """Block-wise fp8 needs per-block expansion; broadcasting would rescale wrongly."""
    payload = torch.randn(128, 128).to(E4M3)
    grid = torch.rand(3, 5)

    with pytest.raises(ValueError, match="weight_block_size"):
        fp8.dequantize(payload, grid)


def test_an_fp8_tensor_with_no_scale_is_an_error_not_a_silent_fit():
    with pytest.raises(ValueError, match="companion scale"):
        fp8.assert_no_orphan_fp8("model.layers.0.mlp.up_proj.weight", E4M3)

    fp8.assert_no_orphan_fp8("model.norm.weight", torch.bfloat16)


# ------------------------------------------------------------ scale discovery


@pytest.mark.parametrize("weight_suffix,act_suffix", SCALE_CONVENTIONS)
def test_every_scale_convention_is_recognized(weight_suffix, act_suffix):
    keys = [
        "model.layers.0.mlp.up_proj.weight",
        f"model.layers.0.mlp.up_proj.{weight_suffix}",
        f"model.layers.0.mlp.up_proj.{act_suffix}",
        "model.norm.weight",
    ]

    assert fp8.scale_keys(keys) == {
        "model.layers.0.mlp.up_proj.weight": f"model.layers.0.mlp.up_proj.{weight_suffix}"
    }
    assert fp8.companion_keys(keys) == {
        f"model.layers.0.mlp.up_proj.{weight_suffix}",
        f"model.layers.0.mlp.up_proj.{act_suffix}",
    }


def test_a_scale_shaped_name_without_a_weight_is_not_treated_as_a_companion():
    """Only a scale that actually belongs to a weight may be dropped."""
    keys = ["model.some_module.weight_scale", "model.other.weight"]

    assert fp8.scale_keys(keys) == {}
    assert fp8.companion_keys(keys) == set()


def test_dequantized_items_drops_scales_and_passes_bf16_through():
    payload, scale = _quantize(torch.randn(8, 8, dtype=torch.bfloat16))
    norm = torch.randn(8, dtype=torch.bfloat16)
    tensors = {
        "m.up_proj.weight": payload,
        "m.up_proj.weight_scale_inv": scale,
        "m.up_proj.activation_scale": torch.tensor(0.5, dtype=torch.bfloat16),
        "m.norm.weight": norm,
    }

    got = dict(fp8.dequantized_items(tensors))

    assert set(got) == {"m.up_proj.weight", "m.norm.weight"}
    assert torch.equal(got["m.norm.weight"], norm)
    assert got["m.up_proj.weight"].dtype == torch.bfloat16


# ------------------------------------------------------- end to end, both codebooks


@pytest.mark.parametrize("codebook", ["ternary", "binary"])
def test_an_fp8_checkpoint_fits_end_to_end(tmp_path, codebook):
    source, reference = _fp8_checkpoint(tmp_path / "src")
    output = tmp_path / f"out-{codebook}"

    report = stream_fit(source, output, _cfg(codebook=codebook))

    assert report.tensors_fitted == 6  # 2 layers x 3 projections
    master = load_file(output / "model.safetensors")
    for key, dequantized in reference.items():
        assert key in master
        assert master[key].dtype == torch.bfloat16
        # the fit reconstructs the dequantized weight, so it must track it, not the
        # raw fp8 payload it would have read without the scale
        error = (master[key].float() - dequantized.float()).abs().mean()
        assert float(error) < float(dequantized.float().abs().mean())


@pytest.mark.parametrize("codebook", ["ternary", "binary"])
def test_no_scale_tensor_survives_into_the_master(tmp_path, codebook):
    source, _ = _fp8_checkpoint(tmp_path / "src")
    output = tmp_path / f"out-{codebook}"

    stream_fit(source, output, _cfg(codebook=codebook))

    master = load_file(output / "model.safetensors")
    leaked = [
        key
        for key in master
        if key.rsplit(".", 1)[-1]
        in set(fp8.WEIGHT_SCALE_SUFFIXES) | set(fp8.ACT_SCALE_SUFFIXES)
    ]
    assert leaked == []


@pytest.mark.parametrize("weight_suffix,act_suffix", SCALE_CONVENTIONS)
def test_each_convention_fits_end_to_end(tmp_path, weight_suffix, act_suffix):
    source, reference = _fp8_checkpoint(
        tmp_path / f"src-{weight_suffix}",
        weight_suffix=weight_suffix,
        act_suffix=act_suffix,
    )
    output = tmp_path / f"out-{weight_suffix}"

    report = stream_fit(source, output, _cfg())

    assert report.tensors_fitted == len(reference)
    assert not any(
        weight_suffix in key for key in load_file(output / "model.safetensors")
    )


def test_the_bf16_tensors_are_copied_through_byte_identically(tmp_path):
    source, _ = _fp8_checkpoint(tmp_path / "src")
    original = load_file(source / "model.safetensors")
    output = tmp_path / "out"

    stream_fit(source, output, _cfg())

    master = load_file(output / "model.safetensors")
    for key in (
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.vision_tower.transformer.layers.0.attention.q_proj.weight",
        "model.multi_modal_projector.linear_1.weight",
    ):
        assert torch.equal(master[key], original[key]), key


def test_the_manifest_covers_only_the_language_model_projections(tmp_path):
    source, _ = _fp8_checkpoint(tmp_path / "src")
    output = tmp_path / "out"

    stream_fit(source, output, _cfg())

    manifest = SwapManifest.load(output)
    names = {entry.name for entry in manifest.entries}
    assert all(".language_model.layers." in name for name in names)
    assert "lm_head" in manifest.kept_fp
    assert any("vision_tower" in name for name in manifest.kept_fp)


# ------------------------------------------------------------------ the config


def test_the_master_config_carries_no_fp8_quantization_config(tmp_path):
    """A stale fp8 config would make transformers dequantize the ternary latents."""
    source, _ = _fp8_checkpoint(tmp_path / "src")
    output = tmp_path / "out"

    stream_fit(source, output, _cfg())

    config = json.loads((output / "config.json").read_text())
    assert "quantization_config" not in config
    assert config["model_type"] == "mistral3"
    assert json.loads((source / "config.json").read_text())["quantization_config"]


def test_stripping_is_a_no_op_on_an_unquantized_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "llama"}))

    assert fp8.strip_quantization_config(path) is False
    assert json.loads(path.read_text()) == {"model_type": "llama"}
    assert fp8.strip_quantization_config(tmp_path / "missing.json") is False


# ----------------------------------------------------------------- the digests


def test_the_input_digest_is_of_the_fp8_source_bytes(tmp_path):
    """Resume must key off the source as it sits on disk, not the dequantized view."""
    import hashlib

    source, _ = _fp8_checkpoint(tmp_path / "src")
    output = tmp_path / "out"

    report = stream_fit(source, output, _cfg())

    raw = hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
    assert report.records["model.safetensors"].input_sha256 == raw


def test_a_repeat_run_skips_the_shard_it_already_fitted(tmp_path):
    source, _ = _fp8_checkpoint(tmp_path / "src")
    output = tmp_path / "out"
    stream_fit(source, output, _cfg())

    again = stream_fit(source, output, _cfg())

    assert again.shards_skipped == 1
    assert again.shards_fitted == 0


def test_a_changed_fp8_source_invalidates_the_resume(tmp_path):
    source, _ = _fp8_checkpoint(tmp_path / "src")
    output = tmp_path / "out"
    stream_fit(source, output, _cfg())
    _fp8_checkpoint(tmp_path / "src", seed=7)

    again = stream_fit(source, output, _cfg())

    assert again.shards_fitted == 1
    assert again.shards_skipped == 0


# ------------------------------------------------------------ the arch preset


def _index_roles(keys):
    preset = resolve_preset("mistral3")
    cfg = TernaryConfig(
        target_modules=list(preset.target_modules),
        keep_fp_modules=list(preset.keep_fp_modules),
    )
    dropped = fp8.companion_keys(keys)
    roles: dict[str, str] = {}
    for key in keys:
        roles[key] = "drop" if key in dropped else classify_tensor(key, cfg)
    return roles


def _fabricated_index(layers=88, vision_layers=48):
    """The real repo's tensor names, at the real multiplicities."""
    keys = [
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.multi_modal_projector.linear_1.weight",
        "model.multi_modal_projector.linear_2.weight",
        "model.multi_modal_projector.norm.weight",
        "model.multi_modal_projector.patch_merger.merging_layer.weight",
        "model.vision_tower.ln_pre.weight",
        "model.vision_tower.patch_conv.weight",
    ]
    for layer in range(layers):
        stem = f"model.language_model.layers.{layer}"
        keys += [
            f"{stem}.input_layernorm.weight",
            f"{stem}.post_attention_layernorm.weight",
        ]
        for proj in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            keys += [
                f"{stem}.{proj}.weight",
                f"{stem}.{proj}.weight_scale_inv",
                f"{stem}.{proj}.activation_scale",
            ]
    for layer in range(vision_layers):
        stem = f"model.vision_tower.transformer.layers.{layer}"
        keys += [f"{stem}.attention_norm.weight", f"{stem}.ffn_norm.weight"]
        for proj in (
            "attention.q_proj",
            "attention.k_proj",
            "attention.v_proj",
            "attention.o_proj",
            "feed_forward.gate_proj",
            "feed_forward.up_proj",
            "feed_forward.down_proj",
        ):
            keys.append(f"{stem}.{proj}.weight")
    return keys


def test_the_mistral3_preset_enumerates_the_whole_model():
    keys = _fabricated_index()

    roles = _index_roles(keys)

    counts = {
        role: sum(1 for r in roles.values() if r == role)
        for role in set(roles.values())
    }
    assert counts["fit"] == 88 * 7
    assert counts["drop"] == 88 * 7 * 2
    # every language-model projection, and nothing else, is fitted
    assert all(
        ".language_model.layers." in key and key.endswith(".weight")
        for key, role in roles.items()
        if role == "fit"
    )


def test_the_preset_keeps_the_tower_the_projector_and_the_head():
    roles = _index_roles(_fabricated_index())

    for key in (
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.vision_tower.transformer.layers.0.attention.q_proj.weight",
        "model.multi_modal_projector.linear_1.weight",
        "model.multi_modal_projector.patch_merger.merging_layer.weight",
    ):
        assert roles[key] == "keep_fp", key


def test_no_linear_falls_through_unclassified():
    """A projection landing in `copy` would be silently kept at full precision."""
    roles = _index_roles(_fabricated_index())

    fell_through = [
        key
        for key, role in roles.items()
        if role == "copy" and ("proj" in key or "merging_layer" in key)
    ]
    assert fell_through == []


def _real_index_keys():
    from pathlib import Path

    root = Path(
        "/mnt/data/hf_cache/hub/models--mistralai--Mistral-Medium-3.5-128B/snapshots"
    )
    if not root.is_dir():
        return None
    for path in root.glob("*/model.safetensors.index.json"):
        return list(json.loads(path.read_text())["weight_map"])
    return None


@pytest.mark.skipif(
    _real_index_keys() is None, reason="the real checkpoint index is not on disk"
)
def test_the_preset_against_the_real_checkpoint_index():
    """The synthetic index is what I believe the model is; this is what it is."""
    keys = _real_index_keys()

    roles = _index_roles(keys)

    counts = {
        role: sum(1 for r in roles.values() if r == role)
        for role in set(roles.values())
    }
    assert counts["fit"] == 616, counts
    assert counts["drop"] == 1232, counts
    assert sum(counts.values()) == len(keys) == 2465
    assert [
        key for key, role in roles.items() if role == "copy" and "norm" not in key
    ] == []
