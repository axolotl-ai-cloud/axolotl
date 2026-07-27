"""CPU-only tests for `run_export`, the seam between the plugin/CLI and the packers."""

import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import TernaryPlugin
from axolotl.integrations.ternary.export import (
    _gate_artifact,
    bake,
    parity,
    run_export,
)
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.swap import MANIFEST_FILENAME, convert_model
from axolotl.utils.dict import DictDefault


def _write_master(directory, bake_weights: bool = True):
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
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    manifest = convert_model(
        model, DictDefault({"output_dir": str(directory), "ternary": {}})
    )
    if bake_weights:
        for name, module in iter_ternary_modules(model):
            module._post_training(model, name)
    model.save_pretrained(directory)
    manifest.save(directory)
    return manifest


def _cfg(directory, **export) -> DictDefault:
    return DictDefault({"output_dir": str(directory), "ternary": {"export": export}})


@pytest.fixture(name="master", scope="module")
def fixture_master(tmp_path_factory):
    directory = tmp_path_factory.mktemp("master")
    return directory, _write_master(directory)


# ----------------------------------------------------------------- happy paths


def test_defaults_write_master_and_hf_bitnet(master):
    master_dir, _ = master

    artifacts = run_export(_cfg(master_dir))

    assert list(artifacts) == ["master_bf16", "hf_bitnet"]
    assert artifacts["master_bf16"] == master_dir
    assert artifacts["hf_bitnet"] == master_dir / "hf_bitnet"
    assert (artifacts["hf_bitnet"] / "model.safetensors").is_file()
    assert (artifacts["hf_bitnet"] / "config.json").is_file()


def test_hf_bitnet_artifact_is_packed_and_stamped(master):
    master_dir, manifest = master

    artifacts = run_export(_cfg(master_dir, formats=["hf_bitnet"]))

    tensors = bake.load_tensors(artifacts["hf_bitnet"])
    entry = manifest.entries[0]
    packed = tensors[f"{entry.name}.weight"]
    assert packed.dtype == torch.uint8
    assert packed.shape == (entry.out_features // 4, entry.in_features)
    assert f"{entry.name}.weight_scale" in tensors

    config = json.loads((artifacts["hf_bitnet"] / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "bitnet"


def test_master_is_baked_and_stamped_even_when_only_packing(master):
    master_dir, _ = master

    run_export(_cfg(master_dir, formats=["hf_bitnet"]))

    config = json.loads((master_dir / "config.json").read_text())
    stamp = config[bake.QUANTIZER_METADATA_KEY]
    assert stamp["format"] == "master_bf16"
    assert stamp["scheme"] == "ternary"


def test_explicit_formats_override_the_config(master):
    master_dir, _ = master

    artifacts = run_export(
        _cfg(master_dir, formats=["master_bf16", "hf_bitnet"]),
        formats=["master_bf16"],
    )

    assert list(artifacts) == ["master_bf16"]


def test_no_formats_is_a_noop(master):
    master_dir, _ = master

    assert run_export(_cfg(master_dir), formats=[]) == {}
    assert run_export(_cfg(master_dir, formats=[])) == {}


def test_output_dir_separates_artifacts_from_the_master(master, tmp_path):
    master_dir, _ = master

    artifacts = run_export(_cfg(master_dir), output_dir=tmp_path)

    assert artifacts["master_bf16"] == tmp_path / "master_bf16"
    assert artifacts["hf_bitnet"] == tmp_path / "hf_bitnet"
    assert (artifacts["master_bf16"] / MANIFEST_FILENAME).is_file()
    # the copied master is the baked master, byte for byte
    copied = bake.load_tensors(artifacts["master_bf16"])
    original = bake.load_tensors(master_dir)
    assert copied.keys() == original.keys()
    assert all(torch.equal(copied[key], original[key]) for key in original)


def test_master_dir_overrides_the_config_output_dir(master, tmp_path):
    master_dir, _ = master

    artifacts = run_export(
        _cfg(tmp_path / "unused"), formats=["hf_bitnet"], master_dir=master_dir
    )

    assert artifacts["hf_bitnet"] == master_dir / "hf_bitnet"


def test_latent_master_is_baked_before_packing(tmp_path):
    master_dir = tmp_path / "latent"
    _write_master(master_dir, bake_weights=False)

    artifacts = run_export(_cfg(master_dir))

    manifest = bake.SwapManifest.load(master_dir)
    weight = bake.load_tensors(master_dir)[f"{manifest.entries[0].name}.weight"]
    codes, scale = bake.derive_codes_and_scale(weight)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    assert scale > 0
    assert (artifacts["hf_bitnet"] / "model.safetensors").is_file()


# ---------------------------------------------------------------- parity gates


def test_parity_gate_runs_for_hf_bitnet(master, monkeypatch):
    master_dir, _ = master
    reports = []
    original = parity.run_parity_gate

    def _spy(*args, **kwargs):
        report = original(*args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(parity, "run_parity_gate", _spy)

    run_export(_cfg(master_dir, formats=["master_bf16", "hf_bitnet"]))

    assert [report.format for report in reports] == ["master_bf16", "hf_bitnet"]
    assert all(report.passed for report in reports)
    assert all(report.code_mismatches == 0 for report in reports)
    assert reports[1].tensors_checked == 14


def test_parity_failure_raises_and_names_the_format(master, monkeypatch):
    master_dir, _ = master

    def _failing(_master, _artifact, fmt, _manifest, **_kwargs):
        return parity.ParityReport(format=fmt, failures=["synthetic mismatch"])

    monkeypatch.setattr(parity, "run_parity_gate", _failing)

    with pytest.raises(RuntimeError, match="hf_bitnet export failed its parity gate"):
        run_export(_cfg(master_dir, formats=["hf_bitnet"]))


def test_corrupted_artifact_fails_the_gate(master, tmp_path):
    master_dir, manifest = master
    artifacts = run_export(
        _cfg(master_dir, formats=["hf_bitnet"], run_parity_gate=False),
        output_dir=tmp_path,
    )
    artifact = artifacts["hf_bitnet"]
    tensors = bake.load_tensors(artifact)
    key = f"{manifest.entries[0].name}.weight"
    tensors[key][0, 0] ^= 0b11
    bake.save_shard(tensors, artifact / "model.safetensors")

    with pytest.raises(RuntimeError, match="codes differ from the master"):
        _gate_artifact("hf_bitnet", master_dir, artifact, manifest)


def test_gate_can_be_disabled(master, monkeypatch):
    master_dir, _ = master

    def _fail(*args, **kwargs):
        raise AssertionError("the parity gate must not run when it is disabled")

    monkeypatch.setattr(parity, "run_parity_gate", _fail)

    run_export(_cfg(master_dir, formats=["hf_bitnet"], run_parity_gate=False))


def test_block_gated_formats_are_verified_while_they_are_packed(master):
    """The GGUF/I2_S containers are gated byte-for-byte at write time, not re-read."""
    master_dir, manifest = master

    for fmt in ("gguf_tq2_0", "gguf_tq1_0", "i2_s"):
        assert fmt not in parity.EXTRACTORS
        assert fmt in parity.BLOCK_GATED_FORMATS
        assert fmt in parity.BLOCK_DECODERS
        # already verified during `_write_artifact`, so the post-hoc gate is a no-op
        _gate_artifact(fmt, master_dir, master_dir, manifest)


def test_a_format_with_no_gate_at_all_is_refused(master):
    master_dir, manifest = master

    with pytest.raises(RuntimeError, match="no parity unpacker and no block gate"):
        _gate_artifact("not_a_format", master_dir, master_dir, manifest)


# -------------------------------------------------------------------- failures


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
        run_export(_cfg(tmp_path))


def test_unknown_format_raises(master):
    master_dir, _ = master

    with pytest.raises(ValueError, match="unknown ternary export format"):
        run_export(_cfg(master_dir), formats=["not_a_format"])


@pytest.mark.parametrize("fmt", ["gguf_tq2_0", "gguf_tq1_0", "i2_s"])
def test_gguf_formats_require_the_optional_package(master, fmt):
    try:
        import gguf  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("gguf is installed; the ImportError path cannot be exercised")
    master_dir, _ = master

    with pytest.raises(ImportError, match="pip install gguf"):
        run_export(_cfg(master_dir), formats=[fmt])


# --------------------------------------------------------------- plugin wiring


def test_post_train_unload_exports_for_real(tmp_path):
    master_dir = tmp_path / "run"
    _write_master(master_dir)
    cfg = _cfg(master_dir, formats=["hf_bitnet"])

    TernaryPlugin().post_train_unload(cfg)

    assert (master_dir / "hf_bitnet" / "model.safetensors").is_file()


def test_cli_export_reaches_the_real_pipeline(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from axolotl.integrations.ternary.cli import PLUGIN_PATH, ternary

    master_dir = tmp_path / "cli"
    _write_master(master_dir)
    cfg = _cfg(master_dir)
    cfg["plugins"] = [PLUGIN_PATH]
    monkeypatch.setattr("axolotl.cli.config.load_cfg", lambda _config, **_kw: cfg)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(
        ternary, ["export", str(config_file), "--format", "hf_bitnet"]
    )

    assert result.exit_code == 0, result.output
    assert f"hf_bitnet: {master_dir / 'hf_bitnet'}" in result.output
    assert (master_dir / "hf_bitnet" / "model.safetensors").is_file()
