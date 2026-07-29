"""CPU-only tests for `axolotl ternary fit-stream` — the shard-at-a-time PTQ pass.

The load-bearing claim is that streaming changes nothing but the memory ceiling: a
checkpoint fitted shard by shard must come out tensor-for-tensor identical to the same
checkpoint fitted by `convert_model` with the whole model resident. Everything else
here — classification, fused stacks, resumption — protects that claim.
"""

import hashlib
import json
import os
import shutil

import pytest
import torch
import yaml
from click.testing import CliRunner
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from axolotl.cli.main import cli
from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.args import (
    UNIMPLEMENTED_CODEBOOKS,
    resolve_ternary_config,
)
from axolotl.integrations.ternary.export import bake, bitplanes, run_export
from axolotl.integrations.ternary.modules import iter_ternary_modules
from axolotl.integrations.ternary.ptq import stream
from axolotl.integrations.ternary.ptq.stream import (
    ShardRecord,
    classify_tensor,
    fit_tensor,
    load_records,
    stream_fit,
    write_records,
    write_shard_atomic,
)
from axolotl.integrations.ternary.swap import SwapManifest, convert_model
from axolotl.utils.dict import DictDefault

from tests.conftest import capture_axolotl_warnings

MASTER_ONLY = {"formats": ["master_bf16"]}

FITTING_SCALE_MODES = ("learnable", "learnable_row", "dual", "trit_planes")


def _binary_has_landed() -> bool:
    """Whether the binary quantizer this module dispatches to is implemented yet."""
    if "binary" in UNIMPLEMENTED_CODEBOOKS:
        return False
    try:
        quant.binary_scale(torch.ones(2, 2))
    except NotImplementedError:
        return False
    return True


BINARY_LANDED = _binary_has_landed()
requires_binary = pytest.mark.skipif(
    not BINARY_LANDED, reason="the binary quantizer has not landed yet"
)


def _cfg(weight_scale: str = "learnable", **ternary) -> DictDefault:
    return DictDefault(
        {
            "ternary": {
                "init": "ternary_fit",
                "weight_scale": weight_scale,
                "lambda_schedule": "none",
                "export": MASTER_ONLY,
                **ternary,
            }
        }
    )


def _llama_config(tie_word_embeddings: bool = True) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=tie_word_embeddings,
    )


def _write_checkpoint(directory, seed: int):
    """Save a tied-embedding Llama whose weights are this seed's, and nobody else's."""
    torch.manual_seed(seed)
    model = LlamaForCausalLM(_llama_config()).to(torch.bfloat16)
    model.save_pretrained(directory, max_shard_size="20KB")
    return directory


@pytest.fixture(name="checkpoint", scope="module")
def fixture_checkpoint(tmp_path_factory):
    """A tied-embedding Llama saved across several shards, as a real conversion sees it."""
    return _write_checkpoint(tmp_path_factory.mktemp("fp-checkpoint"), seed=0)


@pytest.fixture(name="fused")
def fixture_fused(tmp_path):
    """A hand-built checkpoint: a fused expert stack, a router, a tower, a norm."""
    directory = tmp_path / "fused"
    directory.mkdir()
    torch.manual_seed(0)
    first = {
        "model.embed_tokens.weight": torch.randn(32, 16, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(
            16, 16, dtype=torch.bfloat16
        ),
        "model.layers.0.mlp.gate.weight": torch.randn(4, 16, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(16, dtype=torch.bfloat16),
    }
    second = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(
            4, 16, 32, dtype=torch.bfloat16
        ),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(
            16, 16, dtype=torch.bfloat16
        ),
    }
    names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    save_file(first, directory / names[0], {"format": "pt"})
    save_file(second, directory / names[1], {"format": "pt"})
    weight_map = {key: names[0] for key in first} | {key: names[1] for key in second}
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    (directory / "config.json").write_text(json.dumps({"model_type": "llama"}))
    return directory, {**first, **second}


def _fused_cfg(weight_scale: str = "learnable", **ternary) -> DictDefault:
    return _cfg(
        weight_scale,
        target_modules=[
            r".*\.self_attn\.(q|k|v|o)_proj",
            r".*\.experts\.(gate_up|down)_proj",
        ],
        keep_fp_modules=[r"(.*\.)?visual\..*"],
        **ternary,
    )


def _in_memory_fit(checkpoint, cfg) -> dict[str, torch.Tensor]:
    """Fit the same checkpoint the way `axolotl train` does, with the model resident."""
    model = LlamaForCausalLM(_llama_config()).to(torch.bfloat16)
    model.load_state_dict(bake.load_tensors(checkpoint), strict=False)
    convert_model(model, cfg)
    return model.state_dict()


def _digests(directory) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in bake.shard_paths(directory)
        if path.is_file()
    }


# --------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "name,role",
    [
        ("model.layers.0.self_attn.q_proj.weight", "fit"),
        ("model.layers.1.mlp.down_proj.weight", "fit"),
        ("model.embed_tokens.weight", "keep_fp"),
        ("lm_head.weight", "keep_fp"),
        ("model.layers.0.mlp.gate.weight", "keep_fp"),
        ("model.visual.blocks.0.attn.qkv.weight", "keep_fp"),
        ("model.layers.0.input_layernorm.weight", "copy"),
        ("model.layers.0.self_attn.q_proj.bias", "copy"),
        ("model.norm.weight", "copy"),
    ],
)
def test_a_tensor_is_classified_by_its_module_name(name, role):
    ternary_cfg = resolve_ternary_config(
        _cfg(
            target_modules=[
                r".*\.self_attn\.(q|k|v|o)_proj",
                r".*\.mlp\.(gate|up|down)_proj",
            ]
        )
    )

    assert classify_tensor(name, ternary_cfg) == role


def test_the_keep_list_beats_the_target_list():
    ternary_cfg = resolve_ternary_config(
        _cfg(
            target_modules=[r".*\.mlp\..*"],
            keep_fp_modules=[r".*\.mlp\.gate"],
        )
    )

    assert classify_tensor("model.layers.0.mlp.gate.weight", ternary_cfg) == "keep_fp"
    assert classify_tensor("model.layers.0.mlp.up_proj.weight", ternary_cfg) == "fit"


def test_an_explicitly_targeted_aux_module_is_still_fitted():
    """The registry is a default, not a veto — but the swap's warning still fires."""
    ternary_cfg = resolve_ternary_config(_cfg(target_modules=[r".*\.mlp\.gate"]))

    assert classify_tensor("model.layers.0.mlp.gate.weight", ternary_cfg) == "fit"


def test_a_fused_parameter_without_a_weight_suffix_is_matched_whole(fused):
    _, _tensors = fused
    ternary_cfg = resolve_ternary_config(_fused_cfg())

    role = classify_tensor("model.layers.0.mlp.experts.gate_up_proj", ternary_cfg)

    assert role == "fit"


def test_an_unconfigured_target_list_falls_back_to_the_checkpoint_preset(
    checkpoint, tmp_path
):
    """No model to read `model_type` off, so the preset comes from `config.json`."""
    report = stream_fit(checkpoint, tmp_path / "preset-out", _cfg())

    assert report.tensors_fitted == 14
    assert set(report.families) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


# ------------------------------------------------- equality with the resident fit


@pytest.mark.parametrize("weight_scale", FITTING_SCALE_MODES)
def test_a_streamed_fit_equals_the_resident_fit(checkpoint, tmp_path, weight_scale):
    cfg = _cfg(weight_scale)
    output = tmp_path / f"streamed-{weight_scale}"

    stream_fit(checkpoint, output, cfg)

    resident = _in_memory_fit(checkpoint, cfg)
    streamed = bake.load_tensors(output)
    assert streamed
    for key, tensor in streamed.items():
        assert torch.equal(tensor, resident[key]), key


def test_the_streamed_master_reloads_as_baked(checkpoint, tmp_path):
    """The point of writing the reconstruction rather than the codes."""
    output = tmp_path / "streamed"
    cfg = _cfg("trit_planes")
    stream_fit(checkpoint, output, cfg)

    model = LlamaForCausalLM(_llama_config()).to(torch.bfloat16)
    model.load_state_dict(bake.load_tensors(output), strict=False)
    convert_model(model, DictDefault({"base_model": str(output), **cfg}))

    assert all(module.is_baked() for _, module in iter_ternary_modules(model))


@pytest.mark.parametrize("weight_scale", FITTING_SCALE_MODES)
def test_every_fitted_tensor_re_quantizes_to_itself(checkpoint, tmp_path, weight_scale):
    output = tmp_path / f"exact-{weight_scale}"
    stream_fit(checkpoint, output, _cfg(weight_scale))

    manifest = SwapManifest.load(output)
    tensors = bake.load_tensors(output)
    for entry in manifest.entries:
        codes, scale = bake.derive_codes_and_scale(
            tensors[f"{entry.name}.weight"],
            entry.group_size,
            entry.weight_scale,
            manifest.scales_for(entry.name),
        )
        rebuilt = bake.dequantize_derived(
            codes, scale, entry.weight_scale, torch.bfloat16
        )
        assert torch.equal(rebuilt, tensors[f"{entry.name}.weight"]), entry.name


def test_tied_embeddings_are_left_alone(checkpoint, tmp_path):
    output = tmp_path / "tied"
    stream_fit(checkpoint, output, _cfg())

    source = bake.load_tensors(checkpoint)
    streamed = bake.load_tensors(output)
    assert "lm_head.weight" not in streamed
    assert torch.equal(
        streamed["model.embed_tokens.weight"], source["model.embed_tokens.weight"]
    )


# ---------------------------------------------------------- fused expert stacks


def test_a_fused_stack_is_fitted_one_slice_at_a_time(fused, tmp_path):
    directory, tensors = fused
    key = "model.layers.0.mlp.experts.gate_up_proj"
    cfg = _fused_cfg()

    stream_fit(directory, tmp_path / "out", cfg)

    ternary_cfg = resolve_ternary_config(cfg)
    streamed = bake.load_tensors(tmp_path / "out")[key]
    assert streamed.shape == tensors[key].shape
    for index, expert in enumerate(tensors[key]):
        assert torch.equal(streamed[index], fit_tensor(expert, ternary_cfg)), index


def test_each_expert_slice_carries_its_own_scale(fused, tmp_path):
    """A stack fitted whole would share one scale across every expert."""
    directory, tensors = fused
    key = "model.layers.0.mlp.experts.gate_up_proj"

    stream_fit(directory, tmp_path / "out", _fused_cfg())

    streamed = bake.load_tensors(tmp_path / "out")[key]
    scales = {float(expert.abs().max()) for expert in streamed}
    assert len(scales) == len(tensors[key])


def test_a_fused_stack_is_not_listed_as_a_swap_entry(fused, tmp_path):
    """The manifest addresses entries as `<name>.weight`; a fused parameter has none."""
    directory, _ = fused

    report = stream_fit(directory, tmp_path / "out", _fused_cfg())

    manifest = SwapManifest.load(tmp_path / "out")
    assert [entry.name for entry in manifest.entries] == [
        "model.layers.0.self_attn.q_proj"
    ]
    assert report.tensors_fitted == 2
    assert report.families == {"q_proj": 1, "gate_up_proj": 1}


def test_a_fused_stack_is_refused_under_the_free_sum(fused, tmp_path):
    directory, _ = fused

    with pytest.raises(ValueError, match="fused expert stack"):
        stream_fit(directory, tmp_path / "out", _fused_cfg("trit_planes"))


@pytest.mark.parametrize("weight_scale", ["learnable_row", "dual", "trit_planes"])
def test_a_fused_stack_is_refused_on_every_axis_bound_grid(
    fused, tmp_path, weight_scale
):
    """A state-dict key does not say which trailing axis of a stack is the output one."""
    directory, _ = fused

    with pytest.raises(ValueError, match="fused expert stack"):
        stream_fit(
            directory, tmp_path / f"out-{weight_scale}", _fused_cfg(weight_scale)
        )


def test_a_per_tensor_stack_fit_does_not_depend_on_the_expert_layout():
    """transformers ships both layouts: GptOss/Llama4 `(experts, in, out)`, Qwen3-MoE
    `(experts, out, in)`. The grid a streamed fit puts on a stack must be the same
    either way, because nothing in a checkpoint says which one it is looking at."""
    ternary_cfg = resolve_ternary_config(_cfg())
    stack = torch.randn(3, 8, 16)

    fitted = fit_tensor(stack, ternary_cfg)
    transposed = fit_tensor(stack.transpose(1, 2).contiguous(), ternary_cfg)

    assert torch.equal(fitted, transposed.transpose(1, 2))


def test_a_fitted_fused_stack_is_reported_as_covered_by_no_manifest_entry(
    fused, tmp_path, caplog
):
    directory, _ = fused

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        report = stream_fit(directory, tmp_path / "out", _fused_cfg())

    assert report.unmanifested == ["model.layers.0.mlp.experts.gate_up_proj"]
    assert "no manifest entry" in caplog.text
    assert "master_bf16 artifact" in caplog.text


def test_a_fused_stack_is_costed_at_the_width_it_actually_ships_at(fused, tmp_path):
    """No packed format reaches it, so pretending it costs ~1.5 bits is the one number
    a user sizing an MoE artifact must not be given."""
    directory, tensors = fused
    stack = tensors["model.layers.0.mlp.experts.gate_up_proj"]

    report = stream_fit(directory, tmp_path / "out", _fused_cfg())

    assert report.bits_per_weight > 8.0
    assert report.bits_per_weight * report.weights_fitted >= stack.numel() * 16


def test_a_resumed_pass_still_reports_the_unmanifested_stack(fused, tmp_path):
    directory, _ = fused
    output = tmp_path / "out"
    stream_fit(directory, output, _fused_cfg())

    report = stream_fit(directory, output, _fused_cfg())

    assert report.shards_skipped == report.shards
    assert report.unmanifested == ["model.layers.0.mlp.experts.gate_up_proj"]


def test_the_router_and_the_tower_are_written_through_untouched(fused, tmp_path):
    directory, tensors = fused

    stream_fit(directory, tmp_path / "out", _fused_cfg())

    streamed = bake.load_tensors(tmp_path / "out")
    for key in (
        "model.layers.0.mlp.gate.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
    ):
        assert torch.equal(streamed[key], tensors[key]), key


def test_the_solve_device_is_not_where_the_result_lands():
    """The shards are read and written on the host whatever the solver ran on."""
    ternary_cfg = resolve_ternary_config(_cfg())
    weight = torch.randn(8, 16, dtype=torch.bfloat16)

    fitted = fit_tensor(weight, ternary_cfg, device=torch.device("cpu"))

    assert fitted.device == weight.device
    assert fitted.dtype == weight.dtype


def test_fit_tensor_refuses_a_rank_that_is_not_a_matrix_or_a_stack():
    ternary_cfg = resolve_ternary_config(_cfg())

    with pytest.raises(ValueError, match="2-D weight or a 3-D"):
        fit_tensor(torch.randn(8), ternary_cfg)


def test_a_targeted_integer_tensor_is_an_error_not_a_silent_fit(fused, tmp_path):
    directory, _ = fused
    quantized = {
        "model.layers.9.self_attn.q_proj.weight": torch.ones(4, 4).to(torch.int8)
    }
    save_file(
        quantized, directory / "model-00003-of-00003.safetensors", {"format": "pt"}
    )
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.layers.9.self_attn.q_proj.weight"] = (
        "model-00003-of-00003.safetensors"
    )
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))

    with pytest.raises(ValueError, match="torch.int8"):
        stream_fit(directory, tmp_path / "out", _fused_cfg())


# ------------------------------------------------------------------- resumption


def _interrupt_after(monkeypatch, shards: int) -> None:
    """Kill the pass between shards, exactly where a resumed run has to pick up."""
    real = stream.write_shard_atomic
    written: list[str] = []

    def _write(tensors, path, metadata=None):
        if len(written) >= shards:
            raise KeyboardInterrupt("simulated kill")
        written.append(str(path))
        return real(tensors, path, metadata)

    monkeypatch.setattr(stream, "write_shard_atomic", _write)


def test_a_resumed_pass_is_byte_identical_to_an_uninterrupted_one(
    checkpoint, tmp_path, monkeypatch
):
    clean = tmp_path / "clean"
    stream_fit(checkpoint, clean, _cfg())

    resumed = tmp_path / "resumed"
    _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        stream_fit(checkpoint, resumed, _cfg())
    monkeypatch.undo()

    report = stream_fit(checkpoint, resumed, _cfg())

    assert report.shards_skipped == 1
    assert report.shards_fitted == report.shards - 1
    assert report.tensors_fitted == 14
    assert _digests(resumed) == _digests(clean)


def test_a_resumed_pass_keeps_the_manifest_complete(checkpoint, tmp_path, monkeypatch):
    """A skipped shard's entries come out of the manifest, not out of the shard."""
    resumed = tmp_path / "resumed"
    _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        stream_fit(checkpoint, resumed, _cfg("dual"))
    monkeypatch.undo()

    stream_fit(checkpoint, resumed, _cfg("dual"))

    manifest = SwapManifest.load(resumed)
    assert len(manifest.entries) == 14
    assert len(manifest.scales) == 14
    assert manifest.kept_fp


def test_an_interrupted_pass_leaves_no_half_written_shard(
    checkpoint, tmp_path, monkeypatch
):
    output = tmp_path / "killed"
    _interrupt_after(monkeypatch, 1)

    with pytest.raises(KeyboardInterrupt):
        stream_fit(checkpoint, output, _cfg())

    assert not list(output.glob(f"*{stream.PARTIAL_SUFFIX}*"))
    assert len(load_records(output)) == 1
    for path in output.glob("*.safetensors"):
        assert bake.load_shard(path)[0]


def test_a_torn_shard_is_refitted_rather_than_trusted(checkpoint, tmp_path):
    clean = tmp_path / "clean"
    stream_fit(checkpoint, clean, _cfg())
    torn = tmp_path / "torn"
    stream_fit(checkpoint, torn, _cfg())
    victim = sorted(torn.glob("*.safetensors"))[0]
    victim.write_bytes(victim.read_bytes()[:-8])

    report = stream_fit(checkpoint, torn, _cfg())

    assert report.shards_fitted == 1
    assert _digests(torn) == _digests(clean)


def test_no_resume_refits_every_shard(checkpoint, tmp_path):
    output = tmp_path / "out"
    stream_fit(checkpoint, output, _cfg())

    report = stream_fit(checkpoint, output, _cfg(), resume=False)

    assert report.shards_skipped == 0
    assert report.shards_fitted == report.shards


def test_a_resume_onto_a_different_grid_refits_every_shard(checkpoint, tmp_path):
    """The records are keyed to the grid; half a checkpoint on each is not a model."""
    output = tmp_path / "out"
    stream_fit(checkpoint, output, _cfg("learnable"))

    report = stream_fit(checkpoint, output, _cfg("learnable_row"))

    assert report.shards_skipped == 0
    resident = _in_memory_fit(checkpoint, _cfg("learnable_row"))
    for key, tensor in bake.load_tensors(output).items():
        assert torch.equal(tensor, resident[key]), key


def test_a_resume_from_a_different_checkpoint_refits_every_shard(tmp_path, caplog):
    """The record has to key on the input too: A/B-ing two base checkpoints into one
    output directory would otherwise ship a fitted copy of the first one."""
    source = _write_checkpoint(tmp_path / "src", seed=0)
    output = tmp_path / "out"
    stream_fit(source, output, _cfg())
    shutil.rmtree(source)
    _write_checkpoint(source, seed=1)

    caplog.clear()
    with capture_axolotl_warnings(caplog):
        report = stream_fit(source, output, _cfg())

    assert report.shards_skipped == 0
    assert report.shards_fitted == report.shards
    assert "different input bytes" in caplog.text
    resident = _in_memory_fit(source, _cfg())
    for key, tensor in bake.load_tensors(output).items():
        assert torch.equal(tensor, resident[key]), key


def test_a_record_names_the_input_shard_it_was_fitted_from(checkpoint, tmp_path):
    output = tmp_path / "out"

    stream_fit(checkpoint, output, _cfg())

    records = load_records(output)
    for path in bake.shard_paths(checkpoint):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert records[path.name].input_sha256 == digest


def test_a_torn_scale_sidecar_refits_rather_than_raising(checkpoint, tmp_path):
    """`SwapManifest.save` writes the sidecar after the JSON and without a rename, so
    an interruption inside that window is a state a resume has to survive."""
    output = tmp_path / "out"
    stream_fit(checkpoint, output, _cfg("dual"))
    sidecar = output / "ternary_scales.safetensors"
    sidecar.write_bytes(sidecar.read_bytes()[:20])

    report = stream_fit(checkpoint, output, _cfg("dual"))

    assert report.shards_skipped == 0
    assert len(SwapManifest.load(output).scales) == 14


def _record_fsyncs(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = os.fsync

    def _fsync(handle):
        calls.append(handle)
        return real(handle)

    monkeypatch.setattr(os, "fsync", _fsync)
    return calls


def test_a_shard_reaches_the_device_before_it_is_renamed(tmp_path, monkeypatch):
    """Rename-only atomicity survives a killed process, not a lost host."""
    calls = _record_fsyncs(monkeypatch)

    write_shard_atomic({"a": torch.ones(2, 2)}, tmp_path / "shard.safetensors")

    assert len(calls) == 2


def test_the_records_reach_the_device_before_they_are_renamed(tmp_path, monkeypatch):
    calls = _record_fsyncs(monkeypatch)

    write_records(tmp_path, {})

    assert len(calls) == 2


def test_a_resume_without_a_manifest_refits_every_shard(checkpoint, tmp_path):
    output = tmp_path / "out"
    stream_fit(checkpoint, output, _cfg())
    (output / "ternary_manifest.json").unlink()

    report = stream_fit(checkpoint, output, _cfg())

    assert report.shards_skipped == 0
    assert len(SwapManifest.load(output).entries) == 14


def test_the_report_totals_cover_the_work_a_resume_skipped(
    checkpoint, tmp_path, monkeypatch
):
    clean = stream_fit(checkpoint, tmp_path / "clean", _cfg())

    resumed_dir = tmp_path / "resumed"
    _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        stream_fit(checkpoint, resumed_dir, _cfg())
    monkeypatch.undo()
    resumed = stream_fit(checkpoint, resumed_dir, _cfg())

    assert resumed.tensors_fitted == clean.tensors_fitted
    assert resumed.tensors_copied == clean.tensors_copied
    assert resumed.weights_fitted == clean.weights_fitted
    assert resumed.families == clean.families
    assert resumed.bits_per_weight == pytest.approx(clean.bits_per_weight)


# ---------------------------------------------------------------- records on disk


def test_records_survive_a_round_trip(tmp_path):
    records = {
        "shard.safetensors": ShardRecord(
            shard="shard.safetensors",
            tensors=3,
            fitted=2,
            copied=1,
            sha256="abc",
            fingerprint="def",
            weights=64,
            zeros=32,
            families={"q_proj": 2},
        )
    }

    path = write_records(tmp_path, records)

    assert path.name == stream.RECORD_FILENAME
    assert load_records(tmp_path) == records


@pytest.mark.parametrize("payload", ["", "{", "[]", '{"a": 1}', '{"a": {"shard": 1}}'])
def test_an_unreadable_record_file_means_nothing_finished(tmp_path, payload):
    (tmp_path / stream.RECORD_FILENAME).write_text(payload)

    assert load_records(tmp_path) == {}


def test_records_are_written_through_a_temporary(tmp_path):
    write_records(tmp_path, {})

    assert not list(tmp_path.glob(f"*{stream.PARTIAL_SUFFIX}*"))
    assert (tmp_path / stream.RECORD_FILENAME).is_file()


def test_an_atomic_shard_write_keeps_the_safetensors_format(tmp_path):
    path = write_shard_atomic({"a": torch.ones(2, 2)}, tmp_path / "shard.safetensors")

    tensors, metadata = bake.load_shard(path)
    assert torch.equal(tensors["a"], torch.ones(2, 2))
    assert metadata["format"] == "pt"
    assert not list(tmp_path.glob(f"*{stream.PARTIAL_SUFFIX}*"))


# ---------------------------------------------------------- artifacts and stamps


def test_the_pass_writes_a_manifest_a_master_can_be_reloaded_from(checkpoint, tmp_path):
    output = tmp_path / "out"

    stream_fit(checkpoint, output, _cfg("dual"))

    manifest = SwapManifest.load(output)
    assert manifest.model_type == "llama"
    assert manifest.weight_scale == "dual"
    assert manifest.init == "ternary_fit"
    assert manifest.quantizer["scheme"] == "dual_ternary"
    assert manifest.quantizer["weight_dequant_scale"] == "f16_round"
    assert len(manifest.entries) == 14
    assert "model.embed_tokens" in manifest.kept_fp
    assert (output / "ternary_scales.safetensors").is_file()


def test_a_per_tensor_grid_ships_no_scale_sidecar(checkpoint, tmp_path):
    output = tmp_path / "out"

    stream_fit(checkpoint, output, _cfg("learnable"))

    assert not (output / "ternary_scales.safetensors").exists()
    assert not SwapManifest.load(output).scales


def test_the_output_config_carries_the_quantizer_stamp(checkpoint, tmp_path):
    output = tmp_path / "out"

    stream_fit(checkpoint, output, _cfg())

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    stamp = config[bake.QUANTIZER_METADATA_KEY]
    assert stamp["scheme"] == "ternary"
    assert stamp["weight_scale"] == "learnable"
    assert stamp["modules"] == 14


def test_the_shard_layout_is_preserved(checkpoint, tmp_path):
    output = tmp_path / "out"

    report = stream_fit(checkpoint, output, _cfg())

    assert report.shards > 1
    assert {path.name for path in bake.shard_paths(output)} == {
        path.name for path in bake.shard_paths(checkpoint)
    }
    assert (output / "model.safetensors.index.json").is_file()
    assert (output / "generation_config.json").is_file()


def test_the_measured_cost_is_the_sparsity_the_solver_found(checkpoint, tmp_path):
    report = stream_fit(checkpoint, tmp_path / "out", _cfg())

    assert 0.0 < report.zero_fraction < 1.0
    assert report.bits_per_weight == pytest.approx(2.0 - report.zero_fraction)
    assert report.weights_fitted == 14 * 32 * 32 + 6 * 32 * 64 - 8 * 32 * 32
    assert report.weights_per_second > 0.0


@pytest.mark.parametrize("weight_scale,fmt", [("dual", "fv5"), ("trit_planes", "tp9")])
def test_a_two_plane_cost_is_the_container_it_packs_into(
    checkpoint, tmp_path, weight_scale, fmt
):
    """The two-plane grids have flat-cost containers; `2 - zero_fraction` is not their
    cost and understates them by more than 2x, which is the number this command
    exists to give."""
    output = tmp_path / f"cost-{weight_scale}"

    report = stream_fit(checkpoint, output, _cfg(weight_scale))

    manifest = SwapManifest.load(output)
    expected = sum(
        bitplanes.bits_per_weight(fmt, (entry.out_features, entry.in_features))
        * entry.out_features
        * entry.in_features
        for entry in manifest.entries
    )
    assert report.bits_per_weight == pytest.approx(expected / report.weights_fitted)
    assert report.bits_per_weight > 3.0


def test_the_reported_cost_is_the_one_the_packer_measures(checkpoint, tmp_path):
    """One accounting, so the report and the artifact cannot drift apart."""
    master = tmp_path / "master"
    report = stream_fit(checkpoint, master, _cfg("dual"))
    manifest = SwapManifest.load(master)

    bitplanes.export_bitplanes(master, tmp_path / "fv5", manifest, "fv5")

    record = json.loads((tmp_path / "fv5" / "ternary_fv5.json").read_text())
    assert record["bits_per_weight"] == pytest.approx(report.bits_per_weight, rel=1e-6)


def test_the_head_and_the_embedding_are_the_kept_full_precision_modules(
    checkpoint, tmp_path
):
    """A tied head owns no tensor, so nothing the pass reads names it — and a packer
    that is not told about it wraps it in a Linear whose codes are not in the file."""
    output = tmp_path / "out"

    stream_fit(checkpoint, output, _cfg())

    kept = SwapManifest.load(output).kept_fp
    assert "lm_head" in kept
    assert "model.embed_tokens" in kept
    assert not [name for name in kept if "norm" in name]


def test_a_streamed_tied_master_exports_a_loadable_hf_bitnet(checkpoint, tmp_path):
    pytest.importorskip("transformers.integrations.bitnet")
    cfg = _cfg(export={"formats": ["hf_bitnet"]})
    master = tmp_path / "master"
    stream_fit(checkpoint, master, cfg)

    artifact = run_export(cfg, master_dir=master, output_dir=tmp_path / "exports")

    packed = AutoModelForCausalLM.from_pretrained(
        artifact["hf_bitnet"], dtype=torch.float32
    )
    assert type(packed.model.layers[0].self_attn.q_proj).__name__ == "BitLinear"
    assert type(packed.lm_head).__name__ == "Linear"


# ------------------------------------------------------------------- refusals


def test_the_calibrated_init_is_refused(checkpoint, tmp_path):
    cfg = _cfg("learnable", init="ternary_fit_calibrated")

    with pytest.raises(ValueError, match="layer-streaming calibration"):
        stream_fit(checkpoint, tmp_path / "out", cfg)


def test_a_non_fitting_init_is_refused(checkpoint, tmp_path):
    cfg = DictDefault({"ternary": {"init": "absmean", "export": MASTER_ONLY}})

    with pytest.raises(ValueError, match="solve nothing"):
        stream_fit(checkpoint, tmp_path / "out", cfg)


def test_an_init_without_a_streamed_solver_is_refused(checkpoint, tmp_path):
    with pytest.raises(NotImplementedError, match="svid"):
        stream_fit(checkpoint, tmp_path / "out", _cfg("learnable", init="svid"))


def test_the_smoothing_knob_is_refused_rather_than_ignored(checkpoint, tmp_path):
    """`axolotl train` refuses it; a streamed fit that ignored it would be the one
    entry point where the same config means something else."""
    with pytest.raises(NotImplementedError, match="smoothing"):
        stream_fit(checkpoint, tmp_path / "out", _cfg(smoothing=True))


def test_the_subln_knob_says_the_streamed_master_does_not_carry_it(
    checkpoint, tmp_path, caplog
):
    caplog.clear()
    with capture_axolotl_warnings(caplog):
        stream_fit(checkpoint, tmp_path / "out", _cfg(subln=True))

    assert "subln" in caplog.text
    assert not SwapManifest.load(tmp_path / "out").subln


def test_writing_over_the_input_is_refused(checkpoint):
    with pytest.raises(ValueError, match="cannot overwrite its input"):
        stream_fit(checkpoint, checkpoint, _cfg())


def test_a_directory_without_weights_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        stream_fit(tmp_path, tmp_path / "out", _cfg())


def test_an_unknown_architecture_without_a_target_list_is_refused(tmp_path):
    directory = tmp_path / "mystery"
    directory.mkdir()
    save_file({"a.weight": torch.ones(2, 2)}, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps({"model_type": "mystery"}))

    with pytest.raises(ValueError, match="no ternary arch preset"):
        stream_fit(directory, tmp_path / "out", _cfg())


# -------------------------------------------------------------- binary codebook


@pytest.mark.skipif(BINARY_LANDED, reason="the binary quantizer has landed")
def test_the_binary_codebook_is_refused_until_its_quantizer_lands(checkpoint, tmp_path):
    with pytest.raises(NotImplementedError, match="codebook"):
        stream_fit(checkpoint, tmp_path / "out", _cfg(codebook="binary"))


@requires_binary
@pytest.mark.parametrize("weight_scale", ["learnable", "learnable_row"])
def test_a_binary_streamed_fit_holds_only_two_magnitudes(
    checkpoint, tmp_path, weight_scale
):
    output = tmp_path / f"binary-{weight_scale}"

    stream_fit(checkpoint, output, _cfg(weight_scale, codebook="binary"))

    manifest = SwapManifest.load(output)
    tensors = bake.load_tensors(output)
    for entry in manifest.entries:
        weight = tensors[f"{entry.name}.weight"]
        group_size = entry.in_features if weight_scale == "learnable_row" else None
        assert quant.baked_binary_codes_and_scale(weight, group_size) is not None
        assert not bool((weight == 0).any())


@requires_binary
def test_a_binary_fused_stack_is_fitted_per_slice(fused, tmp_path):
    directory, tensors = fused
    key = "model.layers.0.mlp.experts.gate_up_proj"
    cfg = _fused_cfg(codebook="binary")

    stream_fit(directory, tmp_path / "out", cfg)

    ternary_cfg = resolve_ternary_config(cfg)
    streamed = bake.load_tensors(tmp_path / "out")[key]
    for index, expert in enumerate(tensors[key]):
        assert torch.equal(streamed[index], fit_tensor(expert, ternary_cfg)), index


@requires_binary
def test_a_binary_pass_resumes_byte_identically(checkpoint, tmp_path, monkeypatch):
    cfg = _cfg(codebook="binary")
    clean = tmp_path / "clean"
    stream_fit(checkpoint, clean, cfg)

    resumed = tmp_path / "resumed"
    _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        stream_fit(checkpoint, resumed, cfg)
    monkeypatch.undo()

    report = stream_fit(checkpoint, resumed, cfg)

    assert report.shards_skipped == 1
    assert report.bits_per_weight == 1.0
    assert _digests(resumed) == _digests(clean)


@requires_binary
def test_the_binary_codebook_costs_one_bit_a_weight(checkpoint, tmp_path):
    report = stream_fit(checkpoint, tmp_path / "out", _cfg(codebook="binary"))

    assert report.zero_fraction == 0.0
    assert report.bits_per_weight == 1.0


# ------------------------------------------------------------------------- CLI


@pytest.fixture(name="stub_load_cfg")
def fixture_stub_load_cfg(monkeypatch):
    """Replaces `load_cfg` with a plain YAML read (no validation, no GPU probing)."""

    def _load_cfg(config, **kwargs):
        with open(config, encoding="utf-8") as file:
            return DictDefault(yaml.safe_load(file))

    monkeypatch.setattr("axolotl.cli.config.load_cfg", _load_cfg, raising=True)


def _config_file(path, checkpoint, output, **ternary):
    payload = {
        "base_model": str(checkpoint),
        "output_dir": str(output),
        "plugins": ["axolotl.integrations.ternary.TernaryPlugin"],
        "ternary": {
            "init": "ternary_fit",
            "weight_scale": "learnable",
            "lambda_schedule": "none",
            "export": MASTER_ONLY,
            **ternary,
        },
    }
    path.write_text(yaml.safe_dump(payload))
    return path


def test_the_cli_fits_the_configured_checkpoint(checkpoint, tmp_path, stub_load_cfg):
    output = tmp_path / "cli-out"
    config = _config_file(tmp_path / "config.yml", checkpoint, output)

    result = CliRunner().invoke(cli, ["ternary", "fit-stream", str(config)])

    assert result.exit_code == 0, result.output
    assert "14 tensors fitted over 3/3 shards (0 skipped)" in result.output
    assert "bits per weight" in result.output
    assert "per family: " in result.output
    assert SwapManifest.load(output).entries


def test_the_cli_emits_json(checkpoint, tmp_path, stub_load_cfg):
    output = tmp_path / "cli-out"
    config = _config_file(tmp_path / "config.yml", checkpoint, output)

    result = CliRunner().invoke(cli, ["ternary", "fit-stream", str(config), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tensors_fitted"] == 14
    assert payload["weight_scale"] == "learnable"
    assert payload["records"]
    assert payload["bits_per_weight"] < 2.0


def test_the_cli_honours_its_directory_overrides(checkpoint, tmp_path, stub_load_cfg):
    config = _config_file(tmp_path / "config.yml", tmp_path / "absent", tmp_path / "no")
    output = tmp_path / "elsewhere"

    result = CliRunner().invoke(
        cli,
        [
            "ternary",
            "fit-stream",
            str(config),
            "--input-dir",
            str(checkpoint),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "ternary_manifest.json").is_file()
    assert not (tmp_path / "no").exists()


def test_the_cli_skips_finished_shards_on_a_resume(checkpoint, tmp_path, stub_load_cfg):
    output = tmp_path / "cli-out"
    config = _config_file(tmp_path / "config.yml", checkpoint, output)
    runner = CliRunner()
    runner.invoke(cli, ["ternary", "fit-stream", str(config)])

    result = runner.invoke(cli, ["ternary", "fit-stream", str(config)])

    assert result.exit_code == 0, result.output
    assert "0/3 shards (3 skipped)" in result.output


def test_the_cli_refuses_a_calibrated_init_before_touching_the_disk(
    checkpoint, tmp_path, stub_load_cfg
):
    output = tmp_path / "cli-out"
    config = _config_file(
        tmp_path / "config.yml", checkpoint, output, init="ternary_fit_calibrated"
    )

    result = CliRunner().invoke(cli, ["ternary", "fit-stream", str(config)])

    assert result.exit_code != 0
    assert "layer-streaming calibration" in result.output
    assert not output.exists()


def test_the_cli_requires_the_plugin(checkpoint, tmp_path, stub_load_cfg):
    config = _config_file(tmp_path / "config.yml", checkpoint, tmp_path / "out")
    payload = yaml.safe_load(config.read_text())
    payload["plugins"] = []
    config.write_text(yaml.safe_dump(payload))

    result = CliRunner().invoke(cli, ["ternary", "fit-stream", str(config)])

    assert result.exit_code != 0
    assert "does not enable the ternary plugin" in result.output
