"""CPU-only contract tests for the wave-3 seams: the stubs and the signatures around them.

These lock the shapes the implementing passes have to fill — every stub must be
present, callable, and refuse to run rather than quietly return something plausible.
"""

import inspect

import pytest
import torch

from axolotl.integrations.ternary import callbacks as callbacks_module, quant
from axolotl.integrations.ternary.export import bitplanes
from axolotl.integrations.ternary.ptq import stream

WEIGHT = torch.tensor([[0.5, -0.25], [0.125, -0.75]])


def _signature(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


# ---------------------------------------------------------------- binary codebook


@pytest.mark.parametrize(
    "name,args",
    [
        ("binary_scale", (WEIGHT,)),
        ("binary_codes", (WEIGHT, torch.tensor(0.5))),
        ("fake_quant_weight_binary", (WEIGHT,)),
        ("fake_quant_weight_binary_ste", (WEIGHT,)),
    ],
)
def test_the_binary_quantizer_returns_a_tensor(name, args):
    assert isinstance(getattr(quant, name)(*args), torch.Tensor)


def test_the_baked_binary_probe_reads_a_latent_as_latent():
    assert quant.baked_binary_codes_and_scale(WEIGHT) is None


def test_the_binary_entry_points_mirror_the_ternary_ones():
    assert _signature(quant.binary_scale) == _signature(quant.absmean_scale)
    assert _signature(quant.binary_codes) == _signature(quant.ternary_codes)
    assert _signature(quant.fake_quant_weight_binary) == _signature(
        quant.fake_quant_weight
    )
    assert _signature(quant.fake_quant_weight_binary_ste) == _signature(
        quant.fake_quant_weight_ste
    )
    assert _signature(quant.baked_binary_codes_and_scale) == _signature(
        quant.baked_codes_and_scale
    )


def test_the_binary_ste_matches_the_ternary_one():
    assert issubclass(quant.FakeQuantWeightBinarySTE, torch.autograd.Function)
    assert _signature(quant.FakeQuantWeightBinarySTE.forward) == _signature(
        quant.FakeQuantWeightSTE.forward
    )


def test_the_binary_scale_documents_its_closed_form():
    """`mean|w|` is the least-squares optimum here, not the threshold heuristic."""
    doc = quant.binary_scale.__doc__

    assert "mean(|W|)" in doc
    assert "least-squares" in doc


def test_the_baked_binary_probe_documents_the_degenerate_case():
    doc = quant.baked_binary_codes_and_scale.__doc__

    assert "all share one sign" in doc


# -------------------------------------------------------------------- sign_plane


@pytest.mark.parametrize(
    "name,args",
    [
        ("pack_sign_plane", (torch.ones(2, 2, dtype=torch.int8),)),
        ("unpack_sign_plane", (torch.zeros(1, dtype=torch.uint8), (2, 2))),
        (
            "decode_sign_plane",
            (torch.zeros(1, dtype=torch.uint8), torch.tensor(0.5), (2, 2)),
        ),
    ],
)
def test_the_sign_plane_container_answers_for_every_entry_point(name, args):
    assert getattr(bitplanes, name)(*args) is not None


def test_sign_plane_is_registered_as_the_binary_container():
    assert bitplanes.FORMAT_CODEBOOKS[bitplanes.SIGN_PLANE_FORMAT] == "binary"
    assert bitplanes.SIGN_PLANE_FORMAT not in bitplanes.FORMAT_SCALE_MODES
    assert bitplanes.SIGN_PLANE_BITS_PER_WEIGHT == 1.0


def test_the_two_plane_containers_stay_ternary():
    for fmt in (bitplanes.FV5_FORMAT, bitplanes.TP9_FORMAT):
        assert fmt not in bitplanes.FORMAT_CODEBOOKS


def test_the_sign_plane_export_mirrors_the_bitplane_export():
    signature = _signature(bitplanes.export_sign_plane)

    assert signature == _signature(bitplanes.export_bitplanes)[:3]


# -------------------------------------------------------------------- fit-stream


def test_the_stream_entry_point_has_the_agreed_signature():
    assert _signature(stream.stream_fit) == [
        "input_dir",
        "output_dir",
        "cfg",
        "device",
        "resume",
    ]
    defaults = inspect.signature(stream.stream_fit).parameters
    assert defaults["device"].default == "cpu"
    assert defaults["resume"].default is True


def test_the_stream_pass_answers_for_every_entry_point(tmp_path):
    from axolotl.integrations.ternary.args import TernaryConfig

    ternary_cfg = TernaryConfig(target_modules=[r".*\.mlp\.up_proj"])

    assert stream.classify_tensor("model.layers.0.mlp.up_proj.weight", ternary_cfg) == (
        "fit"
    )
    assert stream.fit_tensor(WEIGHT, ternary_cfg).shape == WEIGHT.shape
    assert stream.load_records(tmp_path) == {}
    assert stream.write_records(tmp_path, {}).is_file()
    assert stream.write_shard_atomic(
        {"w": WEIGHT}, tmp_path / "s.safetensors"
    ).is_file()


def test_the_stream_report_round_trips_to_json():
    record = stream.ShardRecord(
        shard="model-00001-of-00002.safetensors",
        tensors=10,
        fitted=7,
        copied=3,
        sha256="abc",
    )
    report = stream.StreamFitReport(
        input_dir="in",
        output_dir="out",
        codebook="ternary",
        init="ternary_fit",
        weight_scale="learnable",
        shards=2,
        shards_fitted=1,
        shards_skipped=1,
        tensors_fitted=7,
        tensors_copied=3,
        records={record.shard: record},
    )

    data = report.to_dict()

    assert data["records"][record.shard]["fitted"] == 7
    assert data["tensors_fitted"] == 7


def test_the_partial_suffix_is_what_makes_a_record_trustworthy():
    assert stream.PARTIAL_SUFFIX.startswith(".")
    assert "temporary" in stream.write_shard_atomic.__doc__


def test_the_stream_module_documents_why_classification_is_by_name():
    assert "state-dict key" in stream.classify_tensor.__doc__
    assert "aux_modules" in stream.__doc__


def test_the_stream_module_documents_the_fused_expert_slice_rule():
    assert "slice at a time" in stream.fit_tensor.__doc__


# --------------------------------------------------------------- the CLI command


def test_fit_stream_is_a_ternary_subcommand():
    from axolotl.integrations.ternary.cli import ternary

    command = ternary.commands["fit-stream"]
    options = {option.name for option in command.params if option.name != "config"}

    assert options == {"input_dir", "output_dir", "device", "resume", "as_json"}


def test_fit_stream_does_not_import_the_stream_engine_at_definition_time():
    """The CLI stays lazy: `--help` must not pay for torch-heavy imports."""
    import axolotl.integrations.ternary.cli as cli_module

    source = inspect.getsource(cli_module.fit_stream_command.callback)

    assert "from .ptq.stream import stream_fit" in source


# ----------------------------------------------------- zero-grad-while-targeted


def test_the_zero_grad_warning_is_a_trainer_callback():
    from transformers import TrainerCallback

    assert issubclass(callbacks_module.ZeroGradWarningCallback, TrainerCallback)


def test_the_zero_grad_warning_tracks_the_targeted_modules():
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.integrations.ternary.swap import convert_model
    from axolotl.utils.dict import DictDefault

    torch.manual_seed(0)
    model = LlamaForCausalLM(
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
    convert_model(model, DictDefault({"ternary": {}}))

    callback = callbacks_module.ZeroGradWarningCallback(model)

    assert len(callback.ternary_modules) == 7
    assert callback.monitored_steps == callbacks_module.ZERO_GRAD_MONITORED_STEPS
    assert callback.seen_grad == set()


def test_the_zero_grad_warning_is_installed_by_the_plugin():
    from axolotl.integrations.ternary import TernaryPlugin

    source = inspect.getsource(TernaryPlugin.add_callbacks_pre_trainer)

    assert "ZeroGradWarningCallback" in source
