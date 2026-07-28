"""CPU-only schema-validation tests for the ternary plugin config."""

import pydantic
import pytest

from axolotl.integrations.ternary.args import (
    TernaryArgs,
    TernaryConfig,
    resolve_ternary_config,
)


def test_defaults_match_design():
    cfg = TernaryConfig()

    assert cfg.target_modules is None
    assert cfg.keep_fp_modules is None
    assert cfg.strict_enumeration is True
    assert cfg.weight_scale == "absmean"
    assert cfg.group_size is None
    assert cfg.activation_bits == 8
    assert cfg.lambda_schedule == "linear"
    assert cfg.lambda_warmup_steps == 1000
    assert cfg.weight_decay_zero_at == 0.5
    assert cfg.init == "absmean"
    assert cfg.init_calibration.num_sequences == 128
    assert cfg.init_calibration.sequence_len == 2048
    assert cfg.init_calibration.dataset is None
    assert cfg.smoothing is False
    assert cfg.subln is False
    assert cfg.int8_forward == "auto"
    assert cfg.fused_fake_quant is True
    assert cfg.log_code_flip_every == 100
    assert cfg.distill.mode is None
    assert cfg.distill.logits_weight == 0.5
    assert cfg.distill.logits_temperature == 2.0
    assert cfg.distill.hidden_weight == 0.0
    assert cfg.distill.attn_relation_layer is None
    assert cfg.distill.schedule == "constant"
    assert cfg.distill.anchor_start == 0.9
    assert cfg.export.formats == ["master_bf16", "hf_bitnet"]
    assert cfg.export.run_parity_gate is True


def test_full_config_parses():
    args = TernaryArgs.model_validate(
        {
            "ternary": {
                "target_modules": [r".*\.self_attn\.(q|k|v|o)_proj"],
                "keep_fp_modules": [r"lm_head"],
                "strict_enumeration": False,
                "weight_scale": "learnable_row",
                "activation_bits": None,
                "lambda_schedule": "sigmoid",
                "lambda_warmup_steps": 0.1,
                "weight_decay_zero_at": 0.25,
                "init": "ternary_fit_calibrated",
                "init_calibration": {
                    "num_sequences": 64,
                    "sequence_len": 1024,
                    "dataset": "HuggingFaceFW/fineweb-edu",
                },
                "smoothing": True,
                "distill": {
                    "mode": "inprocess",
                    "teacher_model": "meta-llama/Llama-3.2-1B",
                    "logits_weight": 0.5,
                    "logits_temperature": 5.0,
                    "hidden_weight": 0.1,
                    "attn_relation_layer": -2,
                    "schedule": "anchored",
                    "anchor_start": 0.8,
                },
                "int8_forward": False,
                "fused_fake_quant": False,
                "log_code_flip_every": 0,
                "export": {"formats": ["master_bf16"], "run_parity_gate": False},
            }
        }
    )

    assert args.ternary.weight_scale == "learnable_row"
    assert args.ternary.group_size is None
    assert args.ternary.activation_bits is None
    assert args.ternary.lambda_warmup_steps == 0.1
    assert args.ternary.int8_forward is False
    assert args.ternary.init_calibration.num_sequences == 64
    assert args.ternary.distill.attn_relation_layer == -2
    assert args.ternary.distill.schedule == "anchored"
    assert args.ternary.distill.anchor_start == 0.8
    assert args.ternary.export.formats == ["master_bf16"]


def test_plugin_args_default_to_a_ternary_block():
    assert TernaryArgs().ternary == TernaryConfig()


@pytest.mark.parametrize("adapter", ["lora", "qlora"])
def test_adapter_rejected(adapter):
    with pytest.raises(pydantic.ValidationError, match="full-finetune only"):
        TernaryArgs.model_validate({"adapter": adapter, "ternary": {}})


@pytest.mark.parametrize(
    "key,value",
    [
        ("use_onebitllms", True),
        ("qat", {"activation_dtype": "int8"}),
        ("quantization", {"weight_dtype": "int8"}),
        ("load_in_8bit", True),
        ("load_in_4bit", True),
    ],
)
def test_conflicting_quantization_keys_rejected(key, value):
    with pytest.raises(pydantic.ValidationError, match="cannot be combined"):
        TernaryArgs.model_validate({key: value, "ternary": {}})


def test_unrelated_keys_are_ignored():
    args = TernaryArgs.model_validate({"base_model": "meta-llama/Llama-3.2-1B"})
    assert args.ternary.weight_scale == "absmean"


def test_group_size_requires_group_scale():
    with pytest.raises(pydantic.ValidationError, match="weight_scale: group"):
        TernaryConfig(group_size=128)


def test_group_scale_requires_group_size():
    with pytest.raises(pydantic.ValidationError, match="requires ternary.group_size"):
        TernaryConfig(weight_scale="group")


@pytest.mark.parametrize("group_size", [0, 1, 8, 31])
def test_group_size_below_the_minimum_is_rejected(group_size):
    """At group_size 1 every value is its own scale, so the quantizer is a no-op."""
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(
            weight_scale="group",
            group_size=group_size,
            export={"formats": ["master_bf16"]},
        )


@pytest.mark.parametrize("group_size", [32, 128, 256])
def test_group_size_at_or_above_the_minimum_is_accepted(group_size):
    cfg = TernaryConfig(
        weight_scale="group",
        group_size=group_size,
        export={"formats": ["master_bf16"]},
    )
    assert cfg.group_size == group_size


@pytest.mark.parametrize(
    "fmt", ["hf_bitnet", "gguf_tq2_0", "gguf_tq1_0", "i2_s", "mask_sign"]
)
def test_group_scale_rejected_for_per_tensor_export_formats(fmt):
    with pytest.raises(pydantic.ValidationError, match="cannot be represented"):
        TernaryConfig(
            weight_scale="group",
            group_size=128,
            export={"formats": ["master_bf16", fmt]},
        )


def test_group_scale_allowed_with_master_only_export():
    cfg = TernaryConfig(
        weight_scale="group", group_size=128, export={"formats": ["master_bf16"]}
    )
    assert cfg.export.formats == ["master_bf16"]


@pytest.mark.parametrize("mode", ["learnable_row", "dual"])
def test_row_and_dual_scale_modes_accepted_with_master_only_export(mode):
    cfg = TernaryConfig(weight_scale=mode, export={"formats": ["master_bf16"]})
    assert cfg.weight_scale == mode
    assert cfg.group_size is None


@pytest.mark.parametrize("mode", ["learnable_row", "dual"])
@pytest.mark.parametrize(
    "fmt", ["hf_bitnet", "gguf_tq2_0", "gguf_tq1_0", "i2_s", "mask_sign"]
)
def test_row_and_dual_scale_modes_rejected_for_per_tensor_formats(mode, fmt):
    with pytest.raises(pydantic.ValidationError, match="cannot be represented"):
        TernaryConfig(weight_scale=mode, export={"formats": ["master_bf16", fmt]})


@pytest.mark.parametrize("mode", ["absmean", "learnable", "learnable_row", "dual"])
def test_group_size_rejected_outside_group_scale(mode):
    """'dual' scales per row, so a group size has nothing to size."""
    with pytest.raises(pydantic.ValidationError, match="only valid with"):
        TernaryConfig(
            weight_scale=mode, group_size=128, export={"formats": ["master_bf16"]}
        )


@pytest.mark.parametrize("mode", ["absmean", "learnable"])
def test_per_tensor_scale_modes_export_to_every_format(mode):
    cfg = TernaryConfig(
        weight_scale=mode,
        export={"formats": ["master_bf16", "hf_bitnet", "mask_sign"]},
    )
    assert cfg.export.formats == ["master_bf16", "hf_bitnet", "mask_sign"]


@pytest.mark.parametrize("fmt", ["gguf_tq2_0", "gguf_tq1_0", "i2_s", "hf_bitnet"])
def test_subln_rejected_for_packed_exports(fmt):
    with pytest.raises(pydantic.ValidationError, match="no tensor"):
        TernaryConfig(subln=True, export={"formats": [fmt]})


def test_subln_rejects_the_default_export_formats():
    """`hf_bitnet` is a default, so the refusal has to happen before training, not after."""
    with pytest.raises(pydantic.ValidationError, match="no tensor"):
        TernaryConfig(subln=True)


def test_subln_allowed_for_master_and_mask_sign():
    cfg = TernaryConfig(subln=True, export={"formats": ["master_bf16", "mask_sign"]})
    assert cfg.subln is True


@pytest.mark.parametrize("steps", [1, 1000, 0.05, 1.0])
def test_lambda_warmup_steps_accepted(steps):
    assert TernaryConfig(lambda_warmup_steps=steps).lambda_warmup_steps == steps


@pytest.mark.parametrize("steps,expected", [(1, int), (1.0, float), (0.5, float)])
def test_lambda_warmup_steps_keeps_its_type(steps, expected):
    """The callback disambiguates absolute steps from a fraction by type."""
    assert isinstance(
        TernaryConfig(lambda_warmup_steps=steps).lambda_warmup_steps, expected
    )


def test_lambda_warmup_steps_narrows_a_bool_to_a_step_count():
    """A YAML `true` must not become a fraction of the run."""
    assert TernaryConfig(lambda_warmup_steps=True).lambda_warmup_steps == 1


@pytest.mark.parametrize("steps", [0, -5])
def test_lambda_warmup_steps_int_must_be_positive(steps):
    with pytest.raises(pydantic.ValidationError, match=">= 1 step"):
        TernaryConfig(lambda_warmup_steps=steps)


@pytest.mark.parametrize("steps", [0.0, 1.5, -0.5])
def test_lambda_warmup_steps_fraction_must_be_in_unit_range(steps):
    with pytest.raises(pydantic.ValidationError, match="fraction of max_steps"):
        TernaryConfig(lambda_warmup_steps=steps)


def test_weight_decay_zero_at_bounds():
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(weight_decay_zero_at=1.5)


def test_activation_bits_only_8_or_none():
    assert TernaryConfig(activation_bits=None).activation_bits is None
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(activation_bits=4)


def test_unknown_enum_values_rejected():
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(weight_scale="absmax")
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(lambda_schedule="cosine")
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(init="magic")
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(export={"formats": ["awq"]})


def test_mask_sign_is_an_export_format():
    cfg = TernaryConfig(export={"formats": ["master_bf16", "mask_sign"]})
    assert cfg.export.formats == ["master_bf16", "mask_sign"]


def test_duplicate_export_formats_rejected():
    with pytest.raises(pydantic.ValidationError, match="duplicate"):
        TernaryConfig(export={"formats": ["hf_bitnet", "hf_bitnet"]})


def test_invalid_target_regex_rejected():
    with pytest.raises(pydantic.ValidationError, match="invalid regex"):
        TernaryConfig(target_modules=[r"(.*\.q_proj"])


def test_router_regex_warns(caplog):
    with caplog.at_level("WARNING"):
        TernaryConfig(target_modules=[r".*gate"])
    assert "router" in caplog.text


def test_anchored_gate_proj_regex_does_not_warn(caplog):
    with caplog.at_level("WARNING"):
        TernaryConfig(target_modules=[r".*\.mlp\.(gate|up|down)_proj"])
    assert "router" not in caplog.text


# ----------------------------------------------------------------------- init


@pytest.mark.parametrize(
    "mode", ["absmean", "ternary_fit", "ternary_fit_calibrated", "svid"]
)
def test_init_modes_accepted(mode):
    scale = "absmean" if mode == "absmean" else "learnable"
    assert TernaryConfig(init=mode, weight_scale=scale).init == mode


def test_retired_init_mode_rejected():
    """`ptq_itf` was superseded by the two ternary_fit tiers."""
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(init="ptq_itf")


def test_init_calibration_accepted_for_the_calibrated_init():
    cfg = TernaryConfig(
        init="ternary_fit_calibrated",
        weight_scale="learnable",
        init_calibration={"num_sequences": 64, "dataset": "wikitext"},
    )

    assert cfg.init_calibration.num_sequences == 64
    assert cfg.init_calibration.sequence_len == 2048
    assert cfg.init_calibration.dataset == "wikitext"


@pytest.mark.parametrize("init", ["absmean", "ternary_fit", "svid"])
def test_init_calibration_rejected_for_the_other_init_modes(init):
    with pytest.raises(pydantic.ValidationError, match="ternary_fit_calibrated"):
        TernaryConfig(init=init, init_calibration={"num_sequences": 64})


@pytest.mark.parametrize("init", ["absmean", "ternary_fit"])
def test_unset_init_calibration_keeps_its_defaults(init):
    """Only an explicit block is a misconfiguration; the defaults ride along."""
    scale = "absmean" if init == "absmean" else "learnable"
    config = TernaryConfig(init=init, weight_scale=scale)
    assert config.init_calibration.num_sequences == 128


@pytest.mark.parametrize(
    "field,value",
    [("num_sequences", 0), ("num_sequences", -1), ("sequence_len", 0)],
)
def test_init_calibration_sizes_must_be_positive(field, value):
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(
            init="ternary_fit_calibrated",
            weight_scale="learnable",
            init_calibration={field: value},
        )


# -------------------------------------------------------------------- distill


def test_distill_teacher_requires_mode():
    with pytest.raises(pydantic.ValidationError, match="requires ternary.distill.mode"):
        TernaryConfig(distill={"teacher_model": "meta-llama/Llama-3.2-1B"})


@pytest.mark.parametrize("knob", [{"hidden_weight": 0.5}, {"attn_relation_layer": -1}])
def test_inprocess_only_distill_knobs_rejected_for_kd_plugin(knob):
    with pytest.raises(pydantic.ValidationError, match="mode: inprocess"):
        TernaryConfig(distill={"mode": "kd_plugin", **knob})


def test_distill_temperature_must_be_positive():
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(distill={"mode": "inprocess", "logits_temperature": 0.0})


def test_anchored_schedule_requires_a_distill_mode():
    with pytest.raises(pydantic.ValidationError, match="anchors a KD term"):
        TernaryConfig(distill={"schedule": "anchored"})


@pytest.mark.parametrize("mode", ["kd_plugin", "inprocess"])
def test_anchored_schedule_accepted_with_a_mode(mode):
    cfg = TernaryConfig(
        distill={"mode": mode, "schedule": "anchored", "anchor_start": 0.95}
    )

    assert cfg.distill.schedule == "anchored"
    assert cfg.distill.anchor_start == 0.95


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_anchor_start_must_be_a_strict_fraction(value):
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(
            distill={"mode": "inprocess", "schedule": "anchored", "anchor_start": value}
        )


def test_unknown_distill_schedule_rejected():
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(distill={"mode": "inprocess", "schedule": "cosine"})


def test_resolve_ternary_config_from_mapping():
    cfg = resolve_ternary_config({"ternary": {"lambda_warmup_steps": 42}})
    assert cfg.lambda_warmup_steps == 42


def test_resolve_ternary_config_defaults_when_absent():
    assert resolve_ternary_config({}) == TernaryConfig()
    assert resolve_ternary_config({"ternary": None}) == TernaryConfig()


def test_resolve_ternary_config_passes_through_model():
    cfg = TernaryConfig(lambda_schedule="none")
    assert resolve_ternary_config({"ternary": cfg}) is cfg


@pytest.fixture(name="merged_config_cls")
def fixture_merged_config_cls():
    """The class shape `merge_input_args` builds when the plugin is loaded."""
    from axolotl.utils.schemas.config import AxolotlInputConfig

    class MergedConfig(AxolotlInputConfig, TernaryArgs):
        pass

    return MergedConfig


MINIMAL_CONFIG = {
    "base_model": "meta-llama/Llama-3.2-1B",
    "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
}


def test_merged_config_exposes_ternary_block(merged_config_cls):
    cfg = merged_config_cls(**MINIMAL_CONFIG, ternary={"lambda_warmup_steps": 500})
    assert cfg.ternary.lambda_warmup_steps == 500


def test_merged_config_rejects_adapter(merged_config_cls):
    with pytest.raises(pydantic.ValidationError, match="full-finetune only"):
        merged_config_cls(**MINIMAL_CONFIG, adapter="lora", ternary={})


@pytest.mark.parametrize(
    "key,value", [("use_onebitllms", True), ("qat", {"activation_dtype": "int8"})]
)
def test_merged_config_rejects_conflicting_quantization(merged_config_cls, key, value):
    with pytest.raises(pydantic.ValidationError, match="cannot be combined"):
        merged_config_cls(**MINIMAL_CONFIG, **{key: value}, ternary={})


def test_resolve_ternary_config_from_dumped_axolotl_cfg(merged_config_cls):
    from axolotl.utils.dict import DictDefault

    cfg = merged_config_cls(**MINIMAL_CONFIG, ternary={"weight_scale": "learnable"})
    dumped = DictDefault(cfg.model_dump(exclude_none=True))
    assert resolve_ternary_config(dumped).weight_scale == "learnable"


# --------------------------------------------------------------- embedding dtype


def test_embedding_dtype_defaults_to_bf16():
    assert TernaryConfig().export.embedding_dtype == "bf16"


@pytest.mark.parametrize("fmt", ["gguf_tq2_0", "gguf_tq1_0", "i2_s"])
def test_int8_embeddings_accepted_for_gguf_formats(fmt):
    cfg = TernaryConfig(export={"formats": [fmt], "embedding_dtype": "int8"})

    assert cfg.export.embedding_dtype == "int8"


@pytest.mark.parametrize(
    "formats",
    [["hf_bitnet"], ["master_bf16"], ["master_bf16", "hf_bitnet"], ["mask_sign"]],
)
def test_int8_embeddings_rejected_without_a_gguf_format(formats):
    """Neither the master nor hf_bitnet can carry quantized embeddings."""
    with pytest.raises(pydantic.ValidationError, match="only representable in the"):
        TernaryConfig(export={"formats": formats, "embedding_dtype": "int8"})


def test_int8_embeddings_warn_about_the_exempt_formats(caplog):
    with caplog.at_level("WARNING"):
        cfg = TernaryConfig(
            export={
                "formats": ["master_bf16", "hf_bitnet", "gguf_tq2_0"],
                "embedding_dtype": "int8",
            }
        )

    assert cfg.export.embedding_dtype == "int8"
    assert "keep full-precision embeddings" in caplog.text
    assert "'hf_bitnet'" in caplog.text and "'master_bf16'" in caplog.text


def test_bf16_embeddings_never_warn(caplog):
    with caplog.at_level("WARNING"):
        TernaryConfig(export={"formats": ["master_bf16", "hf_bitnet"]})

    assert "embeddings" not in caplog.text


def test_unknown_embedding_dtype_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        TernaryConfig(export={"formats": ["gguf_tq2_0"], "embedding_dtype": "int4"})
