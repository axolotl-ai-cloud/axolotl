"""``moe_grouped_mode`` resolution: the routed-expert format read from tiny synthetic checkpoints
(safetensors headers + config), the auto / explicit decision table, and the startup check."""

import logging

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("triton")  # the scattermoe_lora package imports its Triton kernels

from transformers import Qwen3MoeConfig  # noqa: E402

from axolotl.integrations.kernels import grouped_mode as gm  # noqa: E402
from axolotl.integrations.kernels.args import KernelsArgs  # noqa: E402
from axolotl.integrations.kernels.libs.scattermoe_lora import (  # noqa: E402
    nvfp4_moe_loading,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import (  # noqa: E402
    RUNTIME,
)
from axolotl.utils.dict import DictDefault  # noqa: E402

PROJS = ("gate_proj", "up_proj", "down_proj")


def _per_expert(prefix, fmt):
    if fmt == "nvfp4":
        return {
            prefix + "weight": torch.zeros(16, 8, dtype=torch.uint8),
            prefix + "weight_scale": torch.zeros(16, 1, dtype=torch.float8_e4m3fn),
            prefix + "weight_scale_2": torch.ones(()),
            prefix + "input_scale": torch.ones(()),
        }
    if fmt == "block_fp8":
        return {
            prefix + "weight": torch.zeros(16, 16, dtype=torch.float8_e4m3fn),
            prefix + "weight_scale_inv": torch.ones(1, 1),
        }
    if fmt == "fp8_per_tensor":
        return {
            prefix + "weight": torch.zeros(16, 16, dtype=torch.float8_e4m3fn),
            prefix + "weight_scale_inv": torch.ones(()),
        }
    if fmt == "packed_fp4":
        return {
            prefix + "weight": torch.zeros(16, 8, dtype=torch.int8),
            prefix + "scale": torch.zeros(16, 1, dtype=torch.uint8),
        }
    assert fmt == "bf16"
    return {prefix + "weight": torch.zeros(16, 16, dtype=torch.bfloat16)}


def _fused(prefix, fmt):
    if fmt == "block_fp8":
        return {
            prefix + "gate_up_proj": torch.zeros(2, 32, 16, dtype=torch.float8_e4m3fn),
            prefix + "gate_up_proj_scale_inv": torch.ones(2, 1, 1),
            prefix + "down_proj": torch.zeros(2, 16, 16, dtype=torch.float8_e4m3fn),
            prefix + "down_proj_scale_inv": torch.ones(2, 1, 1),
        }
    if fmt == "mxfp4":  # gpt-oss
        return {
            prefix + "gate_up_proj_blocks": torch.zeros(
                2, 32, 1, 16, dtype=torch.uint8
            ),
            prefix + "gate_up_proj_scales": torch.zeros(2, 32, 1, dtype=torch.uint8),
        }
    assert fmt == "bf16"
    return {
        prefix + "gate_up_proj": torch.zeros(2, 32, 16, dtype=torch.bfloat16),
        prefix + "down_proj": torch.zeros(2, 16, 16, dtype=torch.bfloat16),
    }


def _checkpoint(tmp_path, layers, fused=False, num_layers=None, **config):
    """Config + one safetensors shard whose layer ``i`` routed experts are in ``layers[i]``
    (None = a dense layer)."""
    tensors = {"model.embed_tokens.weight": torch.zeros(32, 16, dtype=torch.bfloat16)}
    for i, fmt in enumerate(layers):
        if fmt is None:
            tensors[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(16, 16)
            continue
        prefix = f"model.layers.{i}.mlp.experts."
        if fused:
            tensors.update(_fused(prefix, fmt))
        else:
            for e in range(2):
                for proj in PROJS:
                    tensors.update(_per_expert(f"{prefix}{e}.{proj}.", fmt))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    Qwen3MoeConfig(
        num_hidden_layers=num_layers or len(layers),
        num_experts=2,
        hidden_size=16,
        intermediate_size=16,
        moe_intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        **config,
    ).save_pretrained(tmp_path)
    return DictDefault(base_model=str(tmp_path), use_scattermoe=True)


@pytest.mark.parametrize(
    "layers,fused,kind",
    [
        (["nvfp4", "nvfp4"], False, gm.NVFP4),
        (["block_fp8", "block_fp8"], False, gm.BLOCK_FP8),
        (["block_fp8", "block_fp8"], True, gm.BLOCK_FP8),
        ([None, "bf16"], False, gm.BF16),
        (["bf16"], True, gm.BF16),
        (["packed_fp4", "packed_fp4"], False, gm.PACKED_FP4),
        (["fp8_per_tensor"], False, gm.OTHER),
        (["mxfp4"], True, gm.OTHER),
        ([None, None], False, gm.NO_EXPERTS),
    ],
)
def test_detects_expert_format_from_headers(tmp_path, layers, fused, kind):
    assert gm.detect_expert_format(_checkpoint(tmp_path, layers, fused)).kind == kind


def test_mixed_layers_are_named(tmp_path):
    cfg = _checkpoint(tmp_path, ["nvfp4", "nvfp4", "bf16", "nvfp4"])
    fmt = gm.detect_expert_format(cfg)
    assert fmt.kind == gm.MIXED
    assert "NVFP4 in layers 0-1,3" in fmt.reason and "(bf16) in layers 2" in fmt.reason


def test_layers_past_num_hidden_layers_are_ignored(tmp_path):
    # a multi-token-prediction layer the model never loads can differ from the decoder's experts
    cfg = _checkpoint(tmp_path, ["nvfp4", "nvfp4", "bf16"], num_layers=2)
    assert gm.detect_expert_format(cfg).kind == gm.NVFP4


def test_mtp_layers_are_ignored():
    meta = {
        "model.layers.0.mlp.experts.0.w1.weight": ("U8", (16, 8)),
        "model.layers.0.mlp.experts.0.w1.weight_scale": ("F8_E4M3", (16, 1)),
        "mtp.layers.0.mlp.experts.0.w1.weight": ("BF16", (16, 16)),
    }
    assert gm._expert_layer_formats(meta, num_layers=1) == {0: {gm.NVFP4}}


@pytest.mark.parametrize(
    "kwargs,kind",
    [
        ({}, gm.BLOCK_FP8),
        ({"weight_block_size": [128, 128]}, gm.BLOCK_FP8),
        ({"modules_to_not_convert": ["mlp.experts"]}, gm.BF16),
    ],
)
def test_on_the_fly_fp8_quantization(tmp_path, kwargs, kind):
    cfg = _checkpoint(tmp_path, ["bf16"])
    cfg.model_quantization_config = "FineGrainedFP8Config"
    cfg.model_quantization_config_kwargs = kwargs
    assert gm.detect_expert_format(cfg).kind == kind


def test_bnb_expert_quantization_is_other(tmp_path):
    cfg = _checkpoint(tmp_path, ["bf16"])
    cfg.quantize_moe_experts = True
    assert gm.detect_expert_format(cfg).kind == gm.OTHER


@pytest.mark.parametrize(
    "quantization_config,kind",
    [
        (None, gm.BF16),
        ({"quant_method": "fp8", "weight_block_size": [128, 128]}, gm.BLOCK_FP8),
        ({"quant_method": "fp8"}, gm.OTHER),
        (
            {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "expert_dtype": "fp4",
            },
            gm.PACKED_FP4,
        ),
        (
            {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "moe_quant_algo": "NVFP4",
            },
            gm.NVFP4,
        ),
        (
            {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["model.layers.0.mlp.experts"],
            },
            gm.BF16,
        ),
        ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, gm.NVFP4),
        (
            {
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "layers.1.experts": {"quant_algo": "NVFP4"},
                    "layers.1.attention.wq": {"quant_algo": "FP8_PB_WO"},
                },
            },
            gm.NVFP4,
        ),
        ({"quant_method": "bitsandbytes"}, gm.OTHER),
    ],
)
def test_falls_back_to_quantization_config(
    tmp_path, monkeypatch, quantization_config, kind
):
    extra = {"quantization_config": quantization_config} if quantization_config else {}
    cfg = _checkpoint(tmp_path, ["bf16"], **extra)

    def unreadable(repo_id):
        raise OSError("offline")

    monkeypatch.setattr(nvfp4_moe_loading, "_safetensors_metadata", unreadable)
    fmt = gm.detect_expert_format(cfg)
    assert fmt.kind == kind and "headers unreadable" in fmt.reason


def _fmt(kind):
    return gm.ExpertFormat(kind, f"test {kind}")


@pytest.mark.parametrize(
    "kind,requested,requant,expected",
    [
        (gm.NVFP4, None, False, "nvfp4"),
        (gm.NVFP4, "auto", False, "nvfp4"),
        (gm.NVFP4, "nvfp4", False, "nvfp4"),
        (gm.BLOCK_FP8, "auto", False, "fp8_block"),
        (gm.BLOCK_FP8, "fp8_block", False, "fp8_block"),
        (gm.BLOCK_FP8, "nvfp4", True, "nvfp4"),
        (gm.BLOCK_FP8, "mxfp8", True, "mxfp8"),
        (gm.BF16, "auto", False, None),
        (gm.BF16, "mxfp8", True, "mxfp8"),
        (gm.OTHER, "auto", False, None),
        (gm.NO_EXPERTS, "auto", False, None),
        (gm.NO_EXPERTS, "nvfp4", False, None),
        (gm.NVFP4, "none", False, None),
        (gm.PACKED_FP4, "none", False, None),
        (gm.MIXED, "none", False, None),
    ],
)
def test_decision_table(kind, requested, requant, expected):
    choice = gm.choose_grouped_mode(requested, _fmt(kind), allow_requant=requant)
    assert choice.mode == expected
    assert choice.explicit == (requested not in (None, "auto"))
    assert bool(choice.requant) == ((kind, expected) in gm._REQUANT)


@pytest.mark.parametrize(
    "kind,requested,match",
    [
        (gm.BLOCK_FP8, "nvfp4", "moe_grouped_requant: true"),
        (gm.BLOCK_FP8, "mxfp8", "moe_grouped_requant: true"),
        (gm.BF16, "mxfp8", "moe_grouped_requant: true"),
        (gm.NVFP4, "fp8_block", "can't run these experts"),
        (gm.NVFP4, "mxfp8", "can't run these experts"),
        (gm.BF16, "nvfp4", "can't run these experts"),
        (gm.BF16, "fp8_block", "can't run these experts"),
        (gm.OTHER, "fp8_block", "can't run these experts"),
        (gm.PACKED_FP4, None, "packed-FP4"),
        (gm.PACKED_FP4, "nvfp4", "packed-FP4"),
        (gm.MIXED, None, "differ"),
        (gm.MIXED, "fp8_block", "differ"),
    ],
)
def test_decision_table_refusals(kind, requested, match):
    with pytest.raises(ValueError, match=match):
        gm.choose_grouped_mode(requested, gm.ExpertFormat(kind, "differ"), False)


def test_refusal_names_the_valid_choices():
    with pytest.raises(ValueError) as err:
        gm.choose_grouped_mode("nvfp4", _fmt(gm.BLOCK_FP8), False)
    assert "auto, none, fp8_block, nvfp4 (with moe_grouped_requant: true)" in str(
        err.value
    )


@pytest.fixture
def info_logs(caplog):
    with caplog.at_level(logging.INFO, logger="axolotl"):
        yield caplog


def _resolve_line(caplog):
    lines = [
        r.getMessage() for r in caplog.records if "moe_grouped_mode=" in r.getMessage()
    ]
    assert len(lines) == 1
    return lines[0]


def test_configure_auto_sets_cfg_and_runtime(tmp_path, info_logs, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = _checkpoint(tmp_path, ["block_fp8", "block_fp8"])
    choice = gm.configure_moe_grouped_mode(cfg)
    assert choice.mode == cfg.moe_grouped_mode == RUNTIME.grouped_mode == "fp8_block"
    line = _resolve_line(info_logs)
    assert line.startswith("moe_grouped_mode=fp8_block (auto: ")
    assert "gpu=none | backend=- | requant=no" in line


def test_configure_maps_the_legacy_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = _checkpoint(tmp_path, ["nvfp4"])
    cfg.dsv4_fp4_grouped_mode = "nvfp4"
    assert gm.configure_moe_grouped_mode(cfg).explicit
    assert cfg.moe_grouped_mode == "nvfp4" and cfg.dsv4_fp4_grouped_mode is None


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))

    def set_backend(backend):
        monkeypatch.setattr(gm, "grouped_backend", lambda mode: backend)

    return set_backend


def test_configure_logs_the_backend(tmp_path, info_logs, gpu):
    gpu("deepgemm_fp8")
    gm.configure_moe_grouped_mode(_checkpoint(tmp_path, ["block_fp8"]))
    assert "gpu=sm90 | backend=deepgemm_fp8 | requant=no" in _resolve_line(info_logs)


def test_auto_falls_back_without_a_backend(tmp_path, info_logs, gpu):
    gpu(None)
    cfg = _checkpoint(tmp_path, ["nvfp4"])
    assert gm.configure_moe_grouped_mode(cfg).mode is None
    assert cfg.moe_grouped_mode is None and RUNTIME.grouped_mode is None
    assert any(
        "no nvfp4 grouped backend on sm90" in r.getMessage() for r in info_logs.records
    )
    assert _resolve_line(info_logs).startswith("moe_grouped_mode=none (auto: ")


def test_explicit_mode_without_a_backend_fails_fast(tmp_path, gpu):
    gpu(None)
    cfg = _checkpoint(tmp_path, ["nvfp4"])
    cfg.moe_grouped_mode = "nvfp4"
    with pytest.raises(RuntimeError, match="no grouped backend on this GPU"):
        gm.configure_moe_grouped_mode(cfg)


def test_explicit_requant_logs_its_cost(tmp_path, info_logs, gpu):
    gpu("marlin")
    cfg = _checkpoint(tmp_path, ["block_fp8"])
    cfg.moe_grouped_mode = "nvfp4"
    cfg.moe_grouped_requant = True
    gm.configure_moe_grouped_mode(cfg)
    assert any(
        "~17x the fp8_block weight error" in r.getMessage() for r in info_logs.records
    )
    assert "requant=yes" in _resolve_line(info_logs)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("none", "none"),
        (False, "none"),
        ("FP8_Block", "fp8_block"),
    ],
)
def test_args_normalize_grouped_mode(value, expected):
    assert (
        KernelsArgs.model_validate({"moe_grouped_mode": value}).moe_grouped_mode
        == expected
    )


def test_args_accept_grouped_requant():
    assert KernelsArgs.model_validate({"moe_grouped_requant": True}).moe_grouped_requant


@pytest.mark.parametrize(
    "extra,ok",
    [
        ({"moe_rank0_requant": False}, False),
        ({"moe_rank0_requant": False, "expert_parallel_size": 2}, True),
        ({}, True),
        ({"moe_rank0_requant": True}, True),
    ],
)
def test_mxfp8_cpu_ram_efficient_needs_rank0_requant(extra, ok):
    data = {
        "moe_grouped_mode": "mxfp8",
        "fsdp_config": {"cpu_ram_efficient_loading": True},
        **extra,
    }
    if ok:
        KernelsArgs.model_validate(data)
    else:
        with pytest.raises(ValueError, match="moe_rank0_requant: false"):
            KernelsArgs.model_validate(data)
