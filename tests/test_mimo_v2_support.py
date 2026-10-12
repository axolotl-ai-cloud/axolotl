"""MiMo-V2.6 (`mimo_v2`): the FP8/MXFP4 published layout loads into the native MoE modules."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from transformers.conversion_mapping import register_checkpoint_conversion_mapping

from axolotl.model_support import get_model_support
from axolotl.model_support.mimo_v2 import _redirect_remote_code, _weight_conversions
from axolotl.model_support.mimo_v2.configuration_mimo_v2 import MiMoV2Config
from axolotl.utils.dict import DictDefault

HUB_CONFIG = {
    "architectures": ["MiMoV2ForCausalLM"],
    "auto_map": {
        "AutoConfig": "configuration_mimo_v2.MiMoV2Config",
        "AutoModelForCausalLM": "modeling_mimo_v2.MiMoV2ForCausalLM",
    },
    "model_type": "mimo_v2",
    "vocab_size": 64,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "swa_num_key_value_heads": 4,
    "head_dim": 192,
    "v_head_dim": 128,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "hybrid_layer_pattern": [0, 1, 0],
    "moe_layer_freq": [0, 1, 1],
    "layernorm_epsilon": 1e-6,
    "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.334},
    "swa_rope_theta": 1e4,
    "sliding_window": 4,
    "attention_projection_layout": "fused_qkv",
    "add_swa_attention_sink_bias": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
}
FP4_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def fp8_block(weight):
    """FP8 128x128 blocks, scale grid padded to a multiple of 4 rows like the hub shards."""
    rows, cols = weight.shape
    blocks = weight.reshape(rows // 128, 128, cols // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)) / 448
    quantized = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scale[:, None, :, None]).reshape(rows, cols)
    padded = torch.cat([scale, torch.full((-len(scale) % 4, scale.shape[1]), 7.0)])
    return quantized.reshape(rows, cols), padded, dequantized


def mxfp4(rows, cols):
    """uint8-packed e2m1 pairs (low nibble first) with an e8m0 exponent per 32 values."""
    codes = torch.randint(0, 16, (rows, cols))
    exponents = torch.randint(124, 130, (rows, cols // 32), dtype=torch.uint8)
    values = torch.tensor(FP4_VALUES)[codes] * (
        exponents.float() - 127
    ).exp2().repeat_interleave(32, 1)
    packed = (codes[:, 0::2] | codes[:, 1::2] << 4).to(torch.uint8)
    return packed, exponents, values


def published_checkpoint(config):
    """A V2.6-format checkpoint plus the bf16 tensors the native model should load."""
    torch.manual_seed(0)
    hidden, ckpt, expected = config.hidden_size, {}, {}

    def fp8(name, shape):
        quantized, scale, dequantized = fp8_block(torch.randn(shape) * 0.02)
        ckpt[f"{name}.weight"], ckpt[f"{name}.weight_scale_inv"] = quantized, scale
        return dequantized

    def bf16(name, tensor):
        ckpt[name] = expected[name] = tensor.to(torch.bfloat16)

    bf16("model.embed_tokens.weight", torch.randn(config.vocab_size, hidden))
    bf16("lm_head.weight", torch.randn(config.vocab_size, hidden))
    bf16("model.norm.weight", torch.ones(hidden))
    for i, layer_type in enumerate(config.layer_types):
        p = f"model.layers.{i}"
        kv_heads = config.num_key_value_heads * (
            2 if layer_type != "full_attention" else 1
        )
        q = config.num_attention_heads * config.head_dim
        k, v = kv_heads * config.head_dim, kv_heads * config.v_head_dim
        qkv = fp8(f"{p}.self_attn.qkv_proj", (q + k + v, hidden))
        for name, part in zip("qkv", qkv.split([q, k, v]), strict=True):
            expected[f"{p}.self_attn.{name}_proj.weight"] = part
        bf16(
            f"{p}.self_attn.o_proj.weight",
            torch.randn(hidden, config.num_attention_heads * config.v_head_dim),
        )
        if layer_type == "sliding_attention":
            ckpt[f"{p}.self_attn.attention_sink_bias"] = torch.randn(
                config.num_attention_heads
            ).bfloat16()
            expected[f"{p}.self_attn.sinks"] = ckpt[
                f"{p}.self_attn.attention_sink_bias"
            ]
        bf16(f"{p}.input_layernorm.weight", torch.ones(hidden))
        bf16(f"{p}.post_attention_layernorm.weight", torch.ones(hidden))
        if config.mlp_layer_types[i] == "dense":
            for name, shape in (
                ("gate_proj", (512, hidden)),
                ("up_proj", (512, hidden)),
                ("down_proj", (hidden, 512)),
            ):
                expected[f"{p}.mlp.{name}.weight"] = fp8(f"{p}.mlp.{name}", shape)
            continue
        bf16(f"{p}.mlp.gate.weight", torch.randn(4, hidden))
        ckpt[f"{p}.mlp.gate.e_score_correction_bias"] = torch.zeros(4)
        gate_up, down = [], []
        for e in range(config.n_routed_experts):
            for name, shape in (
                ("gate_proj", (128, hidden)),
                ("up_proj", (128, hidden)),
                ("down_proj", (hidden, 128)),
            ):
                packed, scale, values = mxfp4(*shape)
                ckpt[f"{p}.mlp.experts.{e}.{name}.weight"] = packed
                ckpt[f"{p}.mlp.experts.{e}.{name}.weight_scale"] = scale
                (down if name == "down_proj" else gate_up).append(values)
        expected[f"{p}.mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat(gate_up[e * 2 : e * 2 + 2]) for e in range(4)]
        )
        expected[f"{p}.mlp.experts.down_proj"] = torch.stack(down)
    ckpt["model.mtp.layers.0.enorm.weight"] = torch.ones(hidden).bfloat16()
    ckpt["visual.patch_embed.proj.weight"] = torch.ones(8, 3).bfloat16()
    return ckpt, expected


@pytest.fixture(name="checkpoint_dir")
def checkpoint_dir_fixture(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(HUB_CONFIG))
    # the redirect serves these from axolotl; transformers only needs them on disk
    (tmp_path / "configuration_mimo_v2.py").write_text("")
    (tmp_path / "modeling_mimo_v2.py").write_text("")
    return tmp_path


def test_descriptor_is_discovered_by_registry():
    assert type(get_model_support("mimo_v2")).__name__ == "MiMoV2Support"


def test_hub_config_maps_onto_native_fields():
    config = MiMoV2Config(**HUB_CONFIG)
    assert config.layer_types == [
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert config.mlp_layer_types == ["dense", "sparse", "sparse"]
    assert config.rms_norm_eps == 1e-6
    assert config.rope_parameters["full_attention"]["rope_theta"] == 1e7
    assert config.rope_parameters["sliding_attention"]["rope_theta"] == 1e4
    assert getattr(config, "quantization_config", None) is None


def test_published_checkpoint_loads_dequantized_and_trains(checkpoint_dir):
    config = MiMoV2Config(**HUB_CONFIG)
    ckpt, expected = published_checkpoint(config)
    save_file(ckpt, checkpoint_dir / "model.safetensors")
    _redirect_remote_code(None)
    for key, entries in _weight_conversions().items():
        register_checkpoint_conversion_mapping(key, entries, overwrite=True)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    assert type(model).__module__ == "axolotl.model_support.mimo_v2.modeling_mimo_v2"
    state = model.state_dict()
    # noaux_tc biases collapse in bf16; the model keeps them fp32
    assert (
        state["model.layers.1.mlp.gate.e_score_correction_bias"].dtype == torch.float32
    )
    for name, tensor in expected.items():
        torch.testing.assert_close(state[name], tensor.bfloat16(), msg=name)
    model.save_pretrained(checkpoint_dir / "saved")
    reloaded = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir / "saved", trust_remote_code=True, dtype=torch.bfloat16
    ).state_dict()
    assert all(torch.equal(tensor, reloaded[name]) for name, tensor in state.items())

    model.train()
    out = model(input_ids=torch.randint(0, 64, (1, 8)), labels=torch.ones(1, 8).long())
    out.loss.backward()
    assert model.model.layers[1].mlp.experts.gate_up_proj.grad is not None


def test_published_layout_forces_legacy_merge(checkpoint_dir, tmp_path):
    support = get_model_support("mimo_v2")
    published = DictDefault(base_model_config=str(checkpoint_dir))
    assert support.resolve_lora_merge_method(published, None) == "legacy"
    with pytest.raises(ValueError, match="merge_method: legacy"):
        support.resolve_lora_merge_method(published, "memory_efficient")

    native = tmp_path / "native"
    native.mkdir()
    config = {k: v for k, v in HUB_CONFIG.items() if k != "quantization_config"}
    (native / "config.json").write_text(json.dumps(config))
    saved = DictDefault(base_model_config=str(native))
    assert support.resolve_lora_merge_method(saved, None) == "memory_efficient"
    assert support.resolve_lora_merge_method(saved, "legacy") == "legacy"


def test_save_under_fsdp_generated_subclass_keeps_auto_map(tmp_path, monkeypatch):
    from axolotl.model_support.mimo_v2.modeling_mimo_v2 import MiMoV2ForCausalLM

    monkeypatch.setattr(MiMoV2ForCausalLM, "_auto_class", "AutoModelForCausalLM")
    model = MiMoV2ForCausalLM(MiMoV2Config(**HUB_CONFIG))
    model.config.auto_map = dict(HUB_CONFIG["auto_map"])
    # fully_shard swaps in a generated subclass defined in torch's FSDP package
    model.__class__ = type(
        "FSDPMiMoV2ForCausalLM",
        (MiMoV2ForCausalLM,),
        {"__module__": "torch.distributed.fsdp._fully_shard._fully_shard"},
    )

    model.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["auto_map"]["AutoModelForCausalLM"] == (
        "modeling_mimo_v2.MiMoV2ForCausalLM"
    )
    assert (tmp_path / "modeling_mimo_v2.py").exists()
    assert not list(tmp_path.glob("_fsdp*.py"))
    assert MiMoV2ForCausalLM._auto_class == "AutoModelForCausalLM"
