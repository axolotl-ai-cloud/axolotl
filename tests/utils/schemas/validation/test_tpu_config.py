"""check_tpu_config, run on CPU with the ``mock_tpu`` fixture."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.validation import lora_kernels_auto_enable

from tests.conftest import capture_axolotl_warnings


@pytest.fixture(name="tpu_cfg")
def fixture_tpu_cfg(min_base_cfg):
    return DictDefault({**min_base_cfg, "attn_implementation": "sdpa"})


def _with(cfg, **overrides):
    # DictDefault's `|` keeps the left-hand value on key clashes.
    return DictDefault({**cfg, **overrides})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"adapter": "qlora"}, "qlora"),
        ({"adapter": "lora", "load_in_4bit": True}, "load_in_4bit"),
        ({"adapter": "lora", "load_in_8bit": True}, "load_in_8bit"),
        ({"optimizer": "adamw_bnb_8bit"}, "adamw_bnb_8bit"),
        ({"attn_implementation": "flash_attention_2"}, "flash_attention_2"),
        ({"attn_implementation": "flex_attention"}, "flex_attention"),
        ({"attn_implementation": None, "flash_attention": True}, "flash_attention"),
        ({"attn_implementation": None, "xformers_attention": True}, "xformers"),
        ({"deepspeed": "deepspeed_configs/zero2.json"}, "deepspeed"),
        ({"fsdp_config": {"fsdp_version": 2}}, "fsdp"),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size"),
        ({"context_parallel_size": 2}, "context_parallel_size"),
        ({"expert_parallel_size": 2}, "expert_parallel_size"),
        ({"rl": "dpo"}, "rl"),
        ({"sample_packing": True}, "sample_packing"),
        ({"batch_flattening": True}, "batch_flattening"),
        ({"plugins": ["axolotl.integrations.liger.LigerPlugin"]}, "liger"),
        ({"plugins": ["axolotl.integrations.liger:LigerPlugin"]}, "liger"),
        (
            {
                "plugins": [
                    "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"
                ]
            },
            "cut_cross_entropy",
        ),
        ({"adapter": "lora", "lora_mlp_kernel": True}, "lora_mlp_kernel"),
        ({"adapter": "lora", "lora_qkv_kernel": True}, "lora_qkv_kernel"),
        ({"adapter": "lora", "lora_o_kernel": True}, "lora_o_kernel"),
        ({"adapter": "lora", "lora_embedding_kernel": True}, "lora_embedding_kernel"),
        ({"activation_offloading": True}, "offload"),
        (
            {"activation_checkpointing": {"enabled": True, "offload": {"all": "cpu"}}},
            "offload",
        ),
        ({"gradient_checkpointing": "offload"}, "offload"),
        ({"torch_compile": True}, "torch_compile"),
        ({"tf32": True}, "tf32"),
        ({"fsdp_config": {}}, "fsdp_config"),
        ({"sequence_parallel_degree": 2}, "context_parallel_size"),
        ({"gpu_memory_limit": 20}, "gpu_memory_limit"),
        ({"max_memory": {"0": "20GiB"}}, "max_memory"),
    ],
)
def test_unsupported_options_rejected(mock_tpu, tpu_cfg, overrides, match):
    with pytest.raises(NotImplementedError, match=match):
        validate_config(_with(tpu_cfg, **overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"attn_implementation": "eager"},
        {"attn_implementation": "sdpa"},
        {"adapter": "lora"},
        {"fp16": True, "bf16": False},
        {"tf32": "auto"},
        {"gradient_checkpointing": True},
        {"activation_checkpointing": {"enabled": True}},
    ],
)
def test_supported_options_accepted(mock_tpu, tpu_cfg, overrides):
    validate_config(_with(tpu_cfg, **overrides))


def test_static_shape_defaults(mock_tpu, tpu_cfg):
    cfg = validate_config(tpu_cfg)
    assert cfg.pad_to_sequence_len is True
    assert cfg.dataloader_pin_memory is False


def test_pad_to_sequence_len_false_overridden_with_warning(mock_tpu, tpu_cfg, caplog):
    with capture_axolotl_warnings(caplog):
        cfg = validate_config(_with(tpu_cfg, pad_to_sequence_len=False))
    assert cfg.pad_to_sequence_len is True
    assert "pad_to_sequence_len: false" in caplog.text


def test_lora_kernels_not_auto_enabled(mock_tpu, tpu_cfg):
    assert lora_kernels_auto_enable(DictDefault(adapter="lora").get) is False
    cfg = validate_config(_with(tpu_cfg, adapter="lora"))
    for flag in ("lora_mlp_kernel", "lora_qkv_kernel", "lora_o_kernel"):
        assert not cfg[flag]


def test_inert_off_tpu(tpu_cfg):
    cfg = validate_config(_with(tpu_cfg, torch_compile=True, pad_to_sequence_len=False))
    assert cfg.torch_compile is True
    assert cfg.pad_to_sequence_len is False
