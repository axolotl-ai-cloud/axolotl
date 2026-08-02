import pytest

from axolotl.train import _bf16_default_dtype_safe
from axolotl.utils.dict import DictDefault


@pytest.mark.parametrize(
    "cfg",
    [
        {},
        {"bf16": True},
        {"fsdp_version": 1, "fsdp_config": {"activation_checkpointing": True}},
        {"fsdp_version": 2},
        {"fsdp_version": "2", "fsdp_config": {}},
        {"optimizer": "adamw_torch"},
    ],
)
def test_bf16_default_dtype_safe(cfg):
    assert _bf16_default_dtype_safe(DictDefault(cfg)) is True


@pytest.mark.parametrize(
    "cfg",
    [
        {"fsdp_version": 2, "fsdp_config": {"activation_checkpointing": True}},
        {"fsdp_version": "2", "fsdp_config": {"sync_module_state": True}},
        {
            "fsdp_version": 2,
            "fsdp_config": {"activation_checkpointing": True},
            "optimizer": "adamw_torch",
        },
        {"fsdp_version": 2, "fsdp": "full_shard"},
        {"optimizer": "ao_adamw_4bit"},
        {"optimizer": "ao_adamw_8bit"},
        {"optimizer": "ao_adamw_fp8"},
        {"optimizer": "sinkgd"},
    ],
)
def test_bf16_default_dtype_unsafe(cfg):
    assert _bf16_default_dtype_safe(DictDefault(cfg)) is False
