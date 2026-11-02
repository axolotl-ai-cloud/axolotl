"""Config validation for ``lora_merge_aware`` and ``trl.vllm_merge_check_interval``."""

import pytest
from pydantic import ValidationError

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def _cfg(**overrides):
    cfg = {
        "base_model": "dummy_model",
        "adapter": "lora",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_target_linear": True,
        "lora_merge_aware": "bf16",
        "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-5,
    }
    cfg.update(overrides)
    return DictDefault(cfg)


def test_lora_merge_aware_accepted_and_keeps_lora_kernels_off():
    cfg = validate_config(_cfg())
    assert cfg.lora_merge_aware == "bf16"
    assert not cfg.lora_mlp_kernel
    assert not cfg.lora_qkv_kernel
    assert not cfg.lora_o_kernel


def test_lora_merge_aware_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        validate_config(_cfg(lora_merge_aware="fp8"))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"adapter": "qlora", "load_in_4bit": True}, "adapter: qlora"),
        ({"load_in_8bit": True}, "load_in_8bit"),
        ({"peft_use_dora": True}, "peft_use_dora"),
        ({"lora_mlp_kernel": True}, "fused kernels"),
        (
            {
                "fsdp_version": 2,
                "fsdp_config": {"reshard_after_forward": True},
                "lora_fp32_gradients": False,
            },
            "lora_fp32_gradients: false",
        ),
        (
            {"deepspeed": {"zero_optimization": {"stage": 3}}},
            "DeepSpeed ZeRO-3 .*never gathers them",
        ),
        ({"rl": "grpo", "trl": {"vllm_lora_sync": True}}, "vllm_lora_sync"),
    ],
)
def test_lora_merge_aware_rejects_unsupported_combinations(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_config(_cfg(**overrides))


def test_lora_merge_aware_under_fsdp2_gathers_fp32_factors():
    cfg = validate_config(
        _cfg(fsdp_version=2, fsdp_config={"reshard_after_forward": True})
    )
    assert cfg.lora_merge_aware == "bf16"
    assert cfg.lora_fp32_gradients is True


def test_lora_merge_aware_allows_deepspeed_zero2():
    cfg = validate_config(_cfg(deepspeed={"zero_optimization": {"stage": 2}}))
    assert cfg.lora_merge_aware == "bf16"


def test_vllm_merge_check_interval():
    cfg = validate_config(
        _cfg(lora_merge_aware=None, rl="grpo", trl={"vllm_merge_check_interval": 0})
    )
    assert cfg.trl.vllm_merge_check_interval == 0
    with pytest.raises(ValidationError):
        validate_config(
            _cfg(
                lora_merge_aware=None, rl="grpo", trl={"vllm_merge_check_interval": -1}
            )
        )
