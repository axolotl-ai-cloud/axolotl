"""Tests for ple_cpu_offload config validation and the sdpa bf16 packing warning gate."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


@pytest.fixture()
def gpu_caps():
    return {
        "compute_capability": "sm_100",
        "bf16": True,
        "tf32": False,
        "n_gpu": 1,
        "n_node": 1,
    }


@pytest.fixture()
def env_caps():
    return {"torch_version": "2.12.1"}


class TestPleCpuOffloadValidation:
    """ple_cpu_offload only works where accelerate skips its own device placement."""

    def test_valid_with_4bit(self, min_base_cfg, gpu_caps, env_caps):
        cfg = (
            DictDefault(ple_cpu_offload=True, adapter="qlora", load_in_4bit=True)
            | min_base_cfg
        )
        validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)

    def test_valid_with_8bit(self, min_base_cfg, gpu_caps, env_caps):
        cfg = (
            DictDefault(ple_cpu_offload=True, adapter="lora", load_in_8bit=True)
            | min_base_cfg
        )
        validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)

    def test_requires_quantization(self, min_base_cfg, gpu_caps, env_caps):
        """Without bnb quantization accelerate always moves the model, so the flag would
        silently do nothing."""
        cfg = DictDefault(ple_cpu_offload=True, adapter="lora") | min_base_cfg
        with pytest.raises(ValueError, match="requires load_in_4bit or load_in_8bit"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)

    def test_rejects_fsdp(self, min_base_cfg, gpu_caps, env_caps):
        cfg = (
            DictDefault(
                ple_cpu_offload=True,
                adapter="qlora",
                load_in_4bit=True,
                fsdp_config={"fsdp_version": 2},
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="not compatible with FSDP"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)

    def test_rejects_deepspeed(self, min_base_cfg, gpu_caps, env_caps):
        cfg = (
            DictDefault(
                ple_cpu_offload=True,
                adapter="qlora",
                load_in_4bit=True,
                deepspeed="deepspeed_configs/zero3.json",
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="not compatible with DeepSpeed"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)

    def test_off_by_default(self, min_base_cfg, gpu_caps, env_caps):
        cfg = DictDefault(adapter="lora") | min_base_cfg
        parsed = validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
        assert parsed.ple_cpu_offload is False


class TestSdpaBf16PackingWarning:
    """The torch issue this warns about is pre-Hopper, so sm_90+ must stay quiet."""

    @pytest.mark.parametrize("compute_capability", ["sm_80", "sm_86", "sm_89"])
    def test_warns_before_hopper(
        self, min_base_cfg, gpu_caps, env_caps, caplog, compute_capability
    ):
        gpu_caps["compute_capability"] = compute_capability
        cfg = (
            DictDefault(sample_packing=True, attn_implementation="sdpa", bf16=True)
            | min_base_cfg
        )
        with caplog.at_level("WARNING"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
        assert "0.0 loss" in caplog.text

    @pytest.mark.parametrize(
        "compute_capability", ["sm_90", "sm_100", "sm_103", "sm_120"]
    )
    def test_quiet_on_hopper_and_newer(
        self, min_base_cfg, gpu_caps, env_caps, caplog, compute_capability
    ):
        gpu_caps["compute_capability"] = compute_capability
        cfg = (
            DictDefault(sample_packing=True, attn_implementation="sdpa", bf16=True)
            | min_base_cfg
        )
        with caplog.at_level("WARNING"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
        assert "0.0 loss" not in caplog.text
