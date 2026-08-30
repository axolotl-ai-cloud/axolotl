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

    @pytest.mark.parametrize(
        "quant",
        [
            {},
            # neither pairing reaches a `BitsAndBytesConfig`, so accelerate still moves
            # the model and the flag would silently do nothing
            {"load_in_4bit": True},
            {"load_in_8bit": True},
        ],
    )
    def test_requires_a_bitsandbytes_pairing(
        self, min_base_cfg, gpu_caps, env_caps, quant
    ):
        cfg = DictDefault(ple_cpu_offload=True, adapter="lora", **quant) | min_base_cfg
        if quant.get("load_in_8bit"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
            return
        with pytest.raises(ValueError, match="ple_cpu_offload requires qlora"):
            validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)

    @pytest.mark.parametrize(
        "fsdp_cfg",
        [
            {"fsdp_config": {"fsdp_version": 2}},
            # deprecated but still what `ModelLoader.is_fsdp_enabled` keys on
            {"fsdp": ["full_shard", "auto_wrap"]},
        ],
    )
    def test_rejects_fsdp(self, min_base_cfg, gpu_caps, env_caps, fsdp_cfg):
        cfg = (
            DictDefault(
                ple_cpu_offload=True,
                adapter="qlora",
                load_in_4bit=True,
                **fsdp_cfg,
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


def test_merge_lora_clears_the_flag(tmp_path, monkeypatch):
    """merge-lora forces load_in_4bit off, so leaving ple_cpu_offload set would make the
    validator reject every config that ships the flag on."""
    import yaml

    from axolotl.cli import merge_lora as merge_lora_cli

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "adapter": "qlora",
                "load_in_4bit": True,
                "ple_cpu_offload": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "sequence_len": 512,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,
                "output_dir": str(adapter_dir),
                "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
            }
        )
    )

    captured = {}
    monkeypatch.setattr(
        merge_lora_cli, "do_merge_lora", lambda cfg: captured.update(cfg=cfg)
    )
    merge_lora_cli.do_cli(config=str(config_path))

    assert captured["cfg"].ple_cpu_offload is False


def test_merge_lora_overrides_beat_the_cli_flag(tmp_path, monkeypatch):
    """``--ple-cpu-offload`` reaches ``do_cli`` as a kwarg of the same name as the merge
    override, which used to bind ``load_cfg``'s keyword twice and raise ``TypeError``."""
    import yaml

    from axolotl.cli import merge_lora as merge_lora_cli

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "adapter": "qlora",
                "load_in_4bit": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "sequence_len": 512,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,
                "output_dir": str(adapter_dir),
                "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
            }
        )
    )

    captured = {}
    monkeypatch.setattr(
        merge_lora_cli, "do_merge_lora", lambda cfg: captured.update(cfg=cfg)
    )
    merge_lora_cli.do_cli(
        config=str(config_path), ple_cpu_offload=True, quantize_moe_experts=True
    )

    assert captured["cfg"].ple_cpu_offload is False
    assert captured["cfg"].quantize_moe_experts is False


class TestPleCpuOffloadMultiGpu:
    """DDP is untested rather than broken, so it warns instead of failing."""

    @staticmethod
    def _warnings(monkeypatch):
        """Capture the module logger directly.

        ``caplog`` is not usable here: any earlier test that runs an axolotl CLI entry
        point reconfigures logging for the whole process and the records stop arriving.
        """
        import axolotl.utils.schemas.config as config_module

        seen = []
        monkeypatch.setattr(
            config_module.LOG, "warning", lambda msg, *a, **k: seen.append(str(msg))
        )
        return seen

    def test_warns_on_multi_gpu(self, min_base_cfg, gpu_caps, env_caps, monkeypatch):
        seen = self._warnings(monkeypatch)
        gpu_caps["n_gpu"] = 2
        cfg = (
            DictDefault(ple_cpu_offload=True, adapter="qlora", load_in_4bit=True)
            | min_base_cfg
        )
        validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
        assert any("only been validated on a single GPU" in m for m in seen)

    def test_quiet_on_single_gpu(self, min_base_cfg, gpu_caps, env_caps, monkeypatch):
        seen = self._warnings(monkeypatch)
        cfg = (
            DictDefault(ple_cpu_offload=True, adapter="qlora", load_in_4bit=True)
            | min_base_cfg
        )
        validate_config(cfg, capabilities=gpu_caps, env_capabilities=env_caps)
        assert not any("only been validated on a single GPU" in m for m in seen)
