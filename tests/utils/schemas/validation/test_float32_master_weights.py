"""`float32: true` keeps FP32 master weights while AMP stays explicit."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault


def _normalized(cfg):
    with (
        patch(
            "axolotl.utils.config.load_model_config",
            return_value=SimpleNamespace(model_type="llama", to_dict=dict),
        ),
        patch("axolotl.utils.config.is_torch_bf16_gpu_available", return_value=True),
    ):
        normalize_config(cfg)
    return cfg


class TestFloat32MasterWeights:
    def test_default_keeps_bf16_weights(self, min_base_cfg):
        cfg = _normalized(validate_config(DictDefault(bf16=True) | min_base_cfg))
        assert cfg.bf16 is True
        assert cfg.torch_dtype == torch.bfloat16

    def test_bf16_auto_default_keeps_bf16_weights(self, min_base_cfg):
        cfg = _normalized(validate_config(min_base_cfg))
        assert cfg.bf16 is True
        assert cfg.torch_dtype == torch.bfloat16

    def test_float32_with_bf16_is_fp32_masters_under_bf16_autocast(self, min_base_cfg):
        cfg = _normalized(
            validate_config(DictDefault(bf16=True, float32=True) | min_base_cfg)
        )
        assert cfg.bf16 is True
        assert cfg.fp16 is False
        assert cfg.torch_dtype == torch.float32

    def test_float32_with_fp16_is_fp32_masters_under_fp16_autocast(self, min_base_cfg):
        cfg = _normalized(
            validate_config(
                DictDefault(bf16=False, fp16=True, float32=True) | min_base_cfg
            )
        )
        assert cfg.fp16 is True
        assert cfg.torch_dtype == torch.float32

    @pytest.mark.parametrize("bf16", [None, "auto"])
    def test_float32_without_explicit_amp_is_pure_fp32(self, min_base_cfg, bf16):
        raw = DictDefault(float32=True) | min_base_cfg
        if bf16 is not None:
            raw["bf16"] = bf16
        cfg = _normalized(validate_config(raw))
        assert cfg.bf16 is False
        assert not cfg.fp16
        assert cfg.torch_dtype == torch.float32

    def test_float32_without_amp_rejects_flash_attention(self, min_base_cfg):
        cfg = DictDefault(float32=True, flash_attention=True) | min_base_cfg
        with pytest.raises(ValueError, match="Set `bf16: true`"):
            validate_config(cfg)

    def test_float32_without_amp_rejects_flash_attn_implementation(self, min_base_cfg):
        cfg = (
            DictDefault(float32=True, attn_implementation="flash_attention_2")
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="Set `bf16: true`"):
            validate_config(cfg)

    @pytest.mark.parametrize("half", [{"bfloat16": True}, {"float16": True}])
    def test_float32_contradicts_half_precision_weights(self, min_base_cfg, half):
        cfg = DictDefault(bf16=True, float32=True, **half) | min_base_cfg
        with pytest.raises(ValueError, match="contradict|Remove one"):
            validate_config(cfg)

    def test_resolve_dtype_without_validation_keeps_amp_explicit(self):
        from axolotl.utils.config import resolve_dtype

        cfg = DictDefault(float32=True, bf16="auto")
        with patch(
            "axolotl.utils.config.is_torch_bf16_gpu_available", return_value=True
        ):
            resolve_dtype(cfg)
        assert cfg.bf16 is False
        assert cfg.torch_dtype == torch.float32


FSDP2 = {
    "fsdp_version": 2,
    "fsdp_config": {
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    },
}


CCE_PLUGIN = ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"]


def _validate_with_plugins(cfg):
    prepare_plugins(cfg)
    return validate_config(cfg)


class TestFloat32CutCrossEntropy:
    def test_rejected_without_fsdp2(self, min_base_cfg):
        cfg = (
            DictDefault(
                bf16=True, float32=True, cut_cross_entropy=True, plugins=CCE_PLUGIN
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError) as err:
            _validate_with_plugins(cfg)
        message = str(err.value)
        assert "same half dtype" in message
        assert "FSDP2" in message
        assert "liger_fused_linear_cross_entropy" in message
        assert "drop `float32`" in message

    def test_rejected_when_fsdp2_keeps_fp32_params(self, min_base_cfg):
        fsdp = {
            **FSDP2,
            "fsdp_config": {**FSDP2["fsdp_config"], "mixed_precision_policy": "fp32"},
        }
        cfg = (
            DictDefault(
                bf16=True,
                float32=True,
                cut_cross_entropy=True,
                plugins=CCE_PLUGIN,
                **fsdp,
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="same half dtype"):
            _validate_with_plugins(cfg)

    def test_rejected_with_fsdp_version_but_no_fsdp_config(self, min_base_cfg):
        cfg = (
            DictDefault(
                bf16=True,
                float32=True,
                cut_cross_entropy=True,
                plugins=CCE_PLUGIN,
                fsdp_version=2,
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="same half dtype"):
            _validate_with_plugins(cfg)

    def test_rejected_when_prefixed_policy_keeps_fp32_params(self, min_base_cfg):
        fsdp = {
            **FSDP2,
            "fsdp_config": {
                **FSDP2["fsdp_config"],
                "fsdp_mixed_precision_policy": "fp32",
            },
        }
        cfg = (
            DictDefault(
                bf16=True,
                float32=True,
                cut_cross_entropy=True,
                plugins=CCE_PLUGIN,
                **fsdp,
            )
            | min_base_cfg
        )
        with pytest.raises(ValueError, match="same half dtype"):
            _validate_with_plugins(cfg)

    def test_accepted_with_fsdp2(self, min_base_cfg):
        cfg = (
            DictDefault(
                bf16=True,
                float32=True,
                cut_cross_entropy=True,
                plugins=CCE_PLUGIN,
                **FSDP2,
            )
            | min_base_cfg
        )
        cfg = _validate_with_plugins(cfg)
        assert cfg.float32 is True
        assert cfg.cut_cross_entropy is True

    def test_accepted_without_float32(self, min_base_cfg):
        cfg = (
            DictDefault(bf16=True, cut_cross_entropy=True, plugins=CCE_PLUGIN)
            | min_base_cfg
        )
        cfg = _validate_with_plugins(cfg)
        assert cfg.cut_cross_entropy is True


GPU_CAPS = {
    "compute_capability": "sm_90",
    "bf16": True,
    "tf32": False,
    "n_gpu": 1,
    "n_node": 1,
}
ENV_CAPS = {"torch_version": "2.14.0"}
LORA = {"adapter": "lora", "lora_r": 8, "lora_alpha": 16, "lora_target_linear": True}
LORA_KERNELS = (
    "lora_mlp_kernel",
    "lora_qkv_kernel",
    "lora_o_kernel",
    "lora_embedding_kernel",
)


class TestFloat32LoraKernels:
    @pytest.mark.parametrize("amp", [{"bf16": True}, {"bf16": False, "fp16": True}])
    def test_autocast_skips_kernel_auto_enable(self, min_base_cfg, amp):
        cfg = validate_config(
            DictDefault(float32=True, **amp, **LORA) | min_base_cfg,
            capabilities=GPU_CAPS,
            env_capabilities=ENV_CAPS,
        )
        assert not any(cfg[flag] for flag in LORA_KERNELS)

    def test_pure_fp32_keeps_kernel_auto_enable(self, min_base_cfg):
        cfg = validate_config(
            DictDefault(float32=True, **LORA) | min_base_cfg,
            capabilities=GPU_CAPS,
            env_capabilities=ENV_CAPS,
        )
        assert all(cfg[flag] for flag in LORA_KERNELS)

    def test_bf16_weights_keep_kernel_auto_enable(self, min_base_cfg):
        cfg = validate_config(
            DictDefault(bf16=True, **LORA) | min_base_cfg,
            capabilities=GPU_CAPS,
            env_capabilities=ENV_CAPS,
        )
        assert all(cfg[flag] for flag in LORA_KERNELS)

    @pytest.mark.parametrize("flag", ["lora_qkv_kernel", "lora_mlp_kernel"])
    def test_autocast_rejects_explicit_fused_kernel(self, min_base_cfg, flag):
        cfg = DictDefault(bf16=True, float32=True, **LORA, **{flag: True})
        with pytest.raises(ValueError, match=f"`{flag}`"):
            validate_config(cfg | min_base_cfg)

    def test_autocast_allows_explicit_o_and_embedding_kernels(self, min_base_cfg):
        cfg = validate_config(
            DictDefault(
                bf16=True,
                float32=True,
                lora_o_kernel=True,
                lora_embedding_kernel=True,
                **LORA,
            )
            | min_base_cfg
        )
        assert cfg.lora_o_kernel is True
        assert cfg.lora_embedding_kernel is True
