"""`float32: true` keeps FP32 master weights while AMP stays explicit."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from axolotl.utils.config import normalize_config, validate_config
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
