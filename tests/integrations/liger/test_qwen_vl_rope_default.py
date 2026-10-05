"""``cfg.liger_rope=None`` must resolve to ``True`` for Qwen-VL so the upstream fused (m-)rope kernel is installed."""

from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "model_type",
    [
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen2_vl_text",
        "qwen2_5_vl_text",
        "qwen3_vl_text",
        "qwen3_vl_moe_text",
    ],
)
def test_liger_rope_auto_defaults_to_true_for_qwen_vl(model_type):
    from axolotl.integrations.liger.plugin import LigerPlugin
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            "model_config_type": model_type,
            "liger_rope": None,
            "liger_cross_entropy": False,
            "liger_fused_linear_cross_entropy": True,
            "liger_rms_norm": True,
            "liger_layer_norm": False,
            "liger_glu_activation": False,
            "liger_use_token_scaling": False,
            "torch_compile": False,
            "base_model": "fake/path",
            "trust_remote_code": False,
        }
    )

    captured = {}

    def _record(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        layer_norm: bool = True,
        model=None,
    ):
        captured.update(
            rope=rope,
            cross_entropy=cross_entropy,
            fused_linear_cross_entropy=fused_linear_cross_entropy,
            rms_norm=rms_norm,
            swiglu=swiglu,
            layer_norm=layer_norm,
        )

    from liger_kernel.transformers import monkey_patch as liger_mp

    with patch.dict(
        liger_mp.MODEL_TYPE_TO_APPLY_LIGER_FN,
        {model_type: _record},
        clear=False,
    ):
        LigerPlugin().pre_model_load(cfg)

    assert captured.get("rope") is True, (
        f"Expected rope=True default for {model_type}, got {captured.get('rope')}"
    )


def test_liger_rope_explicit_false_is_respected_for_qwen_vl():
    from axolotl.integrations.liger.plugin import LigerPlugin
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            "model_config_type": "qwen2_5_vl",
            "liger_rope": False,
            "liger_cross_entropy": False,
            "liger_fused_linear_cross_entropy": True,
            "liger_rms_norm": True,
            "liger_layer_norm": False,
            "liger_glu_activation": False,
            "liger_use_token_scaling": False,
            "torch_compile": False,
            "base_model": "fake/path",
            "trust_remote_code": False,
        }
    )

    captured = {}

    def _record(
        rope: bool = True,
        cross_entropy: bool = False,
        fused_linear_cross_entropy: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
        layer_norm: bool = True,
        model=None,
    ):
        captured.update(
            rope=rope,
            cross_entropy=cross_entropy,
            fused_linear_cross_entropy=fused_linear_cross_entropy,
            rms_norm=rms_norm,
            swiglu=swiglu,
            layer_norm=layer_norm,
        )

    from liger_kernel.transformers import monkey_patch as liger_mp

    with patch.dict(
        liger_mp.MODEL_TYPE_TO_APPLY_LIGER_FN,
        {"qwen2_5_vl": _record},
        clear=False,
    ):
        LigerPlugin().pre_model_load(cfg)

    assert captured.get("rope") is False
