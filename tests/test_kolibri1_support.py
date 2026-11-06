"""Kolibri-1 (`kolibri1`) runs on transformers' native implementation."""

from transformers import AutoConfig
from transformers.models.kolibri1 import modeling_kolibri1

from axolotl.common.architectures import MOE_ARCH_BLOCK
from axolotl.monkeypatch.multipack import SUPPORTED_MULTIPACK_MODEL_TYPES
from axolotl.utils.dict import DictDefault


def test_moe_block_and_multipack_registered():
    assert hasattr(modeling_kolibri1, MOE_ARCH_BLOCK["kolibri1"])
    assert "kolibri1" in SUPPORTED_MULTIPACK_MODEL_TYPES


def test_lora_qkv_o_kernels_patch_native_attention(tmp_path):
    from axolotl.monkeypatch.lora_kernels import patch_self_attn_lora

    AutoConfig.for_model("kolibri1").save_pretrained(tmp_path)
    attention_cls = modeling_kolibri1.Kolibri1Attention
    original_forward = attention_cls.forward
    try:
        patch_self_attn_lora(
            DictDefault(
                base_model=str(tmp_path), lora_qkv_kernel=True, lora_o_kernel=True
            )
        )
        assert attention_cls.forward.__name__ == "axolotl_attn_forward"
    finally:
        attention_cls.forward = original_forward
        if hasattr(attention_cls, "_original_forward"):
            delattr(attention_cls, "_original_forward")
