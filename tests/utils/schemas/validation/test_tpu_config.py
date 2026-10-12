"""
Tests for check_tpu_config validator in AxolotlInputConfig.

All tests run without real TPU hardware: is_torch_xla_available() is patched
so the validator fires, and the tests verify that unsupported configs raise
NotImplementedError while safe defaults are auto-applied.
"""

from unittest.mock import patch

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault

XLA_PATCH = "axolotl.utils.schemas.validation.is_torch_xla_available"


@pytest.fixture()
def tpu_cfg(min_base_cfg):
    """Minimal valid config when running on TPU (bf16, sdpa, no adapter)."""
    return min_base_cfg | DictDefault(
        bf16=True,
        attn_implementation="sdpa",
    )


# ---------------------------------------------------------------------------
# Validator is a no-op when XLA is not available
# ---------------------------------------------------------------------------

class TestTpuValidatorNoop:
    def test_noop_when_xla_unavailable(self, min_base_cfg):
        """Validator must not fire on CUDA/CPU boxes."""
        with patch(XLA_PATCH, return_value=False):
            # qlora would normally be blocked on TPU; here it must pass through
            cfg = min_base_cfg | DictDefault(adapter="qlora", load_in_4bit=True)
            # validate_config may raise for other reasons on a GPU box,
            # but it must NOT raise NotImplementedError from check_tpu_config
            try:
                validate_config(cfg)
            except NotImplementedError as exc:
                pytest.fail(f"check_tpu_config fired when XLA unavailable: {exc}")
            except Exception:
                pass  # other validators may reject qlora without a model — that's fine


# ---------------------------------------------------------------------------
# Quantisation / adapter rejections
# ---------------------------------------------------------------------------

class TestQuantisationRejections:
    def test_qlora_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(adapter="qlora")
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="qlora"):
                validate_config(cfg)

    def test_load_in_4bit_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(load_in_4bit=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="load_in_4bit"):
                validate_config(cfg)

    def test_load_in_8bit_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(load_in_8bit=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="load_in_8bit"):
                validate_config(cfg)

    def test_8bit_optimizer_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(optimizer="adamw_bnb_8bit")
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="adamw_bnb_8bit"):
                validate_config(cfg)


# ---------------------------------------------------------------------------
# Attention rejections
# ---------------------------------------------------------------------------

class TestAttentionRejections:
    @pytest.mark.parametrize(
        "attn",
        [
            "flash_attention_2",
            "flash_attention_3",
            "flash_attention_4",
            "flash_attention_torch",
        ],
    )
    def test_flash_attn_impl_rejected(self, tpu_cfg, attn):
        cfg = tpu_cfg | DictDefault(attn_implementation=attn)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match=attn):
                validate_config(cfg)

    def test_flash_attention_flag_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(flash_attention=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="flash_attention"):
                validate_config(cfg)

    def test_xformers_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(xformers_attention=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="xformers"):
                validate_config(cfg)

    def test_eager_passes(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(attn_implementation="eager")
        with patch(XLA_PATCH, return_value=True):
            validated = validate_config(cfg)
        assert validated.attn_implementation == "eager"

    def test_sdpa_passes(self, tpu_cfg):
        with patch(XLA_PATCH, return_value=True):
            validated = validate_config(tpu_cfg)
        assert validated.attn_implementation == "sdpa"


# ---------------------------------------------------------------------------
# Parallelism / framework rejections
# ---------------------------------------------------------------------------

class TestParallelismRejections:
    def test_deepspeed_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(deepspeed="zero3.json")
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="deepspeed"):
                validate_config(cfg)

    def test_fsdp_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(fsdp=["full_shard"])
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="fsdp"):
                validate_config(cfg)

    def test_tensor_parallel_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(tensor_parallel_size=2)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="tensor_parallel_size"):
                validate_config(cfg)

    def test_context_parallel_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(context_parallel_size=2)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="context_parallel_size"):
                validate_config(cfg)

    def test_expert_parallel_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(expert_parallel_size=2)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="expert_parallel_size"):
                validate_config(cfg)


# ---------------------------------------------------------------------------
# Dynamic-shape / RL rejections
# ---------------------------------------------------------------------------

class TestDynamicShapeRejections:
    def test_sample_packing_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(sample_packing=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="sample_packing"):
                validate_config(cfg)

    def test_batch_flattening_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(batch_flattening=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="batch_flattening"):
                validate_config(cfg)

    def test_rl_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(rl="dpo")
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="rl"):
                validate_config(cfg)


# ---------------------------------------------------------------------------
# CUDA-only kernel / plugin rejections
# ---------------------------------------------------------------------------

class TestCudaOnlyKernelRejections:
    def test_liger_plugin_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(
            plugins=["axolotl.integrations.liger.LigerPlugin"]
        )
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="LigerPlugin"):
                validate_config(cfg)

    def test_cut_cross_entropy_plugin_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(
            plugins=["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"]
        )
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="CutCrossEntropyPlugin"):
                validate_config(cfg)

    def test_lora_mlp_kernel_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(lora_mlp_kernel=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="lora_mlp_kernel"):
                validate_config(cfg)

    def test_lora_qkv_kernel_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(lora_qkv_kernel=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="lora_qkv_kernel"):
                validate_config(cfg)

    def test_lora_o_proj_kernel_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(lora_o_proj_kernel=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="lora_o_proj_kernel"):
                validate_config(cfg)


# ---------------------------------------------------------------------------
# Offload / compile rejections
# ---------------------------------------------------------------------------

class TestOffloadCompileRejections:
    def test_activation_offloading_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(activation_offloading=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="activation_offloading"):
                validate_config(cfg)

    def test_gc_offload_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(gradient_checkpointing="offload")
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="offload"):
                validate_config(cfg)

    def test_torch_compile_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(torch_compile=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="torch_compile"):
                validate_config(cfg)


# ---------------------------------------------------------------------------
# Precision rejections
# ---------------------------------------------------------------------------

class TestPrecisionRejections:
    def test_fp16_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(fp16=True, bf16=False)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="fp16"):
                validate_config(cfg)

    def test_tf32_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(tf32=True)
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="tf32"):
                validate_config(cfg)

    def test_gpu_memory_limit_rejected(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(gpu_memory_limit="20GiB")
        with patch(XLA_PATCH, return_value=True):
            with pytest.raises(NotImplementedError, match="gpu_memory_limit"):
                validate_config(cfg)


# ---------------------------------------------------------------------------
# Auto-defaults applied by the validator
# ---------------------------------------------------------------------------

class TestAutoDefaults:
    def test_pad_to_sequence_len_auto_set(self, tpu_cfg):
        with patch(XLA_PATCH, return_value=True):
            validated = validate_config(tpu_cfg)
        assert validated.pad_to_sequence_len is True

    def test_dataloader_drop_last_auto_set(self, tpu_cfg):
        with patch(XLA_PATCH, return_value=True):
            validated = validate_config(tpu_cfg)
        assert validated.dataloader_drop_last is True

    def test_dataloader_pin_memory_auto_set_false(self, tpu_cfg):
        with patch(XLA_PATCH, return_value=True):
            validated = validate_config(tpu_cfg)
        assert validated.dataloader_pin_memory is False

    def test_explicit_pad_to_sequence_len_preserved(self, tpu_cfg):
        cfg = tpu_cfg | DictDefault(pad_to_sequence_len=True)
        with patch(XLA_PATCH, return_value=True):
            validated = validate_config(cfg)
        assert validated.pad_to_sequence_len is True
