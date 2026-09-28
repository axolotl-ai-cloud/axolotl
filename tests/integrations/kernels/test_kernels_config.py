"""CPU-only config-validation tests for KernelsArgs (grouped-mode + lora_mlp_kernel translation)."""

import pydantic
import pytest

from axolotl.integrations.kernels.args import KernelsArgs


def test_grouped_mode_nvfp4_accepted():
    a = KernelsArgs.model_validate(
        {"use_scattermoe": True, "dsv4_fp4_grouped_mode": "nvfp4"}
    )
    assert a.dsv4_fp4_grouped_mode == "nvfp4"


def test_grouped_mode_fp8_rejected():
    # 'fp8' is documented historically but unimplemented for training -> reject, don't no-op.
    with pytest.raises(pydantic.ValidationError, match="not implemented"):
        KernelsArgs.model_validate({"dsv4_fp4_grouped_mode": "fp8"})


def test_grouped_mode_unknown_rejected():
    with pytest.raises(pydantic.ValidationError):
        KernelsArgs.model_validate({"dsv4_fp4_grouped_mode": "int4"})


def test_lora_mlp_kernel_translated_for_dsv4():
    # On a DSV4 MoE run, lora_mlp_kernel intent is preserved as dsv4_shared_mlp_lora_kernel
    # before the generic lora_mlp_kernel is force-disabled.
    a = KernelsArgs.model_validate(
        {"use_scattermoe": True, "use_dsv4_kernels": True, "lora_mlp_kernel": True}
    )
    assert a.dsv4_shared_mlp_lora_kernel is True


def test_lora_mlp_kernel_not_translated_for_non_dsv4():
    # Non-DSV4 MoE run: lora_mlp_kernel is just disabled, no shared-MLP flag.
    a = KernelsArgs.model_validate({"use_scattermoe": True, "lora_mlp_kernel": True})
    assert a.dsv4_shared_mlp_lora_kernel is None


def test_lora_mlp_kernel_explicit_shared_flag_preserved():
    # An explicit dsv4_shared_mlp_lora_kernel is not overwritten by the translation.
    a = KernelsArgs.model_validate(
        {
            "use_scattermoe": True,
            "use_dsv4_kernels": True,
            "lora_mlp_kernel": True,
            "dsv4_shared_mlp_lora_kernel": False,
        }
    )
    assert a.dsv4_shared_mlp_lora_kernel is False


def test_fp8_nonexpert_mode_validation():
    assert (
        KernelsArgs.model_validate(
            {"dsv4_fp8_nonexpert_mode": "bf16"}
        ).dsv4_fp8_nonexpert_mode
        == "bf16"
    )
    with pytest.raises(pydantic.ValidationError):
        KernelsArgs.model_validate({"dsv4_fp8_nonexpert_mode": "int8"})


def test_scattermoe_sonicmoe_mutually_exclusive():
    with pytest.raises(pydantic.ValidationError):
        KernelsArgs.model_validate({"use_scattermoe": True, "use_sonicmoe": True})


# --- intent-based config surface (#5) -------------------------------------------------------
def test_expert_backend_alias_scattermoe():
    a = KernelsArgs.model_validate({"expert_backend": "scattermoe"})
    assert a.use_scattermoe is True


def test_expert_backend_alias_sonicmoe():
    a = KernelsArgs.model_validate({"expert_backend": "sonicmoe"})
    assert a.use_sonicmoe is True


def test_expert_backend_eager_leaves_flags_unset():
    a = KernelsArgs.model_validate({"expert_backend": "eager"})
    assert a.use_scattermoe is None and a.use_sonicmoe is None


def test_expert_backend_invalid_rejected():
    with pytest.raises(pydantic.ValidationError):
        KernelsArgs.model_validate({"expert_backend": "megablocks"})


def test_nonexpert_quantization_valid_values():
    for v in ("none", "bf16", "fp8_blockwise", "nf4"):
        assert (
            KernelsArgs.model_validate(
                {"nonexpert_quantization": v}
            ).nonexpert_quantization
            == v
        )


def test_nonexpert_quantization_invalid_rejected():
    with pytest.raises(pydantic.ValidationError):
        KernelsArgs.model_validate({"nonexpert_quantization": "int8"})


def test_nonexpert_quantization_nvfp4_accepted():
    a = KernelsArgs.model_validate({"nonexpert_quantization": "nvfp4"})
    assert a.nonexpert_quantization == "nvfp4"


def test_moe_grouped_backend_valid_and_invalid():
    for b in ("auto", "marlin", "cutlass", "deepgemm"):
        assert (
            KernelsArgs.model_validate({"moe_grouped_backend": b}).moe_grouped_backend
            == b
        )
    with pytest.raises(pydantic.ValidationError):
        KernelsArgs.model_validate({"moe_grouped_backend": "triton"})


def test_moe_grouped_backend_dequant_rejected_for_training():
    # M1: 'dequant' has no training/autograd path (the dispatch only wires marlin/deepgemm/cutlass);
    # accepting it would silently run cutlass, so it is rejected with an explanatory message.
    with pytest.raises(pydantic.ValidationError, match="not implemented for training"):
        KernelsArgs.model_validate({"moe_grouped_backend": "dequant"})


def test_large_head_attention_validator():
    from axolotl.utils.schemas.config import AxolotlInputConfig

    fn = AxolotlInputConfig.__dict__["validate_large_head_attention"].__func__
    assert fn(AxolotlInputConfig, "AUTO") == "auto"  # case-normalized
    assert fn(AxolotlInputConfig, "sdpa") == "sdpa"
    assert fn(AxolotlInputConfig, None) is None
    with pytest.raises(ValueError, match="large_head_attention must be one of"):
        fn(AxolotlInputConfig, "trtion_flsah")  # typo rejected, not silently passed


def test_moe_dequant_chunk_size_positive_accepted():
    a = KernelsArgs.model_validate({"moe_dequant_chunk_size": 16})
    assert a.moe_dequant_chunk_size == 16


def test_moe_dequant_chunk_size_none_ok():
    assert KernelsArgs.model_validate({}).moe_dequant_chunk_size is None


@pytest.mark.parametrize("bad", [0, -1, -32])
def test_moe_dequant_chunk_size_zero_and_negative_rejected(bad):
    with pytest.raises(pydantic.ValidationError, match="positive integer"):
        KernelsArgs.model_validate({"moe_dequant_chunk_size": bad})


@pytest.mark.parametrize("bad", [2.5, True, "abc"])
def test_moe_dequant_chunk_size_non_integer_rejected(bad):
    with pytest.raises(pydantic.ValidationError, match="positive integer"):
        KernelsArgs.model_validate({"moe_dequant_chunk_size": bad})


@pytest.fixture
def kernel_caplog(caplog, monkeypatch):
    import logging

    logger = logging.getLogger("axolotl.integrations.kernels.plugin")
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
    return caplog


def test_warn_unclaimed_nonexpert_quantization_fires(kernel_caplog):
    # A non-expert quant policy set with no adapter that consumes it -> warn (no silent no-op).
    import logging

    from axolotl.integrations.kernels.adapters import ModelAdapter
    from axolotl.integrations.kernels.plugin import KernelsPlugin

    cfg = {"nonexpert_quantization": "nf4"}
    with kernel_caplog.at_level(logging.WARNING):
        KernelsPlugin._warn_unclaimed_nonexpert_quantization(cfg, [ModelAdapter()])
    assert any(
        "no active model adapter consumes it" in r.message
        for r in kernel_caplog.records
    )


def test_warn_unclaimed_nonexpert_quantization_silent_when_consumed(kernel_caplog):
    import logging

    from axolotl.integrations.kernels.adapters import ModelAdapter
    from axolotl.integrations.kernels.plugin import KernelsPlugin

    class _Consumer(ModelAdapter):
        name = "consumer"

        def consumes_nonexpert_quantization(self, cfg):
            return True

    cfg = {"nonexpert_quantization": "nf4"}
    with kernel_caplog.at_level(logging.WARNING):
        KernelsPlugin._warn_unclaimed_nonexpert_quantization(cfg, [_Consumer()])
    assert not any(
        "no active model adapter consumes it" in r.message
        for r in kernel_caplog.records
    )


@pytest.mark.parametrize("policy", [None, "none", "bf16"])
def test_warn_unclaimed_nonexpert_quantization_skips_noop_policies(
    policy, kernel_caplog
):
    import logging

    from axolotl.integrations.kernels.adapters import ModelAdapter
    from axolotl.integrations.kernels.plugin import KernelsPlugin

    cfg = {} if policy is None else {"nonexpert_quantization": policy}
    with kernel_caplog.at_level(logging.WARNING):
        KernelsPlugin._warn_unclaimed_nonexpert_quantization(cfg, [ModelAdapter()])
    assert not any(
        "no active model adapter consumes it" in r.message
        for r in kernel_caplog.records
    )


def test_nvfp4_merge_aware_accepted_with_sonicmoe_lora():
    a = KernelsArgs.model_validate(
        {
            "use_sonicmoe": True,
            "adapter": "lora",
            "nvfp4_merge_aware": True,
            "nvfp4_merge_aware_start_step": 100,
        }
    )
    assert a.nvfp4_merge_aware is True
    assert a.nvfp4_merge_aware_start_step == 100


def test_nvfp4_merge_aware_fractional_start_step():
    a = KernelsArgs.model_validate(
        {
            "use_sonicmoe": True,
            "adapter": "lora",
            "nvfp4_merge_aware": True,
            "nvfp4_merge_aware_start_step": 0.1,
        }
    )
    assert a.nvfp4_merge_aware_start_step == 0.1


def test_nvfp4_merge_aware_defers_backend_detection():
    args = KernelsArgs.model_validate({"adapter": "lora", "nvfp4_merge_aware": True})
    assert args.nvfp4_merge_aware is True


def test_nvfp4_merge_aware_without_adapter_warns_and_disables():
    from unittest.mock import patch

    with patch("axolotl.integrations.kernels.args.LOG.warning") as warning:
        args = KernelsArgs.model_validate(
            {"use_sonicmoe": True, "nvfp4_merge_aware": True}
        )
    assert args.nvfp4_merge_aware is False
    assert "NVFP4 MERGE WARNING" in warning.call_args.args[0]


def test_nvfp4_merge_aware_start_step_allows_automatic_default():
    args = KernelsArgs.model_validate(
        {"use_sonicmoe": True, "adapter": "lora", "nvfp4_merge_aware_start_step": 5}
    )
    assert args.nvfp4_merge_aware_start_step == 5


def test_nvfp4_merge_aware_disabled_ignores_start_with_warning():
    from unittest.mock import patch

    with patch("axolotl.integrations.kernels.args.LOG.warning") as warning:
        args = KernelsArgs.model_validate(
            {
                "adapter": "lora",
                "nvfp4_merge_aware": False,
                "nvfp4_merge_aware_start_step": 5,
            }
        )
    assert args.nvfp4_merge_aware_start_step is None
    assert "NVFP4 MERGE WARNING" in warning.call_args.args[0]


@pytest.mark.parametrize(
    "kernel", ["lora_mlp_kernel", "lora_qkv_kernel", "lora_o_kernel"]
)
def test_nvfp4_merge_aware_accepts_explicit_lora_kernels(kernel):
    args = KernelsArgs.model_validate(
        {
            "use_sonicmoe": True,
            "adapter": "lora",
            "nvfp4_merge_aware": True,
            kernel: True,
        }
    )
    assert args.nvfp4_merge_aware is True


def test_nvfp4_merge_aware_skips_lora_kernel_auto_enable():
    from axolotl.utils.config import validate_config
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            "base_model": "dummy_model",
            "datasets": [{"path": "dummy_dataset", "type": "alpaca"}],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["q_proj"],
            "plugins": ["axolotl.integrations.kernels.KernelsPlugin"],
            "use_sonicmoe": True,
            "nvfp4_merge_aware": True,
        }
    )
    result = validate_config(cfg)
    assert not result["lora_qkv_kernel"]
    assert not result["lora_o_kernel"]
    assert not result["lora_mlp_kernel"]


@pytest.mark.parametrize("bad", [-1, 1.5, -0.5, True])
def test_nvfp4_merge_aware_start_step_invalid(bad):
    with pytest.raises(pydantic.ValidationError):
        KernelsArgs.model_validate(
            {
                "use_sonicmoe": True,
                "adapter": "lora",
                "nvfp4_merge_aware": True,
                "nvfp4_merge_aware_start_step": bad,
            }
        )


def test_nvfp4_merge_aware_accepts_multilora_plugin():
    args = KernelsArgs.model_validate(
        {"adapter": "multilora", "use_sonicmoe": True, "nvfp4_merge_aware": True}
    )
    assert args.nvfp4_merge_aware
