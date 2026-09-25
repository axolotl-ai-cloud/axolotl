"""Merge-aware setup defaults and explicit ordinary-LoRA warnings."""

from unittest.mock import patch

import pytest

from axolotl.integrations.kernels.merge_aware_setup import (
    configure_modelopt_merge_aware,
)
from axolotl.utils.dict import DictDefault


@pytest.mark.parametrize("adapter", ["lora", "multilora"])
def test_supported_nvfp4_defaults_to_merge_aware(adapter):
    cfg = DictDefault(adapter=adapter, use_sonicmoe=True)
    configure_modelopt_merge_aware(cfg)
    assert cfg.nvfp4_merge_aware is True


def test_explicit_opt_out_warns_and_continues():
    cfg = DictDefault(adapter="lora", use_sonicmoe=True, nvfp4_merge_aware=False)
    with patch("axolotl.integrations.kernels.merge_aware_setup.LOG.warning") as warning:
        configure_modelopt_merge_aware(cfg)
    assert cfg.nvfp4_merge_aware is False
    assert "NVFP4 MERGE WARNING" in warning.call_args.args[0]


@pytest.mark.parametrize("requested", [None, True])
def test_unsupported_backend_warns_without_claiming_merge_aware(requested):
    cfg = DictDefault(
        adapter="multilora", use_scattermoe=True, nvfp4_merge_aware=requested
    )
    with patch("axolotl.integrations.kernels.merge_aware_setup.LOG.warning") as warning:
        configure_modelopt_merge_aware(cfg)
    assert cfg.nvfp4_merge_aware is False
    assert "ordinary LoRA" in warning.call_args.args[0]


def test_full_parameter_training_is_unchanged():
    cfg = DictDefault(use_sonicmoe=True)
    configure_modelopt_merge_aware(cfg)
    assert cfg.nvfp4_merge_aware is None


@pytest.mark.parametrize("rank", ["0", "1"])
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_final_metadata_respects_rank_after_teardown(rank, initialized, enabled):
    from axolotl.integrations.kernels.plugin import KernelsPlugin

    cfg = DictDefault(use_sonicmoe=True, nvfp4_merge_aware=True, output_dir="unused")
    with (
        patch.dict("os.environ", {"RANK": rank}),
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=initialized),
        patch("torch.distributed.get_rank", return_value=int(rank)) as get_rank,
        patch(
            "axolotl.integrations.kernels.libs.sonicmoe.merge_aware_enabled",
            return_value=enabled,
        ),
        patch(
            "axolotl.integrations.kernels.merge_aware_callback.write_merge_aware_metadata"
        ) as write,
    ):
        KernelsPlugin().post_train_unload(cfg)
    assert write.call_count == int(rank == "0" and enabled)
    assert get_rank.call_count == int(initialized)


@pytest.mark.parametrize("backend", [None, "FSDP", "DeepSpeed"])
@pytest.mark.parametrize("requested", [None, True, False])
def test_native_setup_respects_opt_out_and_backend_limits(backend, requested):
    from types import SimpleNamespace

    from axolotl.integrations.kernels.merge_aware_setup import (
        configure_native_merge_aware,
    )

    native = type("NVFP4Tensor", (), {})()
    model = SimpleNamespace(parameters=lambda: iter([native]))
    cfg = DictDefault(adapter="lora", nvfp4_merge_aware=requested)
    with (
        patch(
            "axolotl.monkeypatch.torchao_nvfp4_merge.install_native_nvfp4_merge_aware_lora_linears",
            return_value=1,
        ) as install,
        patch("axolotl.integrations.kernels.merge_aware_setup.LOG.warning") as warning,
    ):
        configure_native_merge_aware(cfg, model, sharded_backend=backend)
    if backend in ("FSDP", "DeepSpeed") and requested is not False:
        install.assert_not_called()
        warning.assert_not_called()
        if backend == "FSDP":
            assert model._axolotl_native_nvfp4_merge_aware_requested
        else:
            assert model._axolotl_native_nvfp4_deepspeed_merge_aware_requested
        assert model._axolotl_native_nvfp4_metadata_requested
    elif backend or requested is False:
        install.assert_not_called()
        assert model._axolotl_merge_aware_unsupported
        assert "NVFP4 MERGE WARNING" in warning.call_args.args[0]
    else:
        install.assert_called_once_with(model)
        warning.assert_not_called()


@pytest.mark.parametrize("adapter", ["lora", "multilora"])
def test_native_setup_warns_when_no_merge_aware_forward_is_available(adapter):
    from types import SimpleNamespace

    from axolotl.integrations.kernels.merge_aware_setup import (
        configure_native_merge_aware,
    )

    model = SimpleNamespace(parameters=lambda: iter([type("NVFP4Tensor", (), {})()]))
    with (
        patch(
            "axolotl.monkeypatch.torchao_nvfp4_merge.install_native_nvfp4_merge_aware_lora_linears",
            return_value=0,
        ),
        patch("axolotl.integrations.kernels.merge_aware_setup.LOG.warning") as warning,
    ):
        configure_native_merge_aware(DictDefault(adapter=adapter), model)
    assert model._axolotl_merge_aware_unsupported
    assert "NVFP4 MERGE WARNING" in warning.call_args.args[0]


@pytest.mark.parametrize("requested", [True, False])
def test_native_multilora_ownership_preserves_explicit_opt_out(requested):
    from types import SimpleNamespace

    from axolotl.integrations.kernels.merge_aware_setup import (
        configure_native_merge_aware,
    )

    model = SimpleNamespace(
        parameters=lambda: iter([type("NVFP4Tensor", (), {})()]),
        _axolotl_multilora_native_merge_aware_managed=True,
    )
    with patch("axolotl.integrations.kernels.merge_aware_setup.LOG.warning") as warning:
        configure_native_merge_aware(
            DictDefault(adapter="multilora", nvfp4_merge_aware=requested), model
        )
    assert warning.called is (not requested)
    assert getattr(model, "_axolotl_merge_aware_unsupported", False) is (not requested)


def test_deepspeed_post_engine_callback_installs_on_engine_module(monkeypatch):
    from types import SimpleNamespace

    from axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora import (
        DeepSpeedNativeNVFP4MergeAwareCallback,
    )

    model = SimpleNamespace(_axolotl_native_nvfp4_deepspeed_merge_aware_requested=True)
    trainer = SimpleNamespace(model_wrapped=SimpleNamespace(module=model))
    installed = []
    monkeypatch.setattr(
        "axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora.install_deepspeed_native_nvfp4_merge_aware_lora_linears",
        lambda target: installed.append(target) or 1,
    )

    control = object()
    assert (
        DeepSpeedNativeNVFP4MergeAwareCallback(trainer).on_train_begin(
            None, None, control
        )
        is control
    )
    assert installed == [model]
    assert model._axolotl_native_nvfp4_deepspeed_merge_aware_installed == 1


def test_deepspeed_post_engine_callback_marks_no_eligible_projection(monkeypatch):
    from types import SimpleNamespace

    from axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora import (
        DeepSpeedNativeNVFP4MergeAwareCallback,
    )

    model = SimpleNamespace(_axolotl_native_nvfp4_deepspeed_merge_aware_requested=True)
    trainer = SimpleNamespace(model_wrapped=SimpleNamespace(module=model))
    monkeypatch.setattr(
        "axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora.install_deepspeed_native_nvfp4_merge_aware_lora_linears",
        lambda _: 0,
    )

    DeepSpeedNativeNVFP4MergeAwareCallback(trainer).on_train_begin(None, None, None)
    assert model._axolotl_merge_aware_unsupported


def test_builder_registers_deepspeed_post_engine_callback():
    from types import SimpleNamespace

    from axolotl.core.builders.base import TrainerBuilderBase
    from axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora import (
        DeepSpeedNativeNVFP4MergeAwareCallback,
    )

    class Builder(TrainerBuilderBase):
        def build(self, total_num_steps):
            del total_num_steps

    builder = object.__new__(Builder)
    builder.cfg = SimpleNamespace(plugins=[])
    builder.model = SimpleNamespace(
        _axolotl_native_nvfp4_deepspeed_merge_aware_requested=True
    )
    trainer = object()

    callbacks = builder.get_post_trainer_create_callbacks(trainer)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], DeepSpeedNativeNVFP4MergeAwareCallback)
    assert callbacks[0].trainer is trainer


@pytest.mark.parametrize("backend", [None, "FSDP", "DeepSpeed"])
@pytest.mark.parametrize("requested", [True, False])
def test_dynamic_input_setup_is_independent_of_merge_aware_opt_out(backend, requested):
    from types import SimpleNamespace

    from axolotl.integrations.kernels.merge_aware_setup import (
        configure_native_merge_aware,
    )

    native = type("NVFP4Tensor", (), {"act_quant_kwargs": object()})()
    model = SimpleNamespace(parameters=lambda: iter([native]))
    cfg = DictDefault(adapter="lora", nvfp4_merge_aware=requested)
    with (
        patch(
            "axolotl.monkeypatch.torchao_nvfp4_dynamic_ste.install_native_nvfp4_dynamic_input_stes"
        ) as install_inputs,
        patch(
            "axolotl.monkeypatch.torchao_nvfp4_merge.install_native_nvfp4_merge_aware_lora_linears",
            return_value=1,
        ),
        patch("axolotl.integrations.kernels.merge_aware_setup.LOG.warning") as warning,
    ):
        configure_native_merge_aware(cfg, model, sharded_backend=backend)
    if backend is None:
        install_inputs.assert_called_once_with(model)
    else:
        install_inputs.assert_not_called()
        assert model._axolotl_native_nvfp4_dynamic_input_gradients_requested == backend
    if not requested:
        assert model._axolotl_merge_aware_unsupported
        assert "explicitly disabled" in warning.call_args.args[1]


@pytest.mark.parametrize("requested", [True, False])
@pytest.mark.parametrize("backend", [None, "FSDP", "DeepSpeed"])
def test_native_multilora_never_installs_generic_dynamic_ste(
    monkeypatch, requested, backend
):
    from types import SimpleNamespace

    import axolotl.monkeypatch.torchao_nvfp4_dynamic_ste as dynamic_ste
    from axolotl.integrations.kernels.merge_aware_setup import (
        configure_native_merge_aware,
    )

    class NVFP4Tensor:
        act_quant_kwargs = object()

    model = SimpleNamespace(
        parameters=lambda: iter([NVFP4Tensor()]),
        _axolotl_multilora_native_merge_aware_managed=True,
    )
    monkeypatch.setattr(
        dynamic_ste,
        "install_native_nvfp4_dynamic_input_stes",
        lambda _: pytest.fail("core generic STE must not replace Multi-LoRA routing"),
    )

    configure_native_merge_aware(
        DictDefault(adapter="multilora", nvfp4_merge_aware=requested),
        model,
        sharded_backend=backend,
    )

    assert (
        getattr(model, "_axolotl_native_nvfp4_dynamic_input_gradients", False) is False
    )
    assert (
        getattr(model, "_axolotl_native_nvfp4_dynamic_input_gradients_requested", None)
        is None
    )
