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
@pytest.mark.parametrize("requested", [None, False])
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
    if backend or requested is False:
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
