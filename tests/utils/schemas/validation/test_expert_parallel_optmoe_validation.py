"""Config validation for the fused optmoe expert-parallel backend."""

import logging

import pytest

from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

EP_PLUGIN = "axolotl.integrations.expert_parallel.ExpertParallelPlugin"
KERNELS_PLUGIN = "axolotl.integrations.kernels.KernelsPlugin"


def _validate(min_base_cfg, **kw):
    base = {
        "plugins": [EP_PLUGIN],
        "expert_parallel_size": 2,
        "expert_parallel_backend": "optmoe",
        "activation_checkpointing": True,
        "fsdp_version": 2,
        "fsdp_config": {"state_dict_type": "FULL_STATE_DICT"},
    }
    cfg = DictDefault({**base, **kw}) | min_base_cfg
    prepare_plugins(cfg)
    return validate_config(cfg)


@pytest.fixture
def warnings(caplog):
    caplog.set_level(logging.WARNING)
    return lambda: "\n".join(r.getMessage() for r in caplog.records)


class TestOptMoEValidation:
    def test_accepts_defaults(self, min_base_cfg, warnings):
        cfg = _validate(min_base_cfg)
        assert cfg.expert_parallel_backend == "optmoe"
        assert cfg.expert_parallel_optmoe_precision == "auto"
        assert "optmoe" not in warnings()

    @pytest.mark.parametrize(
        "extra,ignored",
        [
            ({"use_sonicmoe": True}, "use_sonicmoe"),
            ({"use_scattermoe": True}, "use_scattermoe"),
            ({"experts_implementation": "eager"}, "experts_implementation: eager"),
        ],
    )
    def test_warns_local_kernel_ignored(self, min_base_cfg, warnings, extra, ignored):
        plugins = [EP_PLUGIN, KERNELS_PLUGIN] if "use_" in ignored else [EP_PLUGIN]
        _validate(
            min_base_cfg, plugins=plugins, expert_parallel_optmoe_mode="fused", **extra
        )
        assert "local experts kernel" in warnings() and ignored in warnings()

    def test_bounded_needs_compact(self, min_base_cfg):
        with pytest.raises(ValueError, match="needs expert_parallel_optmoe_compact"):
            _validate(min_base_cfg, expert_parallel_optmoe_bounded=True)
        cfg = _validate(
            min_base_cfg,
            expert_parallel_optmoe_mode="dispatch",
            expert_parallel_optmoe_compact=True,
            expert_parallel_optmoe_bounded=True,
        )
        assert cfg.expert_parallel_optmoe_bounded

    def test_workspace_bound_key(self, min_base_cfg):
        assert _validate(min_base_cfg).expert_parallel_optmoe_workspace_bound == "auto"
        cfg = _validate(min_base_cfg, expert_parallel_optmoe_workspace_bound="max")
        assert cfg.expert_parallel_optmoe_workspace_bound == "max"
        with pytest.raises(ValueError):
            _validate(min_base_cfg, expert_parallel_optmoe_workspace_bound="small")

    def test_stored_context_key(self, min_base_cfg):
        assert _validate(min_base_cfg).expert_parallel_optmoe_stored_context == "auto"
        for value in ("recompute", "always", "off"):
            cfg = _validate(min_base_cfg, expert_parallel_optmoe_stored_context=value)
            assert cfg.expert_parallel_optmoe_stored_context == value
        with pytest.raises(ValueError):
            _validate(min_base_cfg, expert_parallel_optmoe_stored_context="yes")

    def test_nvfp4_merged_cache_key(self, min_base_cfg):
        assert (
            _validate(min_base_cfg).expert_parallel_optmoe_nvfp4_merged_cache == "layer"
        )
        cfg = _validate(min_base_cfg, expert_parallel_optmoe_nvfp4_merged_cache="all")
        assert cfg.expert_parallel_optmoe_nvfp4_merged_cache == "all"
        with pytest.raises(ValueError):
            _validate(min_base_cfg, expert_parallel_optmoe_nvfp4_merged_cache="none")

    def test_nvfp4_headroom_key(self, min_base_cfg):
        assert _validate(min_base_cfg).expert_parallel_optmoe_nvfp4_headroom_log2 == 0
        cfg = _validate(min_base_cfg, expert_parallel_optmoe_nvfp4_headroom_log2=2)
        assert cfg.expert_parallel_optmoe_nvfp4_headroom_log2 == 2
        for value in (-1, 5):
            with pytest.raises(ValueError):
                _validate(
                    min_base_cfg, expert_parallel_optmoe_nvfp4_headroom_log2=value
                )

    def test_warns_dispatch_chunks_ignored(self, min_base_cfg, warnings):
        _validate(min_base_cfg, expert_parallel_dispatch_chunks=2)
        assert "dispatch_chunks is ignored" in warnings()

    def test_warns_precision_without_backend(self, min_base_cfg, warnings):
        _validate(
            min_base_cfg,
            expert_parallel_backend="torch",
            expert_parallel_optmoe_precision="bf16",
        )
        assert "only applies to" in warnings()

    def test_rejects_reentrant_checkpointing(self, min_base_cfg):
        with pytest.raises(ValueError, match="optmoe requires non-reentrant"):
            _validate(min_base_cfg, activation_checkpointing={"reentrant": True})

    def test_rejects_unknown_precision(self, min_base_cfg):
        with pytest.raises(ValueError):
            _validate(min_base_cfg, expert_parallel_optmoe_precision="fp4")

    def test_nvfp4_lora_keeps_use_sonicmoe(self, min_base_cfg, warnings):
        cfg = _validate(
            min_base_cfg,
            plugins=[EP_PLUGIN, KERNELS_PLUGIN],
            use_sonicmoe=True,
            expert_parallel_optmoe_mode="fused",
            expert_parallel_optmoe_precision="nvfp4_lora",
        )
        assert cfg.expert_parallel_optmoe_precision == "nvfp4_lora"
        assert "local experts kernel" not in warnings()
