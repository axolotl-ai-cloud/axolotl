"""Config validation of expert_parallel_size against the sharding, save and mesh layout."""

from types import SimpleNamespace

import pytest

from axolotl.integrations.base import PluginManager
from axolotl.integrations.expert_parallel.args import (
    ExpertParallelArgs,
    validate_expert_parallel_topology,
)
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault

EP_PLUGIN = "axolotl.integrations.expert_parallel.ExpertParallelPlugin"
FULL_FSDP = {
    "fsdp_version": 2,
    "fsdp_config": {"state_dict_type": "FULL_STATE_DICT"},
}


@pytest.fixture(name="ep_plugin")
def fixture_ep_plugin():
    manager = PluginManager.get_instance()
    manager.register(EP_PLUGIN)
    yield
    manager.plugins.pop(EP_PLUGIN, None)


def _validate(min_base_cfg, **kw):
    cfg = DictDefault({"plugins": [EP_PLUGIN], "expert_parallel_size": 2, **kw})
    return validate_config(cfg | min_base_cfg)


@pytest.mark.usefixtures("ep_plugin")
class TestExpertParallelRequiresFsdp2:
    def test_rejects_ddp(self, min_base_cfg):
        with pytest.raises(ValueError, match="expert_parallel_size .* requires FSDP2"):
            _validate(min_base_cfg)

    def test_rejects_fsdp_version_without_config(self, min_base_cfg):
        with pytest.raises(ValueError, match="requires FSDP2"):
            _validate(min_base_cfg, fsdp_version=2)

    def test_rejects_deepspeed(self, min_base_cfg):
        with pytest.raises(ValueError, match="requires FSDP2"):
            _validate(min_base_cfg, deepspeed="deepspeed_configs/zero3_bf16.json")

    def test_accepts_fsdp2(self, min_base_cfg):
        cfg = _validate(min_base_cfg, **FULL_FSDP)
        assert cfg.fsdp_version == 2
        assert cfg.expert_parallel_size == 2

    def test_ep_disabled_allows_ddp(self, min_base_cfg):
        cfg = _validate(min_base_cfg, expert_parallel_size=1)
        assert not cfg.fsdp_config

    def test_standalone_args_skip_topology(self):
        assert ExpertParallelArgs(expert_parallel_size=2).expert_parallel_size == 2


@pytest.mark.usefixtures("ep_plugin")
class TestExpertParallelStateDictType:
    def test_rejects_unset(self, min_base_cfg):
        with pytest.raises(ValueError, match="requires fsdp_config.state_dict_type"):
            _validate(
                min_base_cfg, fsdp_version=2, fsdp_config={"offload_params": False}
            )

    @pytest.mark.parametrize("key", ["state_dict_type", "final_state_dict_type"])
    def test_rejects_sharded(self, min_base_cfg, key):
        fsdp_config = {"state_dict_type": "FULL_STATE_DICT", key: "SHARDED_STATE_DICT"}
        with pytest.raises(ValueError, match=f"fsdp_config.{key}: FULL_STATE_DICT"):
            _validate(min_base_cfg, fsdp_version=2, fsdp_config=fsdp_config)

    def test_accepts_full(self, min_base_cfg):
        cfg = _validate(min_base_cfg, **FULL_FSDP)
        assert cfg.fsdp_config.state_dict_type == "FULL_STATE_DICT"

    def test_accepts_full_final(self, min_base_cfg):
        cfg = _validate(
            min_base_cfg,
            fsdp_version=2,
            fsdp_config={
                "state_dict_type": "FULL_STATE_DICT",
                "final_state_dict_type": "FULL_STATE_DICT",
            },
        )
        assert cfg.fsdp_config.final_state_dict_type == "FULL_STATE_DICT"


@pytest.mark.usefixtures("ep_plugin")
class TestExpertParallelReplicateAxis:
    def test_accepts_replicate_without_shard_axis(self, min_base_cfg):
        cfg = _validate(min_base_cfg, dp_replicate_size=2, **FULL_FSDP)
        assert cfg.dp_replicate_size == 2

    @pytest.mark.parametrize(
        "axes",
        [
            {"dp_shard_size": 2},
            {"context_parallel_size": 2},
            {"dp_shard_size": 2, "context_parallel_size": 2},
        ],
    )
    def test_accepts_replicate_with_shard_axis(self, min_base_cfg, axes):
        cfg = _validate(min_base_cfg, dp_replicate_size=2, **axes, **FULL_FSDP)
        assert cfg.dp_replicate_size == 2


class TestTopologyHelper:
    """The helper runs on any attribute bag, so it can be reused outside pydantic."""

    def test_noop_when_disabled(self):
        validate_expert_parallel_topology(SimpleNamespace(expert_parallel_size=1))

    def test_full_layout_passes(self):
        cfg = SimpleNamespace(
            expert_parallel_size=2,
            fsdp_version=2,
            fsdp_config=SimpleNamespace(
                state_dict_type="FULL_STATE_DICT", final_state_dict_type=None
            ),
            dp_replicate_size=2,
            dp_shard_size=2,
            context_parallel_size=1,
        )
        validate_expert_parallel_topology(cfg)

    def test_fsdp_version_from_fsdp_config(self):
        cfg = SimpleNamespace(
            expert_parallel_size=2,
            fsdp_version=None,
            fsdp_config=SimpleNamespace(
                fsdp_version=2, state_dict_type="FULL_STATE_DICT"
            ),
        )
        validate_expert_parallel_topology(cfg)
