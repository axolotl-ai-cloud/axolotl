"""Context parallelism under DeepSpeed maps onto accelerate's DeepSpeed Ulysses axis."""

import os
from types import SimpleNamespace

import pytest

from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import _get_parallel_config_kwargs


class TestParallelConfigKwargs:
    def test_deepspeed_maps_cp_to_sp_and_replicates_the_rest(self):
        kwargs = _get_parallel_config_kwargs(
            8, context_parallel_size=2, is_fsdp=False, is_deepspeed=True
        )
        assert kwargs == {
            "sp_size": 2,
            "sp_backend": "deepspeed",
            "dp_replicate_size": 4,
        }

    def test_deepspeed_cp_equals_world(self):
        kwargs = _get_parallel_config_kwargs(
            4, context_parallel_size=4, is_fsdp=False, is_deepspeed=True
        )
        assert kwargs == {"sp_size": 4, "sp_backend": "deepspeed"}

    def test_deepspeed_rejects_inconsistent_replicate(self):
        with pytest.raises(ValueError, match="dp_replicate_size"):
            _get_parallel_config_kwargs(
                8,
                context_parallel_size=2,
                dp_replicate_size=2,
                is_fsdp=False,
                is_deepspeed=True,
            )

    def test_fsdp_keeps_cp_axis(self):
        kwargs = _get_parallel_config_kwargs(8, context_parallel_size=2, is_fsdp=True)
        assert kwargs == {"cp_size": 2, "dp_shard_size": 4}


class TestParallelismEnvs:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for key in list(os.environ):
            if key.startswith("PARALLELISM_CONFIG_") or key.startswith("ACCELERATE_"):
                monkeypatch.delenv(key, raising=False)

    def test_deepspeed_exports_sp_axis(self, monkeypatch):
        from axolotl.utils import trainer as trainer_utils

        monkeypatch.setattr(trainer_utils, "get_world_size", lambda: 8)
        cfg = DictDefault(
            context_parallel_size=2,
            deepspeed="deepspeed_configs/zero2.json",
            attn_implementation="flash_attention_2",
        )
        trainer_utils.setup_parallelism_envs(cfg)
        assert os.environ["PARALLELISM_CONFIG_SP_SIZE"] == "2"
        assert os.environ["PARALLELISM_CONFIG_SP_BACKEND"] == "deepspeed"
        assert os.environ["PARALLELISM_CONFIG_SP_SEQ_LENGTH_IS_VARIABLE"] == "true"
        assert (
            os.environ["PARALLELISM_CONFIG_SP_ATTN_IMPLEMENTATION"]
            == "flash_attention_2"
        )
        assert os.environ["PARALLELISM_CONFIG_DP_REPLICATE_SIZE"] == "4"
        assert os.environ["ACCELERATE_USE_PARALLELISM_CONFIG"] == "true"
        assert "PARALLELISM_CONFIG_CP_SIZE" not in os.environ

    def test_deepspeed_autotp_shrinks_the_replicas(self, monkeypatch):
        from axolotl.utils import trainer as trainer_utils

        monkeypatch.setattr(trainer_utils, "get_world_size", lambda: 8)
        cfg = DictDefault(
            context_parallel_size=2,
            tensor_parallel_size=2,
            deepspeed="deepspeed_configs/zero2.json",
        )
        trainer_utils.setup_deepspeed_context_parallel_envs(cfg)
        assert os.environ["PARALLELISM_CONFIG_DP_REPLICATE_SIZE"] == "2"

    def test_fsdp_exports_cp_axis(self, monkeypatch):
        from axolotl.utils import trainer as trainer_utils

        monkeypatch.setattr(trainer_utils, "get_world_size", lambda: 8)
        cfg = DictDefault(context_parallel_size=2, fsdp_config={"fsdp_version": 2})
        trainer_utils.setup_parallelism_envs(cfg)
        assert os.environ["PARALLELISM_CONFIG_CP_SIZE"] == "2"
        assert "PARALLELISM_CONFIG_SP_SIZE" not in os.environ


class TestPluginGate:
    def test_ringmaster_plugin_disabled_under_deepspeed(self):
        from axolotl.integrations.context_parallel.plugin import ContextParallelPlugin

        plugin = ContextParallelPlugin()
        cp = {"size": 4}
        assert plugin._enabled(SimpleNamespace(context_parallel=cp, deepspeed=None))
        assert not plugin._enabled(
            SimpleNamespace(context_parallel=cp, deepspeed="zero3.json")
        )
