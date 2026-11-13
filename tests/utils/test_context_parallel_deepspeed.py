"""Context parallelism under DeepSpeed maps onto accelerate's DeepSpeed Ulysses axis."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axolotl.core.plan.errors import PlanError
from axolotl.core.plan.slices.mesh import resolve_mesh
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import mesh_to_parallelism_kwargs


def _kwargs(world_size, **config):
    return mesh_to_parallelism_kwargs(resolve_mesh(config, world_size=world_size))


class TestParallelConfigKwargs:
    def test_deepspeed_maps_cp_to_sp_and_replicates_the_rest(self):
        kwargs = _kwargs(8, context_parallel_size=2, deepspeed="zero2.json")
        assert kwargs == {
            "sp_size": 2,
            "sp_backend": "deepspeed",
            "dp_replicate_size": 4,
        }

    def test_deepspeed_cp_equals_world(self):
        kwargs = _kwargs(4, context_parallel_size=4, deepspeed="zero2.json")
        assert kwargs == {"sp_size": 4, "sp_backend": "deepspeed"}

    def test_deepspeed_rejects_inconsistent_replicate(self):
        with pytest.raises(ValueError, match="dp_replicate_size"):
            _kwargs(
                8,
                context_parallel_size=2,
                dp_replicate_size=2,
                deepspeed="zero2.json",
            )

    def test_fsdp_keeps_cp_axis(self):
        kwargs = _kwargs(8, context_parallel_size=2, fsdp_config={"fsdp_version": 2})
        assert kwargs == {"cp_size": 2, "dp_shard_size": 4}


class TestParallelismEnvs:
    @pytest.fixture(autouse=True)
    def _clean_env(self):
        with patch.dict(os.environ):
            for key in list(os.environ):
                if key.startswith("PARALLELISM_CONFIG_") or key.startswith(
                    "ACCELERATE_"
                ):
                    del os.environ[key]
            yield

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

    def test_deepspeed_autotp_fills_the_world(self, monkeypatch):
        from axolotl.utils import trainer as trainer_utils

        monkeypatch.setattr(trainer_utils, "get_world_size", lambda: 8)
        cfg = DictDefault(
            context_parallel_size=2,
            tensor_parallel_size=4,
            deepspeed="deepspeed_configs/zero2.json",
        )
        trainer_utils.setup_parallelism_envs(cfg)
        assert os.environ["PARALLELISM_CONFIG_TP_SIZE"] == "4"
        assert "PARALLELISM_CONFIG_DP_REPLICATE_SIZE" not in os.environ

    def test_deepspeed_autotp_rejects_replicas(self, monkeypatch):
        """accelerate's ParallelismConfig refuses TP beside pure dp_replicate."""
        from axolotl.utils import trainer as trainer_utils

        monkeypatch.setattr(trainer_utils, "get_world_size", lambda: 8)
        cfg = DictDefault(
            context_parallel_size=2,
            tensor_parallel_size=2,
            deepspeed="deepspeed_configs/zero2.json",
        )
        with pytest.raises(PlanError) as info:
            trainer_utils.setup_parallelism_envs(cfg)
        assert info.value.key == "dp_replicate_size"

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


@pytest.mark.parametrize("rl", ["dpo", "kto", "orpo", "grpo"])
def test_deepspeed_context_parallel_rejects_every_rl_trainer(monkeypatch, rl):
    from axolotl.utils.config import validate_config

    monkeypatch.setenv("WORLD_SIZE", "4")
    cfg = DictDefault(
        base_model="HuggingFaceTB/SmolLM2-135M",
        learning_rate=1e-3,
        datasets=[{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        sequence_len=2048,
        rl=rl,
        remove_unused_columns=False,
        deepspeed="deepspeed_configs/zero2.json",
        context_parallel_size=2,
    )
    with pytest.raises(ValueError, match="only the SFT trainer"):
        validate_config(cfg)
