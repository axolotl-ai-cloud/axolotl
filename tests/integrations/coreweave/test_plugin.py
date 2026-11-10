"""Checkpoint callback integration with ordinary HF saves."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from axolotl.integrations.coreweave.plugin import (
    CoreWeaveCheckpointCallback,
    CoreWeavePlugin,
    load_weights,
)
from axolotl.utils.delta_checkpoint import write_checkpoint

from .test_checkpoints import config


def test_callback_reads_complete_checkpoint(tmp_path):
    weights = {"a": torch.ones(2)}
    write_checkpoint(
        weights,
        tmp_path / "checkpoint-1",
        identity="base",
        config={"model_type": "test"},
    )
    callback = CoreWeaveCheckpointCallback(config(async_publish=False))
    callback.publisher = MagicMock()
    control = object()
    assert (
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            SimpleNamespace(global_step=1, is_world_process_zero=True),
            control,
        )
        is control
    )
    args = callback.publisher.publish.call_args.args
    assert torch.equal(args[0]["a"], weights["a"])
    assert args[1:] == ({"model_type": "test"}, 1)


def test_non_main_rank_does_not_upload(tmp_path):
    callback = CoreWeaveCheckpointCallback(config())
    callback.publisher = MagicMock()
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=1, is_world_process_zero=False),
        None,
    )
    callback.publisher.publish.assert_not_called()


def test_reader_rejects_delta(tmp_path):
    weights = {"a": torch.ones(2)}
    write_checkpoint(
        weights, tmp_path / "delta", identity="next", base=weights, base_identity="base"
    )
    with pytest.raises(ValueError, match="delta"):
        load_weights(tmp_path / "delta")


def test_adapter_config_rejected():
    with pytest.raises(ValueError, match="full model"):
        CoreWeavePlugin().register(
            {"coreweave": {"deployment_id": "test"}, "adapter": "qlora"}
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"trl": {"use_vllm": True}},
        {"trl": {"async_prefetch": True}},
        {"context_parallel_size": 2},
        {"trl": {"temperature": 0}},
        {"trl": {"top_k": 20}},
        {"rl": "dpo"},
    ],
)
def test_incompatible_rollout_config_is_rejected(settings):
    with pytest.raises(ValueError):
        CoreWeavePlugin().register(
            {"coreweave": config(rollouts=True).model_dump(), "rl": "grpo", **settings}
        )


def test_rollouts_do_not_register_save_publisher():
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault({"coreweave": config(rollouts=True).model_dump()})
    assert CoreWeavePlugin().add_callbacks_post_trainer(cfg, MagicMock()) == []


def async_rollout_cfg(**overrides):
    from axolotl.utils.dict import DictDefault

    cfg = {
        "coreweave": config(rollouts=True).model_dump(),
        "rl": "grpo",
        "adapter": "lora",
        "base_model": "test/model",
        "trl": {"async_prefetch": True, "token_budget": 4096},
        "checkpoint_publish": {"every": 1},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return DictDefault(cfg)


def test_async_rollouts_sample_from_the_hot_loaded_deployment():
    cfg = async_rollout_cfg()
    plugin = CoreWeavePlugin()
    plugin.register(cfg)
    args = plugin.get_training_args(cfg)
    assert args == {
        "rollout_loop": "axolotl.integrations.coreweave.async_rollouts.CoreWeaveRolloutLoop",
        "rollout_loop_kwargs": {
            "gateway_url": "https://gateway.example",
            "model": "policy",
            "api_token_env": "CW_API_TOKEN",
        },
        "external_weight_sync": True,
    }

    worker = MagicMock()
    trainer = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()), rollout_worker=worker
    )
    (callback,) = plugin.add_callbacks_post_trainer(cfg, trainer)
    assert callback.publish_at_start and callback.every == 1
    callback.on_published(3)
    worker.update_model_version.assert_called_once_with(3)


def test_async_rollouts_on_other_ranks_have_no_worker_to_update():
    cfg = async_rollout_cfg()
    trainer = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()), rollout_worker=None
    )
    (callback,) = CoreWeavePlugin().add_callbacks_post_trainer(cfg, trainer)
    callback.on_published(3)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"checkpoint_publish": None}, "checkpoint_publish.every"),
        ({"adapter": None}, "adapter: lora"),
        ({"trl": {"token_budget": None}}, "token_budget"),
        ({"trl": {"top_p": 0.9}}, "temperature sampling"),
        ({"trl": {"temperature": 0}}, "positive temperature"),
        ({"trl": {"vllm_lora_sync": True}}, "rollout_loop"),
        ({"rl": "dpo"}, "rl: grpo"),
    ],
)
def test_incompatible_async_rollout_config_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        CoreWeavePlugin().register(async_rollout_cfg(**overrides))


def test_sync_rollouts_take_no_training_args():
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault({"coreweave": config(rollouts=True).model_dump(), "rl": "grpo"})
    assert CoreWeavePlugin().get_training_args(cfg) is None


def test_bf16_lora_save_configuration_is_supported():
    CoreWeavePlugin().register(
        {
            "coreweave": config().model_dump(),
            "adapter": "lora",
            "base_model": "test/model",
            "bf16": True,
        }
    )


def test_lora_publication_does_not_require_bf16():
    CoreWeavePlugin().register(
        {
            "coreweave": config().model_dump(),
            "adapter": "lora",
            "base_model": "test/model",
            "bf16": False,
        }
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"base_model": None},
        {"load_in_4bit": True},
        {"save_safetensors": False},
        {"trust_remote_code": True},
        {"coreweave": config(rollouts=True).model_dump()},
    ],
)
def test_incompatible_lora_publication_config_is_rejected(settings):
    with pytest.raises(ValueError):
        CoreWeavePlugin().register(
            {
                "coreweave": config().model_dump(),
                "adapter": "lora",
                "base_model": "test/model",
                "bf16": True,
                **settings,
            }
        )


def test_lora_callback_uses_loaded_training_revision():
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(
        {
            "coreweave": config().model_dump(),
            "base_model": "test/model",
            "adapter": "lora",
            "revision_of_model": "main",
        }
    )
    trainer = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_commit_hash="fixed-commit"))
    )
    callback = CoreWeavePlugin().add_callbacks_post_trainer(cfg, trainer)[0]
    assert callback.base_model == "test/model"
    assert callback.revision == "fixed-commit"


def test_a_failed_base_pin_on_any_rank_fails_every_rank(monkeypatch):
    from axolotl.core.publishing import callback, pinned_base, step_publisher as step
    from axolotl.integrations.coreweave.plugin import (
        publish_config,
        step_publisher_factory,
    )

    cfg = async_rollout_cfg()
    config, settings = publish_config(cfg, config_from(cfg))
    monkeypatch.setattr(callback, "resolve_base", lambda *args: "base")
    monkeypatch.setattr(pinned_base, "pin_base", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        step, "StepPublisher", MagicMock(side_effect=AssertionError("collective"))
    )
    comm = SimpleNamespace(
        rank=0, all_gather_object=lambda error: [error, "rank 1: OSError('disk full')"]
    )
    factory = step_publisher_factory(cfg, config, settings, None)
    with pytest.raises(RuntimeError, match="rank 1: OSError"):
        factory(model=MagicMock(), comm=comm, pieces=[])


def config_from(cfg):
    from axolotl.integrations.coreweave.args import CoreWeaveConfig

    return CoreWeaveConfig.model_validate(cfg["coreweave"])
