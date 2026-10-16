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
    cfg = SimpleNamespace(coreweave=config(rollouts=True))
    assert CoreWeavePlugin().add_callbacks_post_trainer(cfg, MagicMock()) == []


def test_bf16_lora_save_configuration_is_supported():
    CoreWeavePlugin().register(
        {
            "coreweave": config().model_dump(),
            "adapter": "lora",
            "base_model": "test/model",
            "bf16": True,
        }
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"bf16": False},
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
