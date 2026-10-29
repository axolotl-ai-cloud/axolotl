"""Merge-rounding check around TRL's merged vLLM weight sync."""

import logging
import types
from collections import defaultdict

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from axolotl.core.trainers.grpo.merge_check import LoraMergeCheckMixin


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(32, 32, bias=False)


def _trainer(interval=10, b_std=1e-5, vllm_lora_sync=False, main=True):
    torch.manual_seed(0)
    model = get_peft_model(
        _Toy().to(torch.bfloat16), LoraConfig(r=4, lora_alpha=8, target_modules=["q"])
    )
    with torch.no_grad():
        model.base_model.model.q.lora_B["default"].weight.normal_(0, b_std)
    synced = []
    trainer = object.__new__(LoraMergeCheckMixin)
    trainer.args = types.SimpleNamespace(
        vllm_merge_check_interval=interval, vllm_lora_sync=vllm_lora_sync
    )
    trainer.accelerator = types.SimpleNamespace(is_main_process=main)
    trainer.state = types.SimpleNamespace(global_step=0)
    trainer._metrics = {"train": defaultdict(list)}
    trainer.vllm_generation = types.SimpleNamespace(
        model=model, sync_weights=lambda: synced.append(trainer.state.global_step)
    )
    trainer._wrap_merged_sync_with_check()
    return trainer, synced


def _sync(trainer, step):
    trainer.state.global_step = step
    trainer.vllm_generation.sync_weights()


def test_checks_first_sync_then_every_interval(caplog):
    trainer, synced = _trainer(interval=10)
    with caplog.at_level(logging.INFO, logger="axolotl.utils.lora_merge_precision"):
        for step in (0, 1, 5, 11, 12, 21):
            _sync(trainer, step)
    assert synced == [0, 1, 5, 11, 12, 21]
    assert len(trainer._metrics["train"]["lora_merge/retained"]) == 3
    assert trainer._lora_merge_check_step == 21
    levels = [
        r.levelno for r in caplog.records if "merged weight sync" in r.getMessage()
    ]
    # the first check (step 1) is informational; later low-retention checks warn
    assert levels == [logging.INFO, logging.WARNING, logging.WARNING]


def test_high_retention_does_not_warn(caplog):
    trainer, _ = _trainer(b_std=1e-1)
    with caplog.at_level(logging.INFO, logger="axolotl.utils.lora_merge_precision"):
        _sync(trainer, 1)
        _sync(trainer, 20)
    assert trainer._metrics["train"]["lora_merge/retained"][-1] > 0.9
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_disabled_by_zero_interval_and_lora_sync():
    for kwargs in ({"interval": 0}, {"vllm_lora_sync": True}):
        trainer, synced = _trainer(**kwargs)
        _sync(trainer, 1)
        assert synced == [1]
        assert "lora_merge/retained" not in trainer._metrics["train"]


def test_only_main_process_measures():
    trainer, synced = _trainer(main=False)
    _sync(trainer, 1)
    assert synced == [1]
    assert "lora_merge/retained" not in trainer._metrics["train"]


def test_merge_aware_model_is_not_measured():
    from axolotl.monkeypatch.lora_merge_aware import install_lora_merge_aware_linears

    trainer, _ = _trainer()
    install_lora_merge_aware_linears(trainer.vllm_generation.model)
    _sync(trainer, 1)
    assert "lora_merge/retained" not in trainer._metrics["train"]


def test_grpo_args_carry_the_interval():
    from axolotl.core.trainers.grpo import GRPOStrategy
    from axolotl.core.trainers.grpo.args import (
        AxolotlAsyncGRPOConfig,
        AxolotlGRPOConfig,
    )
    from axolotl.utils.dict import DictDefault
    from axolotl.utils.schemas.trl import TRLConfig

    for cls in (AxolotlGRPOConfig, AxolotlAsyncGRPOConfig):
        assert cls.__dataclass_fields__["vllm_merge_check_interval"].default == 50
    kwargs = GRPOStrategy.set_training_args_kwargs(
        DictDefault(
            {"context_parallel_size": 1, "trl": TRLConfig(vllm_merge_check_interval=0)}
        )
    )
    assert kwargs["vllm_merge_check_interval"] == 0
    assert "vllm_merge_check_interval" not in GRPOStrategy.set_training_args_kwargs(
        DictDefault({"context_parallel_size": 1, "trl": TRLConfig()})
    )
