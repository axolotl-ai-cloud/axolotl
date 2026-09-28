"""CPU coverage for the TP save path, the EP LoRA checkpoint save, and the TP token count."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import peft
import pytest
import torch
from torch.distributed.fsdp import StateDictType

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.core.trainers.mixins.distributed_parallel import (
    DistributedParallelMixin,
    tp_save_joins_all_ranks,
)


def _tp_trainer(state_dict_type, is_fsdp_enabled=True):
    calls = []
    trainer = object.__new__(DistributedParallelMixin)
    trainer.axolotl_cfg = SimpleNamespace(
        tensor_parallel_size=2, expert_parallel_size=1, adapter=None
    )
    trainer.is_fsdp_enabled = is_fsdp_enabled
    trainer.args = SimpleNamespace(should_save=False, output_dir="out")
    trainer.model = object()
    unwrapped = SimpleNamespace(
        save_pretrained=lambda *a, **k: calls.append(("save_pretrained", k))
    )
    trainer.accelerator = SimpleNamespace(
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(state_dict_type=state_dict_type)
        ),
        unwrap_model=lambda _m: unwrapped,
    )
    trainer._save_model_native = lambda *a: calls.append(("native", a))
    return trainer, calls


@pytest.mark.parametrize(
    "state_dict_type", [StateDictType.FULL_STATE_DICT, "FULL_STATE_DICT"]
)
def test_tp_full_state_dict_non_writing_rank_joins_save_pretrained(state_dict_type):
    trainer, calls = _tp_trainer(state_dict_type)
    trainer.save_model("out")
    assert [c[0] for c in calls] == ["native", "save_pretrained"]
    assert calls[1][1] == {"state_dict": {}, "is_main_process": False}


@pytest.mark.parametrize(
    "state_dict_type", [StateDictType.SHARDED_STATE_DICT, "SHARDED_STATE_DICT"]
)
def test_tp_sharded_state_dict_skips_the_all_rank_barrier(state_dict_type):
    trainer, calls = _tp_trainer(state_dict_type)
    trainer.save_model("out")
    assert calls == [("native", ("out", False))]


def test_tp_without_fsdp_gathers_on_every_rank():
    trainer, calls = _tp_trainer(None, is_fsdp_enabled=False)
    trainer.save_model("out")
    assert calls[1] == (
        "save_pretrained",
        {"state_dict": None, "is_main_process": False},
    )


def test_tp_save_joins_all_ranks_helper():
    def accel(sdt):
        return SimpleNamespace(
            state=SimpleNamespace(fsdp_plugin=SimpleNamespace(state_dict_type=sdt))
        )

    assert tp_save_joins_all_ranks(accel(StateDictType.SHARDED_STATE_DICT), False)
    assert tp_save_joins_all_ranks(accel(StateDictType.FULL_STATE_DICT), True)
    assert not tp_save_joins_all_ranks(accel(StateDictType.SHARDED_STATE_DICT), True)
    assert not tp_save_joins_all_ranks(SimpleNamespace(state=None), True)


def _lora_checkpoint_stub(monkeypatch, *, ep_size, is_fsdp_enabled, quantized):
    class NVFP4Tensor:
        pass

    class FakePeftModel:
        def parameters(self):
            local = NVFP4Tensor() if quantized else torch.zeros(1)
            return [SimpleNamespace(_local_tensor=local)]

    monkeypatch.setattr(peft, "PeftModel", FakePeftModel)
    stub = SimpleNamespace(
        is_fsdp_enabled=is_fsdp_enabled,
        axolotl_cfg=SimpleNamespace(expert_parallel_size=ep_size, fsdp_config=None),
        accelerator=SimpleNamespace(unwrap_model=lambda wrapped: wrapped),
        _is_fsdp2_quantized_param=AxolotlTrainer._is_fsdp2_quantized_param,
    )
    stub._is_fsdp2_checkpoint_save_enabled = lambda: (
        AxolotlTrainer._is_fsdp2_checkpoint_save_enabled(stub)
    )

    from axolotl.integrations.expert_parallel import shard
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin

    saves = SimpleNamespace(
        ep=MagicMock(return_value=True), fsdp2=MagicMock(return_value=True)
    )
    monkeypatch.setattr(
        ExpertParallelPlugin, "_resolve_ep_group", MagicMock(return_value=object())
    )
    monkeypatch.setattr(shard, "save_ep_lora_adapter", saves.ep)
    monkeypatch.setattr(shard, "save_fsdp2_lora_adapter", saves.fsdp2)
    return stub, FakePeftModel(), saves


@pytest.mark.parametrize("is_fsdp_enabled", [True, False])
def test_ep_lora_checkpoint_gathers_bf16_base(monkeypatch, tmp_path, is_fsdp_enabled):
    stub, model, saves = _lora_checkpoint_stub(
        monkeypatch, ep_size=2, is_fsdp_enabled=is_fsdp_enabled, quantized=False
    )
    assert AxolotlTrainer._save_gathered_lora_adapter(stub, model, str(tmp_path))
    saves.ep.assert_called_once()
    saves.fsdp2.assert_not_called()


def test_ep_lora_checkpoint_falls_back_to_quantized_save(monkeypatch, tmp_path):
    stub, model, saves = _lora_checkpoint_stub(
        monkeypatch, ep_size=2, is_fsdp_enabled=True, quantized=True
    )
    saves.ep.return_value = False
    assert AxolotlTrainer._save_gathered_lora_adapter(stub, model, str(tmp_path))
    saves.ep.assert_called_once()
    saves.fsdp2.assert_called_once()


def test_non_ep_bf16_lora_checkpoint_uses_default_save(monkeypatch, tmp_path):
    stub, model, saves = _lora_checkpoint_stub(
        monkeypatch, ep_size=1, is_fsdp_enabled=True, quantized=False
    )
    assert not AxolotlTrainer._save_gathered_lora_adapter(stub, model, str(tmp_path))
    saves.ep.assert_not_called()
    saves.fsdp2.assert_not_called()


def _count_trainer(*, tp_size, cp_size, average_tokens_across_devices=False):
    trainer = object.__new__(AxolotlTrainer)
    trainer.model_accepts_loss_kwargs = True
    trainer.compute_loss_func = None
    trainer._loss_shifts_labels = True
    trainer.args = SimpleNamespace(
        average_tokens_across_devices=average_tokens_across_devices,
        world_size=1,
        n_gpu=1,
    )
    trainer.accelerator = SimpleNamespace(
        parallelism_config=SimpleNamespace(
            tp_size=tp_size, non_data_parallel_size=tp_size * cp_size
        )
    )
    return trainer


def _batches(valid_after_shift):
    labels = torch.full((1, valid_after_shift + 1), 5)
    return [{"labels": labels}]


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "n_tokens"), [(2, 1, 7), (2, 2, 7), (4, 2, 13)]
)
def test_tp_token_count_keeps_only_the_cp_share(tp_size, cp_size, n_tokens):
    trainer = _count_trainer(tp_size=tp_size, cp_size=cp_size)
    count = trainer._get_num_items_in_batch(_batches(n_tokens), torch.device("cpu"))
    assert int(count) == n_tokens // cp_size


def test_tp_token_count_uses_shift_labels_when_present():
    trainer = _count_trainer(tp_size=2, cp_size=1)
    labels = torch.full((1, 10), -100)
    shift_labels = torch.full((1, 10), 5)
    batches = [{"labels": labels, "shift_labels": shift_labels}]
    count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))
    assert int(count) == 10


@pytest.mark.parametrize("average_tokens_across_devices", [False, True])
def test_token_count_passthrough_without_tp(average_tokens_across_devices):
    trainer = _count_trainer(
        tp_size=1,
        cp_size=2,
        average_tokens_across_devices=average_tokens_across_devices,
    )
    count = trainer._get_num_items_in_batch(_batches(7), torch.device("cpu"))
    assert int(count) == 7 // 2
