import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("torchao")
pytest.importorskip("peft")
from peft import LoraConfig
from peft.tuners.lora.layer import Linear
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from axolotl.monkeypatch import (
    torchao_nvfp4_merge_persistence as persistence,
)
from axolotl.monkeypatch.torchao_nvfp4_merge import (
    install_native_nvfp4_merge_aware_lora_linears,
)
from axolotl.monkeypatch.torchao_nvfp4_merge_persistence import (
    NativeNVFP4MergeMetadataCallback,
    capture_static_native_metadata,
    clear_native_metadata,
    update_native_metadata_validity,
    write_native_metadata,
)
from axolotl.train import save_trained_model
from axolotl.utils.dict import DictDefault


def _model():
    base = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
    base.weight = nn.Parameter(
        NVFP4Tensor.to_nvfp4(base.weight.detach()), requires_grad=False
    )
    layer = Linear(
        base,
        "default",
        LoraConfig(r=2, lora_alpha=4),
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
    )
    return nn.ModuleDict({"q_proj": layer})


def test_capture_uses_real_lora_module_fqn_and_persists(tmp_path):
    model = _model()
    assert install_native_nvfp4_merge_aware_lora_linears(model) == 1
    meta = capture_static_native_metadata(model, 3)
    assert meta["backend"] == "native_torchao" and set(meta["targets"]) == {
        "q_proj.weight"
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2}))
    assert write_native_metadata(tmp_path, meta)
    assert (
        json.loads((tmp_path / "adapter_config.json").read_text())["nvfp4_merge_aware"]
        == meta
    )
    clear_native_metadata(tmp_path)
    assert "nvfp4_merge_aware" not in json.loads(
        (tmp_path / "adapter_config.json").read_text()
    )


def test_capture_canonicalizes_wrapped_peft_module_names():
    model = nn.Module()
    model.module = nn.Module()
    model.module.base_model = nn.Module()
    model.module.base_model.model = _model()

    assert install_native_nvfp4_merge_aware_lora_linears(model) == 1

    metadata = capture_static_native_metadata(model)
    assert set(metadata["targets"]) == {"q_proj.weight"}


@pytest.mark.parametrize("remote_invalid", [False, True])
def test_validity_caches_the_global_fallback_state(monkeypatch, remote_invalid):
    model = nn.Module()
    model._axolotl_native_nvfp4_metadata_valid = True
    monkeypatch.setattr(persistence.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(persistence.torch.distributed, "get_backend", lambda: "gloo")

    def all_reduce(flag, op):
        assert op is torch.distributed.ReduceOp.MAX
        flag.fill_(int(remote_invalid))

    monkeypatch.setattr(persistence.torch.distributed, "all_reduce", all_reduce)

    assert update_native_metadata_validity(model) is not remote_invalid
    assert model._axolotl_native_nvfp4_metadata_valid is not remote_invalid


def test_on_save_removes_stale_metadata_after_a_fallback(tmp_path):
    model = nn.Module()
    model._axolotl_native_nvfp4_metadata = {"version": 1}
    model._axolotl_merge_aware_unsupported = True
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        json.dumps({"nvfp4_merge_aware": {"stale": True}})
    )

    NativeNVFP4MergeMetadataCallback().on_save(
        SimpleNamespace(output_dir=tmp_path),
        SimpleNamespace(global_step=7, is_world_process_zero=True),
        None,
        model=model,
    )

    assert "nvfp4_merge_aware" not in json.loads(
        (checkpoint / "adapter_config.json").read_text()
    )


class _AdapterSavingModel(nn.Module):
    def __init__(self, metadata, valid):
        super().__init__()
        self._axolotl_native_nvfp4_metadata = metadata
        self._axolotl_native_nvfp4_metadata_valid = valid

    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(
            json.dumps({"r": 2, "nvfp4_merge_aware": {"stale": True}})
        )


def _save_cfg(output_dir):
    return DictDefault(
        {
            "output_dir": str(output_dir),
            "qat": None,
            "relora": False,
            "adapter": "lora",
            "expert_parallel_size": 1,
            "fsdp_config": None,
            "fsdp": None,
            "deepspeed": None,
            "local_rank": 0,
            "rl": None,
            "llmcompressor": None,
        }
    )


@pytest.mark.parametrize(
    ("valid", "unsupported"),
    [(True, False), (False, False), (None, False), (True, True)],
)
def test_final_rank_zero_save_uses_cached_validity_without_collectives(
    monkeypatch, tmp_path, valid, unsupported
):
    metadata = {"backend": "native_torchao", "targets": {"q_proj.weight": {}}}
    model = _AdapterSavingModel(metadata, valid)
    model._axolotl_merge_aware_unsupported = unsupported

    def fail_collective(*_args, **_kwargs):
        raise AssertionError("rank-zero final save must not enter a collective")

    monkeypatch.setattr(persistence.torch.distributed, "all_reduce", fail_collective)
    save_trained_model(
        _save_cfg(tmp_path), SimpleNamespace(is_fsdp_enabled=False), model
    )

    config = json.loads((tmp_path / "adapter_config.json").read_text())
    if valid and not unsupported:
        assert config["nvfp4_merge_aware"] == metadata
    else:
        assert "nvfp4_merge_aware" not in config


def test_on_train_end_caches_global_invalid_before_rank_zero_final_save(
    monkeypatch, tmp_path
):
    metadata = {"backend": "native_torchao", "targets": {"q_proj.weight": {}}}
    model = _AdapterSavingModel(metadata, True)
    callback = NativeNVFP4MergeMetadataCallback()
    monkeypatch.setattr(persistence.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(persistence.torch.distributed, "get_backend", lambda: "gloo")

    def remote_fallback(flag, op):
        assert op is torch.distributed.ReduceOp.MAX
        flag.fill_(1)

    monkeypatch.setattr(persistence.torch.distributed, "all_reduce", remote_fallback)
    callback.on_train_end(None, None, None, model=model)

    assert not model._axolotl_native_nvfp4_metadata_valid
    monkeypatch.setattr(
        persistence.torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("final save entered a collective"),
    )
    save_trained_model(
        _save_cfg(tmp_path), SimpleNamespace(is_fsdp_enabled=False), model
    )

    assert "nvfp4_merge_aware" not in json.loads(
        (tmp_path / "adapter_config.json").read_text()
    )


@pytest.mark.parametrize("unsupported", [False, True])
def test_distributed_save_model_uses_cached_validity_without_collectives(
    monkeypatch, tmp_path, unsupported
):
    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
        Trainer,
    )
    from axolotl.monkeypatch import torchao_deepspeed, torchao_tp_lora

    metadata = {"backend": "native_torchao", "targets": {"q_proj.weight": {}}}
    model = _AdapterSavingModel(metadata, True)
    model._axolotl_merge_aware_unsupported = unsupported
    trainer = object.__new__(DistributedParallelMixin)
    trainer.model = model
    trainer.args = SimpleNamespace(should_save=True, output_dir=str(tmp_path))
    monkeypatch.setattr(
        torchao_deepspeed,
        "native_nvfp4_zero3_peft_state_dict",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        torchao_tp_lora,
        "native_nvfp4_tp_peft_state_dict",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        persistence.torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("rank-zero save entered a collective"),
    )

    def save_model(_self, output_dir, _internal_call):
        path = Path(output_dir or _self.args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text(json.dumps({"r": 2}))

    monkeypatch.setattr(Trainer, "save_model", save_model)
    trainer.save_model()

    config = json.loads((tmp_path / "adapter_config.json").read_text())
    if unsupported:
        assert "nvfp4_merge_aware" not in config
    else:
        assert config["nvfp4_merge_aware"] == metadata


def test_fsdp_captures_recipe_before_layout_normalization():
    from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fsdp import (
        normalize_dense_nvfp4_scales,
    )
    from axolotl.monkeypatch.torchao_nvfp4_merge_metadata import (
        build_native_merge_aware_metadata,
    )

    model = _model()
    original = build_native_merge_aware_metadata(
        {"q_proj.weight": model["q_proj"].base_layer.weight}, 0
    )
    persistence.prepare_sharded_native_metadata(model)
    normalize_dense_nvfp4_scales(model)
    assert model._axolotl_native_nvfp4_metadata == original


def test_fsdp_recipe_receiver_never_reads_meta_weights(monkeypatch):
    model = nn.Module()
    metadata = {"backend": "native_torchao", "targets": {"q_proj.weight": "recipe"}}
    monkeypatch.setattr(persistence.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(persistence.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        model, "named_modules", lambda: pytest.fail("read peer meta weights")
    )

    def broadcast(payload, src):
        assert src == 0
        payload[0] = {"metadata": metadata}

    monkeypatch.setattr(
        persistence.torch.distributed, "broadcast_object_list", broadcast
    )
    persistence.prepare_sharded_native_metadata(model)
    assert model._axolotl_native_nvfp4_metadata == metadata


def test_fsdp_recipe_failure_is_broadcast_without_a_guarantee(monkeypatch):
    model = _model()
    monkeypatch.setattr(persistence.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(persistence.torch.distributed, "get_rank", lambda: 0)
    messages = []

    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("unreadable recipe")

    def broadcast(payload, src):
        assert src == 0
        messages.append(payload[0])

    monkeypatch.setattr(persistence, "build_native_merge_aware_metadata", fail_capture)
    monkeypatch.setattr(
        persistence.torch.distributed, "broadcast_object_list", broadcast
    )
    persistence.prepare_sharded_native_metadata(model)
    assert messages == [{"error": "unreadable recipe"}]
    assert model._axolotl_native_nvfp4_metadata is None
    assert not model._axolotl_native_nvfp4_metadata_valid


@pytest.mark.parametrize("valid", [True, False])
def test_fsdp_final_adapter_save_persists_cached_native_metadata(
    monkeypatch, tmp_path, valid
):
    from axolotl.integrations.expert_parallel import shard

    metadata = {"backend": "native_torchao", "targets": {"q_proj.weight": "recipe"}}
    model = _AdapterSavingModel(metadata, valid)

    def save_adapter(target, output):
        target.save_pretrained(output)
        return True

    monkeypatch.setattr(shard, "save_fsdp2_lora_adapter", save_adapter)
    monkeypatch.setattr(persistence.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "0")
    save_trained_model(
        _save_cfg(tmp_path), SimpleNamespace(is_fsdp_enabled=True), model
    )
    saved = json.loads((tmp_path / "adapter_config.json").read_text())
    assert saved.get("nvfp4_merge_aware") == (metadata if valid else None)
