"""temporarily_merge_adapter shim: exact LoRA merge -> unmerge round trips for vLLM syncs."""

import copy
import inspect
import types
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Embedding as LoraEmbedding, Linear as LoraLinear
from peft.tuners.tuners_utils import BaseTuner
from torch import nn
from trl import GRPOTrainer

from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer
from axolotl.monkeypatch.peft import exact_unmerge
from axolotl.monkeypatch.peft.exact_unmerge import (
    exact_adapter_merges,
    exact_unmerge_vllm_sync,
    patch_peft_lora_exact_unmerge,
)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(256, 192, bias=False)
        self.o = nn.Linear(192, 256, bias=True)

    def forward(self, x):
        return self.o(self.q(x))


class _EmbeddingToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(512, 64)


@pytest.fixture(autouse=True)
def _patched():
    patch_peft_lora_exact_unmerge()


def _peft(dtype=torch.bfloat16, seed=0, **lora_kwargs):
    torch.manual_seed(seed)
    base = _Toy()
    with torch.no_grad():
        # small weights next to the delta, where unmerge rounding is most visible
        base.q.weight.mul_(0.05)
    model = get_peft_model(
        base.to(dtype),
        LoraConfig(r=8, lora_alpha=16, target_modules=["q", "o"], **lora_kwargs),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 2e-3)
    return model


def _peft_embedding():
    torch.manual_seed(0)
    base = _EmbeddingToy()
    with torch.no_grad():
        base.emb.weight.mul_(0.02)
    model = get_peft_model(
        base.to(torch.bfloat16),
        LoraConfig(r=8, lora_alpha=16, target_modules=["emb"]),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_embedding" in name:
                param.normal_(0, 2e-2)
    return model


def _base_weights(model):
    return {
        n: p.detach().clone() for n, p in model.named_parameters() if "base_layer" in n
    }


def _step(model, scale=1e-4):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.add_(torch.sign(torch.randn_like(param)) * scale)


def _assert_restored(model, base):
    for name, param in model.named_parameters():
        if name in base:
            assert torch.equal(param, base[name]), name


def _stock(cls):
    return inspect.unwrap(cls.merge), inspect.unwrap(cls.unmerge)


def _lora_linears(model):
    return [m for m in model.modules() if isinstance(m, LoraLinear)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_repeated_temporary_merges_restore_base_weights(dtype):
    model = _peft(dtype)
    base = _base_weights(model)
    for _ in range(20):
        _step(model)
        with model.temporarily_merge_adapter():
            assert all(m.merged for m in _lora_linears(model))
    _assert_restored(model, base)
    assert not any(m.merged for m in _lora_linears(model))


def test_residual_chunks_cover_every_row(monkeypatch):
    monkeypatch.setattr(exact_unmerge, "_CHUNK_ELEMENTS", 256 * 7)
    model = _peft()
    base = _base_weights(model)
    for _ in range(5):
        _step(model)
        with model.temporarily_merge_adapter():
            pass
    _assert_restored(model, base)


def test_merged_weights_match_stock_peft():
    model = _peft()
    reference = copy.deepcopy(model)
    merge, _ = _stock(LoraLinear)
    for module in _lora_linears(reference):
        merge(module)
    with model.temporarily_merge_adapter():
        for (name, got), (_, expected) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert torch.equal(got, expected), name


def test_plain_merge_unmerge_and_merge_and_unload_record_nothing():
    model = _peft()
    reference = copy.deepcopy(model)
    merge, unmerge = _stock(LoraLinear)
    with patch.object(exact_unmerge, "unmerge_residual") as record:
        for _ in range(3):
            _step(model)
            with torch.no_grad():
                for (name, src), (_, dst) in zip(
                    model.named_parameters(), reference.named_parameters(), strict=True
                ):
                    if "lora_" in name:
                        dst.copy_(src)
            model.merge_adapter()
            model.unmerge_adapter()
            for module in _lora_linears(reference):
                merge(module)
            for module in _lora_linears(reference):
                unmerge(module)
        # plain round trips still drift exactly like stock PEFT
        for (name, got), (_, expected) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert torch.equal(got, expected), name
        model.merge_and_unload()
    record.assert_not_called()


def test_fp32_weights_are_not_recorded():
    model = _peft(torch.float32)
    with patch.object(exact_unmerge, "unmerge_residual") as record:
        with model.temporarily_merge_adapter():
            assert not any(
                m.__dict__.get("_unmerge_residuals") for m in _lora_linears(model)
            )
    record.assert_not_called()


def test_residual_indices_are_int32():
    assert exact_unmerge._index_dtype(2**31 - 1) == torch.int32
    assert exact_unmerge._index_dtype(2**31) == torch.int64
    model = _peft()
    with model.temporarily_merge_adapter():
        indices, values = model.base_model.model.q._unmerge_residuals["default"]
        assert indices.dtype == torch.int32
        assert values.dtype == torch.bfloat16
        assert 0 < indices.numel() < 256 * 192
    assert "_unmerge_residuals" not in model.base_model.model.q.__dict__


def test_refuses_when_already_merged():
    model = _peft()
    model.merge_adapter()
    with pytest.raises(ValueError, match="needs unmerged adapters"):
        with model.temporarily_merge_adapter():
            pass
    model.unmerge_adapter()


def test_safe_merge_and_multiple_adapters():
    model = _peft()
    model.add_adapter("other", LoraConfig(r=4, lora_alpha=8, target_modules=["q", "o"]))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B.other" in name:
                param.normal_(0, 2e-3)
    base = _base_weights(model)
    with model.temporarily_merge_adapter(safe_merge=True):
        pass
    _assert_restored(model, base)
    with model.temporarily_merge_adapter(adapter_names=["default", "other"]):
        assert model.base_model.model.q.merged_adapters == ["default", "other"]
    _assert_restored(model, base)


def test_dora_keeps_peft_behaviour():
    model = _peft(use_dora=True)
    with model.temporarily_merge_adapter():
        assert not model.base_model.model.q.__dict__.get("_unmerge_residuals")
    assert not model.base_model.model.q.merged


def test_embedding_round_trip_is_exact():
    model = _peft_embedding()
    base = _base_weights(model)
    for _ in range(20):
        _step(model)
        with model.temporarily_merge_adapter():
            pass
    _assert_restored(model, base)


def test_unpatched_embedding_drifts_in_bf16():
    model = _peft_embedding()
    emb = model.base_model.model.emb
    before = emb.base_layer.weight.detach().clone()
    merge, unmerge = _stock(LoraEmbedding)
    for _ in range(20):
        _step(model)
        merge(emb)
        unmerge(emb)
    assert not torch.equal(emb.base_layer.weight, before)


def test_offloaded_base_layer_round_trips_and_keeps_residuals_on_cpu():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import set_module_tensor_to_device

    model = _peft()
    q = model.base_model.model.q
    original = q.base_layer.weight.detach().clone()
    add_hook_to_module(
        q.base_layer,
        AlignDevicesHook(
            execution_device="cpu",
            offload=True,
            weights_map={"weight": original.clone()},
        ),
    )
    set_module_tensor_to_device(q.base_layer, "weight", "meta")
    for _ in range(5):
        _step(model)
        # PEFT onloads the layer for merge() and offloads it again afterwards
        with model.temporarily_merge_adapter():
            assert q.base_layer.weight.device.type == "meta"
            residual = q._unmerge_residuals["default"]
            assert all(t.device.type == "cpu" for t in residual)
    assert torch.equal(q.base_layer._hf_hook.weights_map["weight"], original)


def test_trl_style_sync_calls_are_exact_and_restored():
    model = _peft()
    base = _base_weights(model)
    for _ in range(10):
        _step(model)
        with exact_adapter_merges(model):
            # TRL's first sync walks the merged weights twice (metadata, then stream)
            for _ in range(2):
                model.merge_adapter()
                model.unmerge_adapter()
    _assert_restored(model, base)
    assert "merge_adapter" not in model.__dict__
    assert "unmerge_adapter" not in model.__dict__


def test_vllm_sync_wrapper_only_wraps_peft_models():
    model = _peft()
    base = _base_weights(model)

    def sync_weights():
        model.merge_adapter()
        model.unmerge_adapter()

    generation = types.SimpleNamespace(model=model, sync_weights=sync_weights)
    exact_unmerge_vllm_sync(generation)
    assert generation.sync_weights is not sync_weights
    for _ in range(10):
        _step(model)
        generation.sync_weights()
    _assert_restored(model, base)

    plain = types.SimpleNamespace(model=nn.Linear(2, 2), sync_weights=sync_weights)
    exact_unmerge_vllm_sync(plain)
    assert plain.sync_weights is sync_weights


def test_shim_is_skipped_when_peft_provides_it(monkeypatch):
    for cls in (LoraLinear, LoraEmbedding):
        merge, unmerge = _stock(cls)
        monkeypatch.setattr(cls, "merge", merge)
        monkeypatch.setattr(cls, "unmerge", unmerge)

    def native(self, adapter_names=None, safe_merge=False):
        raise NotImplementedError

    monkeypatch.setattr(BaseTuner, "temporarily_merge_adapter", native)
    patch_peft_lora_exact_unmerge()
    assert BaseTuner.temporarily_merge_adapter is native
    assert LoraLinear.merge is _stock(LoraLinear)[0]
    assert LoraEmbedding.unmerge is _stock(LoraEmbedding)[1]


def test_patch_is_idempotent():
    methods = [(c.merge, c.unmerge) for c in (LoraLinear, LoraEmbedding)]
    shim = BaseTuner.temporarily_merge_adapter
    patch_peft_lora_exact_unmerge()
    assert methods == [(c.merge, c.unmerge) for c in (LoraLinear, LoraEmbedding)]
    assert BaseTuner.temporarily_merge_adapter is shim


def test_grpo_trainer_wraps_its_vllm_sync():
    model = _peft()
    base = _base_weights(model)

    def sync_weights():
        model.merge_adapter()
        model.unmerge_adapter()

    def stub_init(self, *args, **kwargs):
        self.args = types.SimpleNamespace(
            activation_offloading=False, vllm_merge_check_interval=0
        )
        self.optimizer = self.optimizer_cls_and_kwargs = None
        self.model = model
        self.vllm_generation = types.SimpleNamespace(
            model=model, sync_weights=sync_weights
        )

    with patch.object(GRPOTrainer, "__init__", stub_init):
        trainer = AxolotlGRPOTrainer()
    assert trainer.vllm_generation.sync_weights is not sync_weights
    for _ in range(5):
        _step(model)
        trainer.vllm_generation.sync_weights()
    _assert_restored(model, base)
