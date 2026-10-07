"""CPU coverage for latent mixing on native TorchAO NVFP4 merge-aware LoRA."""

import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import CheckpointError, checkpoint

pytest.importorskip("torchao")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor  # noqa: E402

from axolotl.core.trainers.mixins.nvfp4_latent_mix import (  # noqa: E402
    NVFP4LatentMixMixin,
)
from axolotl.integrations.kernels.merge_aware_latent_mix import (  # noqa: E402
    NATIVE,
    NVFP4LatentMix,
    latent_mix_paths,
    mark_latent_mix_path,
)
from axolotl.monkeypatch.torchao_nvfp4_merge import (  # noqa: E402
    install_native_nvfp4_merge_aware_lora_linears,
    native_latent_forward_enabled,
    set_native_latent_forward,
)
from axolotl.utils.dict import DictDefault  # noqa: E402

DIM, LAYERS, R = 64, 3, 8


@pytest.fixture(autouse=True)
def _reset_latent_forward():
    yield
    set_native_latent_forward(False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return F.silu(self.proj(x)) + x


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(LAYERS))
        self.checkpointed = False

    def forward(self, x):
        for block in self.blocks:
            if self.checkpointed:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


def _model(install: bool):
    torch.manual_seed(0)
    model = _Stack()
    for block in model.blocks:
        weight = block.proj.weight.detach()
        block.proj.weight = nn.Parameter(
            NVFP4Tensor.to_nvfp4(
                weight,
                per_tensor_scale=weight.float().abs().amax() / (6.0 * 448.0),
                is_swizzled_scales=False,
            ),
            requires_grad=False,
        )
    peft_model = get_peft_model(
        model, LoraConfig(r=R, lora_alpha=2 * R, target_modules=["proj"])
    )
    torch.manual_seed(1)
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            # non-zero B, so merge-aware and unmerged forwards differ
            nn.init.normal_(param, std=0.05)
    if install:
        assert install_native_nvfp4_merge_aware_lora_linears(peft_model) == LAYERS
        mark_latent_mix_path(peft_model, NATIVE)
    return peft_model


def _run(model, checkpointed=False, latent=None, flip_before_backward=False):
    model.base_model.model.checkpointed = checkpointed
    torch.manual_seed(2)
    x = torch.randn(2, 16, DIM, dtype=torch.bfloat16)
    if latent is not None:
        set_native_latent_forward(latent)
    loss = model(x).float().pow(2).mean()
    if flip_before_backward:
        set_native_latent_forward(not latent)
    loss.backward()
    set_native_latent_forward(False)
    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    model.zero_grad(set_to_none=True)
    return loss.detach(), grads


def _sampled_run(model, mix, checkpointed=False):
    with mix.micro_batch() as latent:
        return latent, _run(model, checkpointed)


def _bitwise_equal(a, b):
    return (
        torch.equal(a[0], b[0])
        and a[1].keys() == b[1].keys()
        and all(torch.equal(a[1][name], b[1][name]) for name in a[1])
    )


@pytest.mark.parametrize("checkpointed", [False, True])
def test_zero_probability_is_bitwise_merge_aware(checkpointed):
    model = _model(install=True)
    reference = _run(model, checkpointed)
    mix = NVFP4LatentMix(0.0, latent_mix_paths(model))
    for _ in range(3):
        latent, result = _sampled_run(model, mix, checkpointed)
        assert latent is False
        assert _bitwise_equal(result, reference)
    assert not native_latent_forward_enabled()


@pytest.mark.parametrize("checkpointed", [False, True])
def test_latent_draw_is_bitwise_plain_peft_lora(checkpointed):
    patched, plain = _model(install=True), _model(install=False)
    merge_aware = _run(patched, checkpointed, latent=False)
    latent = _run(patched, checkpointed, latent=True)

    assert _bitwise_equal(latent, _run(plain, checkpointed))
    assert not _bitwise_equal(latent, merge_aware)


@pytest.mark.parametrize("latent", [False, True])
def test_checkpointing_replays_the_held_draw(latent):
    model = _model(install=True)
    assert _bitwise_equal(
        _run(model, checkpointed=True, latent=latent),
        _run(model, checkpointed=False, latent=latent),
    )


def test_draw_flipped_before_recompute_is_not_silent():
    model = _model(install=True)
    expected = _run(model, checkpointed=True, latent=True)
    try:
        flipped = _run(model, checkpointed=True, latent=True, flip_before_backward=True)
    except CheckpointError:
        model.zero_grad(set_to_none=True)
        return
    assert not _bitwise_equal(flipped, expected)


def test_sampler_mixes_both_forwards_and_resets():
    patched, plain = _model(install=True), _model(install=False)
    merge_aware = _run(patched, latent=False)
    unmerged = _run(plain)
    mix = NVFP4LatentMix(0.5, latent_mix_paths(patched), seed=3)
    draws = []
    for _ in range(8):
        latent, result = _sampled_run(patched, mix)
        draws.append(latent)
        assert _bitwise_equal(result, unmerged if latent else merge_aware)
        assert not native_latent_forward_enabled()
    assert any(draws) and not all(draws)
    assert (mix.micro_batches, mix.latent_micro_batches) == (8, sum(draws))


def test_sampler_restores_after_exception():
    mix = NVFP4LatentMix(0.999, frozenset({NATIVE}))
    with pytest.raises(RuntimeError), mix.micro_batch() as latent:
        assert latent and native_latent_forward_enabled()
        raise RuntimeError
    assert not native_latent_forward_enabled()


def _draws(seed, process_index, n=400, p=0.5):
    mix = NVFP4LatentMix(p, frozenset(), seed=seed, process_index=process_index)
    out = []
    for _ in range(n):
        with mix.micro_batch() as latent:
            out.append(latent)
    return out


def test_draws_are_private_and_deterministic_per_seed_and_rank():
    torch.manual_seed(0)
    state = torch.get_rng_state()
    import random

    random.seed(0)
    py_state = random.getstate()

    rank0 = _draws(seed=42, process_index=0)
    assert rank0 == _draws(seed=42, process_index=0)
    assert rank0 != _draws(seed=42, process_index=1)
    assert rank0 != _draws(seed=7, process_index=0)
    assert 0.4 < sum(rank0) / len(rank0) < 0.6
    assert torch.equal(torch.get_rng_state(), state)
    assert random.getstate() == py_state


@pytest.mark.parametrize("p", [-0.1, 1.0])
def test_sampler_rejects_out_of_range(p):
    with pytest.raises(ValueError):
        NVFP4LatentMix(p, frozenset())


class _RecordingTrainer(NVFP4LatentMixMixin):
    """Skips Trainer.__init__; records the latent flag seen inside training_step."""

    def __new__(cls, cfg, model, seed=42):
        self = object.__new__(cls)
        self.axolotl_cfg = cfg
        self.model = model
        self.args = types.SimpleNamespace(seed=seed, process_index=0)
        self.seen = []
        return self

    def __init__(self, *args, **kwargs):
        pass


def _recording_super(self, *args, **kwargs):
    self.seen.append(native_latent_forward_enabled())
    return torch.tensor(0.0)


@pytest.fixture
def recording_super(monkeypatch):
    from transformers import Trainer

    monkeypatch.setattr(Trainer, "training_step", _recording_super)


@pytest.mark.parametrize("p", [None, 0, 0.0])
def test_mixin_is_a_passthrough_when_unset(recording_super, p):
    model = nn.Linear(2, 2)
    mark_latent_mix_path(model, NATIVE)
    trainer = _RecordingTrainer(DictDefault(nvfp4_merge_aware_latent_mix=p), model)
    for _ in range(20):
        trainer.training_step(model, {})
    assert trainer._nvfp4_latent_mix is None
    assert trainer.seen == [False] * 20


def test_mixin_holds_one_draw_per_training_step(recording_super):
    model = nn.Linear(2, 2)
    mark_latent_mix_path(model, NATIVE)
    trainer = _RecordingTrainer(DictDefault(nvfp4_merge_aware_latent_mix=0.5), model)
    for _ in range(200):
        trainer.training_step(model, {})
        assert not native_latent_forward_enabled()
    mix = trainer._nvfp4_latent_mix
    assert mix is not None and mix.micro_batches == 200
    assert sum(trainer.seen) == mix.latent_micro_batches
    assert 0 < mix.latent_micro_batches < 200
    assert trainer.seen == _draws(seed=42, process_index=0, n=200)


def test_mixin_warns_without_a_supported_path(recording_super):
    from unittest.mock import patch

    model = nn.Linear(2, 2)
    trainer = _RecordingTrainer(DictDefault(nvfp4_merge_aware_latent_mix=0.5), model)
    with patch(
        "axolotl.integrations.kernels.merge_aware_latent_mix.LOG.warning"
    ) as warning:
        trainer.training_step(model, {})
        trainer.training_step(model, {})
    assert warning.call_count == 1
    assert trainer._nvfp4_latent_mix is None
    assert trainer.seen == [False, False]
