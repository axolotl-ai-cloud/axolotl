"""Unit tests for the Adam-mini optimizer factory."""

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault

adam_mini = pytest.importorskip("adam_mini")

TINY = dict(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
)


def _build(model, **cfg):
    builder = SimpleNamespace(
        cfg=DictDefault({"optimizer": "adam_mini", **cfg}), model=model
    )
    trainer_kwargs: dict = {}
    TrainerBuilderBase._configure_optimizer(
        builder,
        {
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
        },
        trainer_kwargs,
    )
    factory_cls, optimizer_kwargs = trainer_kwargs["optimizer_cls_and_kwargs"]
    return factory_cls()(model, None, **optimizer_kwargs)


def _step(model, optimizer):
    ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()


def test_adam_mini_from_config_steps_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    model = LlamaForCausalLM(LlamaConfig(**TINY))
    optimizer = _build(model, optim_args="n_heads=4")

    assert isinstance(optimizer, adam_mini.Adam_mini)
    assert (optimizer.dim, optimizer.n_heads, optimizer.n_kv_heads) == (16, 4, 2)
    assert optimizer.world_size == 1
    decay = {g["name"]: g["weight_decay"] for g in optimizer.param_groups}
    assert decay["model.embed_tokens.weight"] == decay["lm_head.weight"] == 0.0
    assert decay["model.layers.0.mlp.up_proj.weight"] == 0.01

    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    _step(model, optimizer)
    assert not torch.equal(before, model.model.layers[0].self_attn.q_proj.weight)


def test_adam_mini_uses_config_head_dim():
    model = Qwen3ForCausalLM(Qwen3Config(**TINY, head_dim=8))
    optimizer = _build(model)
    _step(model, optimizer)

    q_proj = model.model.layers[0].self_attn.q_proj.weight
    assert optimizer.state[q_proj]["vmean"].shape[0] == 4


def test_adam_mini_rejects_fused_experts():
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            **TINY,
            head_dim=4,
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
        )
    )
    with pytest.raises(ValueError, match="only supports 1D/2D parameters"):
        _build(model)


@pytest.mark.parametrize(
    "overrides",
    [
        {"adapter": "lora"},
        {"embedding_lr": 1e-5},
        {"lr_groups": [{"name": "x", "modules": ["q_proj"], "lr": 1e-5}]},
    ],
)
def test_adam_mini_config_rejected(min_base_cfg, overrides):
    with pytest.raises(ValueError, match="adam_mini"):
        validate_config(min_base_cfg | DictDefault(optimizer="adam_mini", **overrides))
