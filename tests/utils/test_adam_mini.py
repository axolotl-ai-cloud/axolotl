"""Unit tests for the Adam-mini optimizer factory."""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.utils.dict import DictDefault

adam_mini = pytest.importorskip("adam_mini")


def test_adam_mini_from_config_steps_on_cpu():
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    builder = SimpleNamespace(cfg=DictDefault({"optimizer": "adam_mini"}), model=model)
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
    optimizer = factory_cls()(model, None, **optimizer_kwargs)

    assert isinstance(optimizer, adam_mini.Adam_mini)
    assert (optimizer.dim, optimizer.n_heads, optimizer.n_kv_heads) == (16, 4, 2)

    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    model(
        input_ids=torch.tensor([[1, 2, 3, 4]]), labels=torch.tensor([[1, 2, 3, 4]])
    ).loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.model.layers[0].self_attn.q_proj.weight)
