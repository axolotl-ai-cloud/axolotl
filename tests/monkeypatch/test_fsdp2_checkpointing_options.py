"""FSDP2's checkpoint wrapper applies the activation_checkpointing block's settings: the
selective policy, checkpoint kwargs, every_n_layers and checkpoint input offload."""

import copy
import functools
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

NUM_LAYERS = 4


@pytest.fixture(autouse=True)
def cpu_accelerator():
    # the offload manager reads the accelerator type at init; a CPU-only host has none
    with mock.patch.object(
        torch.accelerator, "current_accelerator", return_value=torch.device("cpu")
    ):
        yield


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            attn_implementation="sdpa",
        )
    ).train()


def _apply_fsdp_ac(model):
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    from axolotl.monkeypatch.accelerate.fsdp2 import fsdp2_apply_ac

    plugin = SimpleNamespace(
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
        ),
        transformer_cls_names_to_wrap=["LlamaDecoderLayer"],
        activation_checkpointing_offload=False,
    )
    return fsdp2_apply_ac(
        SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin)), model
    )


def _wrappers(model):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
    )

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, CheckpointWrapper)
    }


def _grads(model, ids):
    model.zero_grad(set_to_none=True)
    model(input_ids=ids, labels=ids).loss.backward()
    return {
        name.replace("._checkpoint_wrapped_module", ""): param.grad.clone()
        for name, param in model.named_parameters()
    }


@pytest.fixture
def ids():
    return torch.randint(0, 64, (2, 16), generator=torch.Generator().manual_seed(1))


def test_wrapper_applies_the_selective_policy(ids):
    from axolotl.monkeypatch.selective_checkpointing import (
        build_selective_checkpointing,
    )
    from axolotl.utils.activation_checkpointing import (
        FSDPCheckpointing,
        set_fsdp_checkpointing,
    )

    ref = _model()
    model = copy.deepcopy(ref)
    context_fn = build_selective_checkpointing(model, save=["attention"])
    set_fsdp_checkpointing(model, FSDPCheckpointing(context_fn=context_fn))
    _apply_fsdp_ac(model)

    wrappers = _wrappers(model)
    assert len(wrappers) == NUM_LAYERS
    for wrapper in wrappers.values():
        keywords = wrapper.checkpoint_fn.keywords
        assert keywords["context_fn"] is context_fn
        assert keywords["use_reentrant"] is False
    expected, got = _grads(ref, ids), _grads(model, ids)
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, msg=name)


def test_every_n_layers_counts_layers_in_forward_order(ids):
    from axolotl.utils.activation_checkpointing import (
        FSDPCheckpointing,
        set_fsdp_checkpointing,
    )

    ref = _model()
    model = copy.deepcopy(ref)
    set_fsdp_checkpointing(model, FSDPCheckpointing(every_n_layers=2))
    _apply_fsdp_ac(model)

    assert sorted(_wrappers(model)) == ["model.layers.0", "model.layers.2"]
    expected, got = _grads(ref, ids), _grads(model, ids)
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, msg=name)


def test_checkpoint_kwargs_reach_torch_checkpoint():
    from axolotl.utils.activation_checkpointing import (
        FSDPCheckpointing,
        set_fsdp_checkpointing,
    )

    model = _model()
    set_fsdp_checkpointing(
        model, FSDPCheckpointing(checkpoint_kwargs={"preserve_rng_state": False})
    )
    _apply_fsdp_ac(model)

    for wrapper in _wrappers(model).values():
        assert wrapper.checkpoint_fn.keywords["preserve_rng_state"] is False


def test_hidden_states_offload_marks_each_wrapped_layer_input(ids):
    from axolotl.monkeypatch.checkpoint_activation_offload import (
        get_checkpoint_hidden_states_offloading_ctx_manager,
    )
    from axolotl.utils.activation_checkpointing import (
        FSDPCheckpointing,
        set_fsdp_checkpointing,
    )

    ref = _model()
    model = copy.deepcopy(ref)
    set_fsdp_checkpointing(model, FSDPCheckpointing(offload_inputs=True))
    _apply_fsdp_ac(model)
    offload = get_checkpoint_hidden_states_offloading_ctx_manager(
        use_streams=False, min_offload_size=0
    )

    expected = _grads(ref, ids)
    with offload:
        got = _grads(model, ids)

    # a CPU host has nothing to offload to, so only the marking is visible here
    assert offload.stats.marked_tensors == NUM_LAYERS
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, msg=name)


def test_without_options_the_wrapper_matches_the_default():
    model = _model()
    _apply_fsdp_ac(model)

    wrappers = _wrappers(model)
    assert len(wrappers) == NUM_LAYERS
    for wrapper in wrappers.values():
        assert "context_fn" not in wrapper.checkpoint_fn.keywords


def test_wrapped_layers_do_not_fill_the_kv_cache(ids):
    model = _model()
    assert model.config.use_cache
    _apply_fsdp_ac(model)

    out = model(input_ids=ids, labels=ids)
    out.loss.backward()

    # as under transformers' checkpointing: no layer appends to a cache in training,
    # so the recompute cannot append the same keys and values a second time
    cache = out.past_key_values
    assert cache is None or cache.get_seq_length() == 0


def _loader(cfg, model):
    from axolotl.loaders.model import ModelLoader

    loader = ModelLoader(cfg, tokenizer=None)
    loader.model = model
    return loader


def _fsdp_cfg(min_base_cfg, block):
    from axolotl.utils.config import validate_config
    from axolotl.utils.dict import DictDefault

    return validate_config(
        min_base_cfg
        | DictDefault(
            fsdp_version=2,
            fsdp_config={"reshard_after_forward": True},
            activation_checkpointing={"implementation": "fsdp", **block},
        )
    )


def test_loader_attaches_the_block_settings(min_base_cfg):
    from axolotl.utils.activation_checkpointing import (
        checkpointing_enable_name,
        get_fsdp_checkpointing,
    )

    cfg = _fsdp_cfg(
        min_base_cfg,
        {
            "kwargs": {"every_n_layers": 2, "preserve_rng_state": False},
            "selective": {"save": ["attention"]},
            "offload": {"hidden_states": "cpu", "saves": "cpu"},
        },
    )
    model = _model()
    loader = _loader(cfg, model)
    loader._apply_selective_checkpointing()
    loader._configure_fsdp_activation_checkpointing()

    options = get_fsdp_checkpointing(model)
    assert options.every_n_layers == 2
    assert options.checkpoint_kwargs == {"preserve_rng_state": False}
    assert options.offload_inputs is True
    assert options.context_fn.__qualname__ == (
        "build_sac_offload_context_fn.<locals>.context_fn"
    )
    # transformers' checkpointing stays untouched
    enable = getattr(model, checkpointing_enable_name(model))
    assert not getattr(enable, "_axolotl_sac", False)


def test_loader_rejects_split_checkpointing_models(min_base_cfg):
    cfg = _fsdp_cfg(min_base_cfg, {})
    model = _model()
    model._enable_split_checkpointing = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match="implementation: hf"):
        _loader(cfg, model)._configure_fsdp_activation_checkpointing()
