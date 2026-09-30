"""LoRA adapters (Linear and expert ``ParamWrapper``) stay inside the activation-checkpointed
region: they are recomputed in backward and receive the same grads as without checkpointing."""

import copy
import functools

import pytest
import torch


def _tiny_peft_moe():
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
    )
    model = Qwen3MoeForCausalLM(cfg)
    return get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
        ),
    )


def _lora_grads(model, batch):
    model.zero_grad(set_to_none=True)
    model(**batch).loss.backward()
    return {
        n.replace("._checkpoint_wrapped_module", ""): p.grad.clone()
        for n, p in model.named_parameters()
        if "lora_" in n
    }


def _count_lora_forwards(model):
    counts = {}

    def hook(name):
        def _h(_m, _i, _o):
            counts[name] = counts.get(name, 0) + 1

        return _h

    for name, module in model.named_modules():
        if name.endswith(".lora_A.default") or name.endswith(".lora_B.default"):
            module.register_forward_hook(hook(name))
    return counts


@pytest.fixture
def batch():
    ids = torch.randint(0, 128, (2, 8), generator=torch.Generator().manual_seed(1))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}


def test_fsdp2_checkpoint_wrapper_keeps_adapters(batch):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

    from axolotl.monkeypatch.accelerate.fsdp2 import _activation_checkpoint_wrapper_fn

    ref = _tiny_peft_moe()
    ckpt = copy.deepcopy(ref)
    apply_activation_checkpointing(
        ckpt,
        checkpoint_wrapper_fn=_activation_checkpoint_wrapper_fn(),
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Qwen3MoeDecoderLayer}
        ),
    )
    counts = _count_lora_forwards(ckpt)
    expected = _lora_grads(ref, batch)
    got = _lora_grads(ckpt, batch)

    assert set(got) == set(expected) and len(got) >= 4 * 2 * 2
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, msg=name)
    # every Linear adapter ran twice per step: forward and the checkpoint recompute
    assert counts and all(c == 2 for c in counts.values()), counts


def test_hf_gradient_checkpointing_keeps_adapters(batch):
    ref = _tiny_peft_moe()
    ckpt = copy.deepcopy(ref)
    ckpt.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    ckpt.train()
    ref.train()
    counts = _count_lora_forwards(ckpt)
    expected = _lora_grads(ref, batch)
    got = _lora_grads(ckpt, batch)

    assert set(got) == set(expected)
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, msg=name)
    assert counts and all(c == 2 for c in counts.values()), counts


def test_fsdp2_apply_ac_wraps_each_layer_once(batch):
    from types import SimpleNamespace

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

    from axolotl.monkeypatch.accelerate.fsdp2 import (
        _has_checkpoint_wrapper,
        fsdp2_apply_ac,
    )

    ref = _tiny_peft_moe()
    ckpt = copy.deepcopy(ref)
    plugin = SimpleNamespace(
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Qwen3MoeDecoderLayer}
        ),
        transformer_cls_names_to_wrap=["Qwen3MoeDecoderLayer"],
        activation_checkpointing_offload=False,
    )
    accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))

    assert not _has_checkpoint_wrapper(ckpt)
    fsdp2_apply_ac(accelerator, ckpt)
    assert _has_checkpoint_wrapper(ckpt)

    wrappers = [m for m in ckpt.modules() if isinstance(m, CheckpointWrapper)]
    layers = [m for m in ref.modules() if isinstance(m, Qwen3MoeDecoderLayer)]
    assert len(wrappers) == len(layers)
    assert all(
        isinstance(w._checkpoint_wrapped_module, Qwen3MoeDecoderLayer) for w in wrappers
    )

    counts = _count_lora_forwards(ckpt)
    expected = _lora_grads(ref, batch)
    got = _lora_grads(ckpt, batch)
    assert set(got) == set(expected)
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, msg=name)
    # one checkpoint per layer: forward plus a single recompute
    assert counts and all(c == 2 for c in counts.values()), counts


def test_patch_routes_accelerate_ac_through_axolotl_wrapper():
    import accelerate.accelerator

    from axolotl.monkeypatch.accelerate.fsdp2 import (
        fsdp2_apply_ac,
        patch_accelerate_fsdp2,
    )

    patch_accelerate_fsdp2()
    assert accelerate.accelerator.fsdp2_apply_ac is fsdp2_apply_ac
