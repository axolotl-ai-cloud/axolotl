"""Nemotron-H mixer attention discovery and real PEFT kernel parity."""

import copy

import pytest
import torch
from torch import nn

from axolotl.monkeypatch.lora_kernels import find_self_attn_in_layer


def test_mixer_discovery_skips_other_blocks_and_deduplicates():
    attn = nn.Module()
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        setattr(attn, name, nn.Linear(4, 4))
    block = nn.Module()
    block.mixer = attn
    assert list(find_self_attn_in_layer(block)) == [attn]
    block.self_attn = attn
    assert list(find_self_attn_in_layer(block)) == [attn]
    assert list(find_self_attn_in_layer(nn.Linear(4, 4))) == []
    other = nn.Module()
    other.mixer = nn.Linear(4, 4)
    assert list(find_self_attn_in_layer(other)) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tiny_nemotron_h_peft_attention_kernels(monkeypatch, dtype):
    from peft import LoraConfig, get_peft_model
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHAttention,
        NemotronHForCausalLM,
    )

    from axolotl.monkeypatch import lora_kernels
    from axolotl.utils.dict import DictDefault

    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        layers_block_type=["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mtp_layers_block_type=[],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(11)
    model = (
        get_peft_model(
            NemotronHForCausalLM(config),
            LoraConfig(
                task_type="CAUSAL_LM",
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        .cuda()
        .to(dtype)
    )
    for name, p in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(p, std=0.02)
    reference = copy.deepcopy(model)
    ids = torch.randint(0, 32, (2, 7), device="cuda")
    expected = reference(input_ids=ids, use_cache=False).logits
    expected.float().square().mean().backward()
    oracle = None
    if dtype == torch.bfloat16:
        oracle = copy.deepcopy(model).float()
        oracle(input_ids=ids, use_cache=False).logits.square().mean().backward()
    cfg = DictDefault(lora_qkv_kernel=True, lora_o_kernel=True, lora_mlp_kernel=False)
    original = NemotronHAttention.forward
    had = hasattr(NemotronHAttention, "_original_forward")
    saved = getattr(NemotronHAttention, "_original_forward", None)
    monkeypatch.setattr(
        lora_kernels, "get_attention_cls_from_config", lambda _: NemotronHAttention
    )
    try:
        lora_kernels.patch_self_attn_lora(cfg)
        lora_kernels.apply_lora_kernel_patches(model, cfg)
        attn = model.model.model.layers[0].mixer
        assert attn.apply_qkv.__func__ is lora_kernels.apply_lora_qkv
        assert attn.apply_o.__func__ is lora_kernels.apply_lora_o
        actual = model(input_ids=ids, use_cache=False).logits
        actual.float().square().mean().backward()
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
        for (name, p), (_, q) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            if p.requires_grad:
                assert p.grad is not None and q.grad is not None, name
                if dtype == torch.bfloat16:
                    actual_grad, reference_grad = p.grad.float(), q.grad.float()
                    oracle_grad = dict(oracle.named_parameters())[name].grad
                    # BF16 fusion changes rounding points; bound errors against FP32 too.
                    assert (
                        actual_grad - reference_grad
                    ).norm() / reference_grad.norm() < 0.01, name
                    assert (
                        actual_grad - reference_grad
                    ).abs().max() < 0.02 * reference_grad.abs().max(), name
                    for grad in (actual_grad, reference_grad):
                        assert (
                            grad - oracle_grad
                        ).norm() / oracle_grad.norm() < 0.015, name
                else:
                    torch.testing.assert_close(
                        p.grad, q.grad, rtol=1e-4, atol=1e-8, msg=name
                    )

    finally:
        NemotronHAttention.forward = original
        if had:
            NemotronHAttention._original_forward = saved
        elif hasattr(NemotronHAttention, "_original_forward"):
            del NemotronHAttention._original_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("fp32", [False, True])
def test_nemotron_dense_relu2_mlp_kernels(checkpoint, fp32):
    from peft import LoraConfig, get_peft_model
    from torch.utils.checkpoint import checkpoint as checkpoint_fn
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMLP

    from axolotl.kernels.lora import apply_lora_mlp_relu2
    from axolotl.utils.lora_precision import upcast_lora_parameters

    torch.manual_seed(87)
    cfg = NemotronHConfig(hidden_size=32, intermediate_size=64, mlp_hidden_act="relu2")
    model = (
        get_peft_model(
            NemotronHMLP(cfg),
            LoraConfig(r=4, lora_alpha=8, target_modules=["up_proj", "down_proj"]),
        )
        .cuda()
        .bfloat16()
    )
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(parameter, std=0.02)
    if fp32:
        upcast_lora_parameters(model)
    reference = copy.deepcopy(model)
    x = torch.randn(2, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    forward = lambda inputs: apply_lora_mlp_relu2(model.base_model.model, inputs)
    actual = (
        checkpoint_fn(forward, x, use_reentrant=False) if checkpoint else forward(x)
    )
    expected = reference(ref_x)
    for _ in range(2):
        actual.float().square().mean().backward(retain_graph=True)
        expected.float().square().mean().backward(retain_graph=True)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.002)
    assert (
        x.grad.float() - ref_x.grad.float()
    ).norm() / ref_x.grad.float().norm() < 0.03
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if p.requires_grad:
            assert p.grad.dtype == (torch.float32 if fp32 else torch.bfloat16)
            assert (
                p.grad.float() - q.grad.float()
            ).norm() / q.grad.float().norm() < 0.03, name


@pytest.mark.parametrize("block", ["mlp", "moe"])
def test_nemotron_relu2_installer_dense_and_shared(block):
    from peft import LoraConfig, get_peft_model
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    from axolotl.kernels.lora import apply_lora_mlp_relu2
    from axolotl.monkeypatch.lora_kernels import apply_lora_kernel_patches
    from axolotl.utils.dict import DictDefault

    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        layers_block_type=[block],
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        mtp_layers_block_type=[],
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        n_group=1,
        topk_group=1,
    )
    model = get_peft_model(
        NemotronHForCausalLM(config),
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            target_modules=["up_proj", "down_proj"],
        ),
    )
    apply_lora_kernel_patches(model, DictDefault(lora_mlp_kernel=True))
    mixer = model.model.model.layers[0].mixer
    dense = mixer if block == "mlp" else mixer.shared_experts
    assert dense.forward.__func__ is apply_lora_mlp_relu2
    if block == "moe":
        assert mixer.experts.forward.__func__ is not apply_lora_mlp_relu2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu2_variable_lengths_share_compiled_kernels(dtype):
    import triton

    from axolotl.kernels.relu2 import _relu2_backward, _relu2_forward, apply_relu2

    forward_hashes, backward_hashes = set(), set()
    for count in (257, 513, 2049):
        x = torch.randn(count, device="cuda", dtype=dtype, requires_grad=True)
        reference = x.detach().clone().requires_grad_(True)
        upstream = torch.randn_like(x)
        actual = apply_relu2(x)
        expected = reference.clamp_min(0).square()
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, reference.grad)
        out = torch.empty_like(x)
        grid = (triton.cdiv(count, 1024),)
        forward_hashes.add(
            _relu2_forward.warmup(x, out, count, BLOCK=1024, grid=grid).hash
        )
        backward_hashes.add(
            _relu2_backward.warmup(upstream, x, out, count, BLOCK=1024, grid=grid).hash
        )
    assert len(forward_hashes) == len(backward_hashes) == 1
