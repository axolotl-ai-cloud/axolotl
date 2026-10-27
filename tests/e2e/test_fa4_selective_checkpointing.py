"""GPU check: SAC saves FA4 attention without changing the loss or the gradients."""

import pytest
import torch

from axolotl.monkeypatch.attention import (
    flash_attn_4 as fa4,
    flash_attn_4_ops as fa4_ops,
)
from axolotl.monkeypatch.selective_checkpointing import apply_selective_checkpointing

LAYERS = 2
SEQ = 128


def _fa4_ready():
    if not torch.cuda.is_available() or not fa4_ops._REGISTERED:
        return False
    try:
        import flash_attn.cute  # noqa: F401
    except ImportError:
        return False
    return fa4.fa4_usable()


pytestmark = [
    # FA4 also runs on SM90, but CI only has it on B200, where it covers the SM100 backward.
    pytest.mark.gpu_blackwell,
    pytest.mark.skipif(
        not _fa4_ready(),
        reason="requires a GPU with Flash Attention 4 (flash_attn.cute)",
    ),
]


def _model(arch, sac):
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
    )

    common = {
        "vocab_size": 512,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "attn_implementation": "flash_attention_4",
        "use_cache": False,
    }
    if arch == "mistral":
        config = MistralConfig(sliding_window=32, **common)
        cls = MistralForCausalLM
    else:
        config = LlamaConfig(**common)
        cls = LlamaForCausalLM
    torch.manual_seed(0)
    model = cls(config).to("cuda", torch.bfloat16)
    if sac:
        apply_selective_checkpointing(model, save=["attention"])
    model.gradient_checkpointing_enable()
    model.train()
    return model


def _batch(packed):
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 512, (1, SEQ), generator=gen).cuda()
    kwargs = {"input_ids": ids, "labels": ids}
    if packed:
        half = torch.arange(SEQ // 2)
        kwargs["position_ids"] = torch.cat([half, half])[None].cuda()
    return kwargs


def _step(arch, sac, packed, monkeypatch):
    fa4.configure_fa4()
    calls = {"fwd": 0}
    kind = fa4_ops.VARLEN if packed else fa4_ops.DENSE
    func = fa4_ops._ORIGINALS[kind]
    forward = func.forward

    def counting(ctx, *args, **kwargs):
        calls["fwd"] += 1
        return forward(ctx, *args, **kwargs)

    monkeypatch.setattr(func, "forward", staticmethod(counting))
    model = _model(arch, sac)
    loss = model(**_batch(packed)).loss
    loss.backward()
    grads = {n: p.grad.float().clone() for n, p in model.named_parameters()}
    monkeypatch.undo()
    return loss.detach(), grads, calls["fwd"]


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "varlen"])
def test_sac_saves_fa4_attention(monkeypatch, packed):
    loss_gc, grads_gc, calls_gc = _step("llama", False, packed, monkeypatch)
    loss_sac, grads_sac, calls_sac = _step("llama", True, packed, monkeypatch)
    assert calls_gc == 2 * LAYERS
    assert calls_sac == LAYERS
    assert torch.equal(loss_gc, loss_sac)
    for name, grad in grads_gc.items():
        # FA4's backward accumulates dQ with atomics, so grads match to rounding
        torch.testing.assert_close(
            grads_sac[name], grad, rtol=2e-2, atol=2e-3, msg=name
        )


def test_sac_recomputes_sliding_window_fa4(monkeypatch):
    loss_gc, grads_gc, _ = _step("mistral", False, False, monkeypatch)
    loss_sac, grads_sac, calls_sac = _step("mistral", True, False, monkeypatch)
    assert calls_sac == 2 * LAYERS
    assert torch.equal(loss_gc, loss_sac)
    for name, grad in grads_gc.items():
        torch.testing.assert_close(
            grads_sac[name], grad, rtol=2e-2, atol=2e-3, msg=name
        )
