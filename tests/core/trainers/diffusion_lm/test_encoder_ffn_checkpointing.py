"""DiffusionGemma encoder feed-forward checkpointing matches the uncheckpointed model."""

from __future__ import annotations

import copy

import torch
from peft import LoraConfig, get_peft_model
from transformers import DiffusionGemmaConfig

from axolotl.model_support.diffusion_gemma.modeling import (
    AxolotlDiffusionGemmaForBlockDiffusion,
    _encoder_layer_forward,
)
from axolotl.model_support.native_adapters import share_diffusion_gemma_tied_lora
from axolotl.monkeypatch.selective_checkpointing import clear_registered_saves

_TARGET = (
    r".*\.(encoder\.language_model|decoder)\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)


def _config() -> DiffusionGemmaConfig:
    return DiffusionGemmaConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 96,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 16,
            "global_head_dim": 32,
            "sliding_window": 8,
            "num_experts": 8,
            "top_k_experts": 2,
            "max_position_embeddings": 128,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "patch_size": 16,
            "position_embedding_size": 16,
        },
        canvas_length=16,
    )


def _model():
    torch.manual_seed(0)
    model = AxolotlDiffusionGemmaForBlockDiffusion(_config()).float()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "experts" in name:
                param.normal_(std=0.05)
    peft = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=_TARGET,
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            lora_dropout=0.0,
        ),
    )
    share_diffusion_gemma_tied_lora(peft)
    torch.manual_seed(1)
    with torch.no_grad():
        for name, param in peft.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.05)
    return peft


def test_encoder_feed_forward_checkpointing_matches_and_writes_the_cache_once():
    plain = _model()
    checkpointed = copy.deepcopy(plain)
    share_diffusion_gemma_tied_lora(checkpointed)
    checkpointed.get_base_model().gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    encoder_layers = checkpointed.get_base_model().model.encoder.language_model.layers
    decoder_layers = checkpointed.get_base_model().model.decoder.layers
    assert all(
        layer.forward.__func__ is _encoder_layer_forward
        for layer in (*encoder_layers, *decoder_layers)
    )
    # the split forward replaces HF's whole-layer decoder checkpoint
    assert not any(layer.gradient_checkpointing for layer in decoder_layers)
    plain.train()
    checkpointed.train()

    gen = torch.Generator().manual_seed(2)
    prompt = torch.randint(3, 128, (2, 24), generator=gen)
    canvas = torch.randint(3, 128, (2, 16), generator=gen)
    out_plain = plain(input_ids=prompt, decoder_input_ids=canvas)
    out_ckpt = checkpointed(input_ids=prompt, decoder_input_ids=canvas)
    torch.testing.assert_close(out_ckpt.logits, out_plain.logits)
    for out in (out_plain, out_ckpt):
        cache = getattr(out, "past_key_values", None)
        if cache is not None:
            assert cache.get_seq_length() == prompt.shape[1]

    weight = torch.randn_like(out_plain.logits)
    (out_plain.logits * weight).sum().backward()
    (out_ckpt.logits * weight).sum().backward()
    grads_plain = {n: p.grad for n, p in plain.named_parameters() if p.requires_grad}
    grads_ckpt = {
        n: p.grad for n, p in checkpointed.named_parameters() if p.requires_grad
    }
    assert grads_plain.keys() == grads_ckpt.keys()
    assert any("experts" in n for n in grads_plain)
    for name, grad in grads_plain.items():
        assert grad is not None and grads_ckpt[name] is not None, name
        torch.testing.assert_close(grads_ckpt[name], grad, rtol=1e-5, atol=1e-6)

    checkpointed.get_base_model().gradient_checkpointing_disable()
    assert all("forward" not in layer.__dict__ for layer in encoder_layers)


def _near_tie_routers(model):
    """Tie experts 0 and 1 exactly at the top-k boundary (expert 2 is the clear top-1),
    then break the tie in alternate directions on successive router calls: a stand-in
    for a nondeterministic kernel, so the recompute's topk flips relative to the
    forward's unless the forward's is saved. The logits keep a (scaled) dependence on
    the hidden states so gradients still flow through the router input."""
    base = model.get_base_model()
    calls = {"n": 0}

    def hook(_module, _inputs, output):
        calls["n"] += 1
        logits = output * 0.01
        logits = torch.cat((logits[..., :1], logits[..., :1], logits[..., 2:]), -1)
        offset = torch.zeros(output.shape[-1], dtype=output.dtype)
        offset[0] = 1.0 + (1e-4 if calls["n"] % 2 else -1e-4)
        offset[1] = 1.0
        offset[2] = 5.0
        return logits + offset

    for name, module in base.named_modules():
        if name.endswith(".router"):
            module.proj.register_forward_hook(hook)
    return calls


def _grads(model, prompt, canvas, weight):
    out = model(input_ids=prompt, decoder_input_ids=canvas)
    (out.logits * weight).sum().backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


def test_saved_router_topk_keeps_recompute_routing_on_near_ties():
    clear_registered_saves()
    try:
        reference = _model()
        _near_tie_routers(reference)
        reference.train()
        gen = torch.Generator().manual_seed(3)
        prompt = torch.randint(3, 128, (2, 24), generator=gen)
        canvas = torch.randint(3, 128, (2, 16), generator=gen)
        weight = torch.randn(2, 16, 128, generator=gen)
        expected = _grads(reference, prompt, canvas, weight)

        def checkpointed_grads():
            model = _model()
            model.get_base_model().gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            _near_tie_routers(model)
            model.train()
            return _grads(model, prompt, canvas, weight)

        def worst(grads):
            return max(float((grads[n] - g).abs().max()) for n, g in expected.items())

        # The injected tie-break also nudges the recomputed router values by 1e-4,
        # a small gradient change; an expert swap moves them by orders more.
        saved = checkpointed_grads()
        for name, grad in expected.items():
            torch.testing.assert_close(saved[name], grad, rtol=1e-3, atol=1e-4)

        # Without the topk save the recompute takes the other branch of the tie.
        clear_registered_saves()
        import axolotl.model_support.diffusion_gemma.modeling as modeling

        original = modeling._register_router_topk_saves
        modeling._register_router_topk_saves = lambda model: 0
        try:
            unsaved = checkpointed_grads()
        finally:
            modeling._register_router_topk_saves = original
        assert worst(unsaved) > 20 * worst(saved), (worst(unsaved), worst(saved))
        assert worst(unsaved) > 1e-3, "near-tie recompute should swap experts"
    finally:
        clear_registered_saves()


def test_reentrant_checkpointing_is_rejected():
    model = _model()
    try:
        import pytest

        with pytest.raises(ValueError, match="use_reentrant: false"):
            model.get_base_model().gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
    finally:
        clear_registered_saves()


def test_checkpointing_under_autocast_replays_the_same_casts():
    """Autocast's cast cache made the forward and recompute op streams differ."""
    clear_registered_saves()
    try:
        plain = _model()
        checkpointed = copy.deepcopy(plain)
        share_diffusion_gemma_tied_lora(checkpointed)
        checkpointed.get_base_model().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        gen = torch.Generator().manual_seed(4)
        prompt = torch.randint(3, 128, (2, 24), generator=gen)
        canvas = torch.randint(3, 128, (2, 16), generator=gen)
        grads = []
        for model in (plain, checkpointed):
            model.train()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = model(input_ids=prompt, decoder_input_ids=canvas).logits
            logits.float().square().mean().backward()
            grads.append(
                {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
            )
        for name, grad in grads[0].items():
            torch.testing.assert_close(grads[1][name], grad, rtol=2e-2, atol=1e-4)
    finally:
        clear_registered_saves()
