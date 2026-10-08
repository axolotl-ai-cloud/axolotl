"""DiffusionGemma routed experts: ScatterMoE (+ fused expert LoRA) vs the stock experts.

DiffusionGemma ties its encoder language model and decoder weights, so one PEFT
``target_parameters`` adapter per expert tensor is shared by both passes
(``share_diffusion_gemma_tied_lora``). These tests run the same tiny model through
transformers' stock ``grouped_mm`` experts and through ScatterMoE with the
``ParamWrapper`` fast path, forward and backward, with nonzero LoRA on attention,
dense MLP and both expert tensors, and with the router frozen.
"""

from __future__ import annotations

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import DiffusionGemmaConfig

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover - CPU CI
    pytest.skip("ScatterMoE needs CUDA", allow_module_level=True)

from axolotl.integrations.kernels.libs.scattermoe_lora import (  # noqa: E402
    experts as smoe_experts,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (  # noqa: E402
    register_scattermoe_experts,
)
from axolotl.model_support.diffusion_gemma.modeling import (  # noqa: E402
    AxolotlDiffusionGemmaForBlockDiffusion,
)
from axolotl.model_support.native_adapters import (  # noqa: E402
    share_diffusion_gemma_tied_lora,
)

_TARGET = (
    r".*\.(encoder\.language_model|decoder)\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)


def _config() -> DiffusionGemmaConfig:
    return DiffusionGemmaConfig(
        text_config={
            "vocab_size": 512,
            "hidden_size": 256,
            "intermediate_size": 384,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 64,
            "global_head_dim": 128,
            "sliding_window": 32,
            "num_experts": 16,
            "top_k_experts": 4,
            "max_position_embeddings": 512,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
            "final_logit_softcapping": 30.0,
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 64,
            "patch_size": 16,
            "position_embedding_size": 16,
        },
        canvas_length=32,
    )


def _peft(model, expert_rank):
    peft = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=_TARGET,
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            rank_pattern={r"experts\.(gate_up_proj|down_proj)": expert_rank},
            alpha_pattern={r"experts\.(gate_up_proj|down_proj)": expert_rank},
            lora_dropout=0.0,
            task_type=None,
        ),
    )
    share_diffusion_gemma_tied_lora(peft)
    return peft


def _pair(dtype, expert_rank=4):
    register_scattermoe_experts()
    torch.manual_seed(0)
    base = AxolotlDiffusionGemmaForBlockDiffusion(_config())
    with torch.no_grad():  # non-trivial router + experts
        for name, param in base.named_parameters():
            if "per_expert_scale" in name or name.endswith("router.scale"):
                param.uniform_(0.5, 1.5)
            elif "experts" in name:
                param.normal_(std=0.05)
    stock = base.to("cuda", dtype)
    fused = copy.deepcopy(stock)
    stock.set_experts_implementation("grouped_mm")
    fused.set_experts_implementation("scattermoe")
    stock, fused = _peft(stock, expert_rank), _peft(fused, expert_rank)
    torch.manual_seed(1)
    with torch.no_grad():
        for (name, a), (_, b) in zip(
            stock.named_parameters(), fused.named_parameters(), strict=True
        ):
            if "lora_" in name:
                a.normal_(std=0.05)
                b.copy_(a)
    return stock, fused


def _batch():
    gen = torch.Generator().manual_seed(2)
    prompt = torch.randint(3, 512, (2, 96), generator=gen).cuda()
    canvas = torch.randint(3, 512, (2, 32), generator=gen).cuda()
    return prompt, canvas


def _pin_routing(model):
    """Route by token position instead of by score, keeping the router's own weights.

    Top-k over near-tied router probabilities flips between two numerically
    different (both correct) expert kernels — e.g. 0.0631674 vs 0.0631673 at the
    k boundary — which swaps an expert for one token and moves the logits by
    ~0.2. Pinning the expert set isolates the experts path (base + fused LoRA)
    while every routing weight still comes from the frozen router.
    """

    def hook(module, _inputs, output):
        probabilities = output[0]
        tokens, experts = probabilities.shape
        top_k = output[2].shape[-1]
        position = torch.arange(tokens, device=probabilities.device)[:, None]
        index = (
            position * 5 + torch.arange(top_k, device=position.device) * 3
        ) % experts
        weights = probabilities.gather(-1, index)
        weights = weights / weights.sum(-1, keepdim=True)
        weights = weights * module.per_expert_scale[index]
        return probabilities, weights, index

    for name, module in model.named_modules():
        if name.endswith(".router"):
            module.register_forward_hook(hook)


def _run(model, prompt, canvas):
    return model(input_ids=prompt, decoder_input_ids=canvas).logits.float()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scattermoe_matches_stock_experts_forward_backward_with_lora(
    dtype, monkeypatch
):
    stock, fused = _pair(dtype)
    calls = {"fused_lora": 0}
    original = smoe_experts.scattermoe_experts_forward

    def counting(self, *args, **kwargs):
        if getattr(self, "_scattermoe_lora", None):
            calls["fused_lora"] += 1
        return original(self, *args, **kwargs)

    for registry_owner in (fused,):
        for module in registry_owner.modules():
            if type(module).__name__ == "DiffusionGemmaTextExperts":
                assert module.config._experts_implementation == "scattermoe"
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    monkeypatch.setitem(ALL_EXPERTS_FUNCTIONS._global_mapping, "scattermoe", counting)
    import inspect

    from transformers.integrations import use_experts_implementation

    canon = (
        inspect.signature(use_experts_implementation)
        .parameters["experts_interface"]
        .default
    )
    if canon is not None:
        monkeypatch.setitem(canon._global_mapping, "scattermoe", counting)

    _pin_routing(stock)
    _pin_routing(fused)
    prompt, canvas = _batch()
    ref = _run(stock, prompt, canvas)
    out = _run(fused, prompt, canvas)
    # 2 layers x (encoder + decoder) experts calls, all through the fused LoRA path
    assert calls["fused_lora"] == 4

    weight = torch.randn_like(ref)
    (ref * weight).sum().backward()
    (out * weight).sum().backward()
    grads_ref = {
        n: p.grad.float() for n, p in stock.named_parameters() if p.requires_grad
    }
    grads_out = {
        n: p.grad.float() for n, p in fused.named_parameters() if p.requires_grad
    }
    assert grads_ref.keys() == grads_out.keys()
    assert any("experts" in n for n in grads_ref)
    assert not any("router" in n for n in grads_ref)

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    worst = 0.0
    if dtype is torch.bfloat16:
        # Judge bf16 against an fp32 reference of the same model: the fused path
        # may not be noisier than stock bf16 itself (single logits reach ~0.04 and
        # single LoRA grads ~6% relative in either).
        exact, _ = _pair(torch.float32)
        _pin_routing(exact)
        gold = _run(exact, prompt, canvas)
        (gold * weight.float()).sum().backward()
        stock_err = (ref - gold.detach()).abs()
        fused_err = (out - gold.detach()).abs()
        assert float(fused_err.max()) <= 2 * float(stock_err.max()) + 1e-3
        assert float(fused_err.mean()) <= 1.5 * float(stock_err.mean()) + 1e-4
        grads_gold = {
            n: p.grad.float() for n, p in exact.named_parameters() if p.requires_grad
        }
        for name, g_gold in grads_gold.items():
            stock_rel = rel(grads_ref[name], g_gold)
            fused_rel = rel(grads_out[name], g_gold)
            worst = max(worst, fused_rel)
            assert fused_rel <= 2 * stock_rel + 5e-3, (name, fused_rel, stock_rel)
    else:
        # ScatterMoE's Triton dots run fp32 inputs in TF32 (~1e-3 relative).
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        for name, g_ref in grads_ref.items():
            r = rel(grads_out[name], g_ref)
            worst = max(worst, r)
            assert r < 1e-2, (name, r)
    print(
        f"{dtype}: max|dlogits| vs stock={float((out - ref).abs().max()):.3e} "
        f"worst grad rel (vs {'fp32 gold' if dtype is torch.bfloat16 else 'stock'})"
        f"={worst:.3e} over {len(grads_ref)} trainable tensors"
    )
    # router stays frozen and gradient-free in both paths
    for model in (stock, fused):
        for name, param in model.named_parameters():
            if "router" in name:
                assert not param.requires_grad and param.grad is None, name
