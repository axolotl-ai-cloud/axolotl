"""Mistral Large 4 LoRA on a block-FP8 base through transformers' FineGrainedFP8 layers: an fp8
forward (activations quantized 1x128) and a dequantized-weight backward for dX.

The reference is a dequantized-bf16 forward on every ``FP8Linear`` (what Axolotl patched in before
transformers made the layers trainable). Both share the backward formula, so the gradients differ
only through the fp8 forward's activation quantization. The fused MLA attention-LoRA kernels are
checked against plain PEFT on the same base.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
        reason="CUDA with FP8 support (sm89+) required",
    ),
]

# 128-aligned so every FP8 projection is whole 128x128 blocks
TEXT = dict(
    vocab_size=512,
    hidden_size=256,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=2,
    # every expert per token and (below) every key per query: no discrete routing / top-k choice
    # for rounding to flip, so the whole-model comparisons stay continuous
    n_routed_experts=4,
    num_experts_per_tok=4,
    kv_lora_rank=128,
    qk_rope_head_dim=64,
    qk_nope_head_dim=64,
    v_head_dim=128,
    index_topk=2048,
    index_head_dim=128,
    index_n_heads=2,
    first_k_dense_replace=1,
    kv_source_layers=[0, 1, 0, 3],
    max_position_embeddings=4096,
    pad_token_id=0,
)
VISION = dict(
    model_type="pixtral",
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    patch_size=14,
    image_size=28,
    head_dim=16,
)
# bf16 in the block-FP8 checkpoint
NOT_CONVERTED = [
    "model.vision_tower",
    "model.multi_modal_projector",
    "lm_head",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "indexer.wk",
    "indexer.weights_proj",
]
FP8_LINEARS = (
    "self_attn.q_proj",
    "self_attn.o_proj",
    "indexer.wq_b",
    "shared_experts.gate_proj",
    "shared_experts.up_proj",
    "shared_experts.down_proj",
    "0.mlp.gate_proj",
    "0.mlp.up_proj",
    "0.mlp.down_proj",
)
LORA_TARGETS = (
    r".*language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)"
    r"|mlp\.(shared_experts\.)?(gate_proj|up_proj|down_proj))"
)


def _rel(got, want):
    got, want = got.float(), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def _dequant_forward(self, x):
    from axolotl.kernels.quantize import dequantize_fp8

    if self.weight.element_size() > 1:
        return F.linear(x, self.weight, self.bias)
    w = dequantize_fp8(self.weight, self.weight_scale_inv, x.dtype)
    return F.linear(x, w, self.bias)


@pytest.fixture(scope="module")
def bf16_dir(tmp_path_factory):
    from transformers import AutoModelForImageTextToText, MistralLarge4Config

    torch.manual_seed(0)
    config = MistralLarge4Config(text_config=TEXT, vision_config=VISION)
    path = tmp_path_factory.mktemp("ml4_bf16")
    model = AutoModelForImageTextToText.from_config(config).to(torch.bfloat16)
    model.save_pretrained(path)
    return str(path)


def _load_fp8(bf16_dir):
    from transformers import AutoModelForImageTextToText, FineGrainedFP8Config

    return AutoModelForImageTextToText.from_pretrained(
        bf16_dir,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        quantization_config=FineGrainedFP8Config(
            weight_block_size=(128, 128), modules_to_not_convert=NOT_CONVERTED
        ),
    )


def _peft(model):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model, LoraConfig(r=8, lora_alpha=16, target_modules=LORA_TARGETS)
    )
    gen = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.02)
    return model


def _step(model, input_ids):
    model.zero_grad(set_to_none=True)
    embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_()
    out = model(inputs_embeds=embeds, labels=input_ids)
    out.loss.backward()
    grads = {
        n: p.grad.float().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return out.logits.detach().float(), out.loss.detach().float(), grads, embeds.grad


def _input_ids():
    gen = torch.Generator(device="cpu").manual_seed(1)
    return torch.randint(1, TEXT["vocab_size"], (2, 64), generator=gen).cuda()


def test_block_fp8_layout(bf16_dir):
    from transformers.integrations.finegrained_fp8 import FP8Experts, FP8Linear

    model = _load_fp8(bf16_dir)
    modules = dict(model.named_modules())
    fp8 = {n for n, m in modules.items() if isinstance(m, FP8Linear)}
    for suffix in FP8_LINEARS:
        assert any(n.endswith(suffix) for n in fp8), suffix
    assert not any("vision_tower" in n or n.endswith("kv_b_proj") for n in fp8)
    assert all(
        isinstance(m, FP8Experts)
        for n, m in modules.items()
        if n.endswith("mlp.experts")
    )


def test_native_fp8_matches_dequant_forward(bf16_dir, monkeypatch):
    from transformers.integrations.finegrained_fp8 import FP8Linear

    model = _peft(_load_fp8(bf16_dir))
    input_ids = _input_ids()
    logits, loss, grads, d_embeds = _step(model, input_ids)
    monkeypatch.setattr(FP8Linear, "forward", _dequant_forward)
    ref_logits, ref_loss, ref_grads, ref_d_embeds = _step(model, input_ids)

    lora = sorted(n for n in ref_grads if "lora_" in n)
    worst = max(lora, key=lambda n: _rel(grads[n], ref_grads[n]))
    print(
        f"\nlogits rel {_rel(logits, ref_logits):.3e} | loss {loss.item():.6f} vs "
        f"{ref_loss.item():.6f} | dX rel {_rel(d_embeds, ref_d_embeds):.3e} | "
        f"worst LoRA grad rel {_rel(grads[worst], ref_grads[worst]):.3e} ({worst})"
    )
    assert grads.keys() == ref_grads.keys()
    assert len(lora) == 2 * sum(
        1 for n in dict(model.named_modules()) if n.endswith(".lora_A.default")
    )
    assert _rel(logits, ref_logits) < 8e-2
    assert abs(loss.item() - ref_loss.item()) < 1e-2
    assert _rel(d_embeds, ref_d_embeds) < 8e-2
    for name in lora:
        assert _rel(grads[name], ref_grads[name]) < 1.5e-1, name


@pytest.mark.parametrize("suffix", FP8_LINEARS)
def test_fp8_linear_dx_matches_dequant_forward(bf16_dir, monkeypatch, suffix):
    from transformers.integrations.finegrained_fp8 import FP8Linear

    model = _load_fp8(bf16_dir)
    layer = next(
        m
        for n, m in model.named_modules()
        if n.endswith(suffix) and isinstance(m, FP8Linear)
    )
    gen = torch.Generator(device="cpu").manual_seed(2)
    x = torch.randn(2, 64, layer.in_features, generator=gen).cuda().bfloat16()
    grad = torch.randn(2, 64, layer.out_features, generator=gen).cuda().bfloat16()

    def run():
        xi = x.clone().requires_grad_()
        out = layer(xi)
        out.backward(grad)
        return out.detach(), xi.grad

    out, dx = run()
    monkeypatch.setattr(FP8Linear, "forward", _dequant_forward)
    ref_out, ref_dx = run()
    print(f"\n{suffix}: out rel {_rel(out, ref_out):.3e} dX rel {_rel(dx, ref_dx):.3e}")
    # fp8 forward quantizes the activations 1x128
    assert _rel(out, ref_out) < 4e-2
    # same dequantized-weight backward on both sides
    assert _rel(dx, ref_dx) < 1e-3


@pytest.mark.parametrize("native_fp8", [True, False], ids=["native_fp8", "dequant"])
def test_attn_lora_kernel_matches_peft_on_fp8(bf16_dir, native_fp8, monkeypatch):
    from axolotl.integrations.kernels.libs.dsv4 import lora_fp8
    from axolotl.integrations.kernels.libs.mistral_large4 import (
        patch_mistral_large4_attn_lora,
    )

    model = _peft(_load_fp8(bf16_dir))
    input_ids = _input_ids()
    ref_logits, ref_loss, ref_grads, ref_d_embeds = _step(model, input_ids)

    # q/kv_b/o on all 4 layers, kv_a on the 3 KV-source layers
    assert patch_mistral_large4_attn_lora(model, native_fp8=native_fp8) == 15
    native_calls = []
    orig_apply = lora_fp8._NativeFp8Lora.apply

    def counting_apply(*args):
        native_calls.append(1)
        return orig_apply(*args)

    monkeypatch.setattr(lora_fp8._NativeFp8Lora, "apply", counting_apply)
    logits, loss, grads, d_embeds = _step(model, input_ids)

    lora = sorted(n for n in ref_grads if "lora_" in n)
    worst = max(lora, key=lambda n: _rel(grads[n], ref_grads[n]))
    print(
        f"\nnative_fp8={native_fp8}: logits rel {_rel(logits, ref_logits):.3e} | dX rel "
        f"{_rel(d_embeds, ref_d_embeds):.3e} | worst LoRA grad rel "
        f"{_rel(grads[worst], ref_grads[worst]):.3e} ({worst})"
    )
    # q_proj + o_proj take the native fp8 GEMM; kv_a / kv_b stay bf16
    assert len(native_calls) == (8 if native_fp8 else 0)
    assert grads.keys() == ref_grads.keys()
    assert _rel(logits, ref_logits) < 8e-2
    assert abs(loss.item() - ref_loss.item()) < 1e-2
    assert _rel(d_embeds, ref_d_embeds) < 8e-2
    for name in lora:
        assert _rel(grads[name], ref_grads[name]) < 1.5e-1, name
