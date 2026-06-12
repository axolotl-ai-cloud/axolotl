"""GPU integration: frozen torchao NVFP4 experts + PEFT expert-LoRA through ScatterMoE.

Exercises the experts-interface fused-LoRA fast-path (no parametrization merge) on a tiny
DiffusionGemma. Requires CUDA + torchao.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required")

DEV = "cuda"


def _tiny_model():
    from transformers import (
        DiffusionGemmaConfig,
        DiffusionGemmaForBlockDiffusion,
        DiffusionGemmaTextConfig,
        Gemma4VisionConfig,
    )

    torch.manual_seed(0)
    tc = DiffusionGemmaTextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=8,
        num_global_key_value_heads=2, sliding_window=8, max_position_embeddings=256,
        num_experts=8, top_k_experts=2, moe_intermediate_size=32,
    )
    vc = Gemma4VisionConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=16)
    cfg = DiffusionGemmaConfig(text_config=tc, vision_config=vc, canvas_length=8)
    return DiffusionGemmaForBlockDiffusion(cfg).to(DEV, torch.bfloat16)


def test_nvfp4_expert_lora_trains_through_scattermoe_fused():
    from peft import LoraConfig, get_peft_model

    from axolotl.integrations.diffusion_gemma.quant_compat import quantize_experts_to_fp4
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        register_scattermoe_experts,
    )

    register_scattermoe_experts()  # installs fast-path + fp4-add patches
    model = _tiny_model()
    model.config.text_config._experts_implementation = "scattermoe"
    quantize_experts_to_fp4(model, "nvfp4")

    cfg = LoraConfig(
        r=4, lora_alpha=8, target_modules=[],
        target_parameters=["experts.gate_up_proj", "experts.down_proj"],
    )
    pm = get_peft_model(model, cfg)
    pm.train()

    inp = torch.randint(0, 128, (1, 6), device=DEV)
    dec = torch.randint(0, 128, (1, 8), device=DEV)
    am = torch.ones(1, 6, dtype=torch.long, device=DEV)

    out = pm(input_ids=inp, attention_mask=am, decoder_input_ids=dec).logits
    assert out.shape == (1, 8, 128)
    out.float().pow(2).mean().backward()

    expert_lora = [p for n, p in pm.named_parameters() if "lora_" in n and "experts" in n]
    assert expert_lora, "no expert LoRA params were created"
    nonzero = sum(1 for p in expert_lora if p.grad is not None and p.grad.abs().sum() > 0)
    assert nonzero > 0, "no gradient reached the expert LoRA through the fused path"


def test_fused_expert_lora_matches_eager_merge():
    """Parity: ScatterMoE fused expert-LoRA == eager (PEFT merge) on bf16 experts.

    bf16 experts isolate the LoRA-fusion path from FP4 quantization noise. The
    reference runs the same weights with the default (grouped_mm) experts, where PEFT
    applies LoRA via its `base + delta` parametrization.
    """
    import copy

    from peft import LoraConfig, get_peft_model

    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        register_scattermoe_experts,
    )

    register_scattermoe_experts()
    base = _tiny_model()
    lcfg = lambda: LoraConfig(  # noqa: E731
        r=4, lora_alpha=8, target_modules=[],
        target_parameters=["experts.gate_up_proj", "experts.down_proj"],
    )

    ref = get_peft_model(copy.deepcopy(base), lcfg())          # default experts (merge)
    fused_base = copy.deepcopy(base)
    fused_base.config.text_config._experts_implementation = "scattermoe"
    fused = get_peft_model(fused_base, lcfg())

    # Make the LoRA non-trivial (B inits to zero) and identical across both models.
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.02)
        ref_lora = {n: p for n, p in ref.named_parameters() if "lora_" in n}
        for n, p in fused.named_parameters():
            if n in ref_lora:
                p.copy_(ref_lora[n])

    inp = torch.randint(0, 128, (1, 6), device=DEV)
    dec = torch.randint(0, 128, (1, 8), device=DEV)
    am = torch.ones(1, 6, dtype=torch.long, device=DEV)

    def run(model):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=inp, attention_mask=am, decoder_input_ids=dec).logits
        out.float().pow(2).mean().backward()
        grads = {n: p.grad.float().clone() for n, p in model.named_parameters()
                 if "lora_" in n and "experts" in n and p.grad is not None}
        return out.float(), grads

    o_ref, g_ref = run(ref)
    o_fused, g_fused = run(fused)
    assert torch.allclose(o_ref, o_fused, atol=2e-2, rtol=2e-2), (
        f"forward mismatch: max |Δ| = {(o_ref - o_fused).abs().max().item()}"
    )
    assert g_ref and g_ref.keys() == g_fused.keys()
    for n in g_ref:
        assert torch.allclose(g_ref[n], g_fused[n], atol=2e-2, rtol=2e-2), f"grad mismatch on {n}"
