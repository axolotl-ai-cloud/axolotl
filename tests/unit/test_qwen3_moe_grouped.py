from __future__ import annotations

import copy
import torch

from axolotl.monkeypatch.models.qwen3_moe.modeling import (
    patch_qwen3_moe_grouped_experts,
)


def _load_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "tiny-random/qwen3-moe", trust_remote_code=True
    )
    model.eval()
    return model


def test_qwen3_grouped_parity():
    torch.manual_seed(42)
    base_model = _load_model()
    grouped_model = copy.deepcopy(base_model)

    patched = patch_qwen3_moe_grouped_experts(grouped_model, mlp_impl="grouped")
    assert patched > 0

    input_ids = torch.randint(
        0, base_model.config.vocab_size, (2, 4), dtype=torch.long
    )

    with torch.no_grad():
        vanilla_logits = base_model(input_ids=input_ids).logits
        grouped_logits = grouped_model(input_ids=input_ids).logits

    assert torch.allclose(
        vanilla_logits, grouped_logits, atol=1e-5, rtol=1e-5
    ), "Grouped MoE output deviates from vanilla Qwen3 MoE"
