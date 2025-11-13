from __future__ import annotations

import copy
import torch

from axolotl.monkeypatch.models.glm4_moe.modeling import (
    patch_glm4_moe_grouped_experts,
)


def _load_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "tiny-random/glm-4-moe",
        trust_remote_code=True,
    )
    model.eval()
    return model


def test_glm4_grouped_parity():
    torch.manual_seed(123)
    base_model = _load_model()
    grouped_model = copy.deepcopy(base_model)

    patched = patch_glm4_moe_grouped_experts(grouped_model, mlp_impl="grouped")
    assert patched > 0

    input_ids = torch.randint(
        0,
        base_model.config.vocab_size,
        (2, 8),
        dtype=torch.long,
    )

    with torch.no_grad():
        vanilla_logits = base_model(input_ids=input_ids).logits
        grouped_logits = grouped_model(input_ids=input_ids).logits

    assert torch.allclose(
        vanilla_logits,
        grouped_logits,
        atol=1e-5,
        rtol=1e-5,
    ), "Grouped MoE output deviates from vanilla GLM4 MoE"
