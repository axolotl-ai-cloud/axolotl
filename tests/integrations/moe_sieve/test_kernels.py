"""Optimized kernels must consume compact factors without falling back to PEFT merging."""

import copy

import pytest
import torch
from peft import get_peft_model
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from axolotl.integrations.moe_sieve.peft import (
    MoeSieveLoraConfig,
    SelectiveExpertParamWrapper,
    register_selected_experts,
)
from axolotl.integrations.moe_sieve.selection import packed_experts

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("kernel", ["scattermoe", "sonicmoe"])
@pytest.mark.parametrize("selected", [[0, 1], []])
def test_fused_compact_experts_match_eager(kernel, selected, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(123)
    base = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=384,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=8,
            num_experts_per_tok=2,
            _attn_implementation="eager",
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    selection = {
        name: {"num_experts": 8, "parameter_shapes": shapes, "selected_experts": [0, 1]}
        for name, (_, shapes) in packed_experts(base).items()
    }
    config = MoeSieveLoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[],
        target_parameters=[
            f"{name}.{key}"
            for name, spec in selection.items()
            for key in spec["parameter_shapes"]
        ],
        moe_sieve_selection=selection,
    )
    register_selected_experts(base, config)
    model = get_peft_model(base, config)
    experts = model.base_model.model.model.layers[0].mlp.experts
    for module in experts.modules():
        if isinstance(module, SelectiveExpertParamWrapper):
            if not selected:
                module.selected_experts = ()
                module.lora_A["default"].weight = torch.nn.Parameter(
                    module.lora_A["default"].weight[:0].detach()
                )
                module.lora_B["default"].weight = torch.nn.Parameter(
                    module.lora_B["default"].weight[:, :0].detach()
                )
            torch.nn.init.normal_(module.lora_B["default"].weight, std=0.01)
    reference = copy.deepcopy(experts)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    ids = torch.arange(256, device="cuda").reshape(128, 2) % 8
    weights = torch.softmax(torch.randn(128, 2, device="cuda"), dim=-1).to(
        torch.bfloat16
    )
    expected = reference(reference_x, ids, weights)
    expected.float().square().mean().backward()
    if kernel == "scattermoe":
        from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
            register_scattermoe_experts,
        )

        register_scattermoe_experts()
    else:
        from axolotl.integrations.kernels.libs.sonicmoe.experts import (
            register_sonicmoe_experts,
        )

        register_sonicmoe_experts()
    experts.get_base_layer().config._experts_implementation = kernel

    def forbid_merge(*args, **kwargs):
        raise AssertionError(
            "optimized LoRA path fell back to packed-weight parametrization"
        )

    monkeypatch.setattr(SelectiveExpertParamWrapper, "_activate_lora", forbid_merge)
    actual = experts(x, ids, weights)
    actual.float().square().mean().backward()
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.03)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=2e-6, rtol=0.05)
    reference_params = dict(reference.named_parameters())
    for name, parameter in experts.named_parameters():
        if "lora_" in name:
            assert parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, reference_params[name].grad, atol=2e-6, rtol=0.05
            )
        else:
            assert parameter.grad is None
