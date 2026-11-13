"""GPU check: Gated DeltaNet trees run on TRL's Triton kernels (``patch_tree_gdn``),
which async GRPO's packing picks, score every sample as it scores alone.

TRL's own TestTreePackingParity checks those kernels' parity on its Qwen3.5 models,
so this end-to-end run through axolotl's packing_support is a slow test.
"""

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture
def tiny_qwen3_5(monkeypatch):
    pytest.importorskip("triton")
    from transformers import (
        AttentionInterface,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
    )
    from transformers.masking_utils import AttentionMaskInterface
    from trl.experimental.async_grpo import packing

    from tests.core.async_test_utils import reference_tree_attention

    for registry in (AttentionInterface, AttentionMaskInterface):
        monkeypatch.setattr(registry, "_global_mapping", dict(registry._global_mapping))
    # FA kernels need half precision; the reference keeps the check on the GDN layers.
    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
    )
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(config).cuda()


def test_trls_tree_gated_delta_net_matches_samples_alone(tiny_qwen3_5):
    from peft import LoraConfig, get_peft_model

    from axolotl.monkeypatch.models.tree_packing import packing_support

    from tests.core.async_test_utils import (
        assert_tree_rows_match_alone,
        branching_samples,
    )

    # As axolotl hands it over: wrapped by PEFT.
    peft_model = get_peft_model(
        tiny_qwen3_5, LoraConfig(r=4, target_modules=["q_proj"])
    )
    support = packing_support(peft_model)
    assert support.mode == "tree"
    assert support.owners == {"Qwen3_5GatedDeltaNet": "trl"}
    peft_model.unload().requires_grad_(True)
    shared_len, chain_len = assert_tree_rows_match_alone(
        tiny_qwen3_5, branching_samples(), rtol=1e-3, atol=1e-3
    )
    assert shared_len < chain_len
