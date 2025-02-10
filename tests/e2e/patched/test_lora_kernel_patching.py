"""Integration tests for LoRA activation kernels."""
# pylint: disable=redefined-outer-name

import pytest
import torch
from accelerate.state import PartialState
from peft import PeftModelForCausalLM, get_peft_config
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.testing_utils import require_read_token

from axolotl.kernels.lora import apply_lora_mlp_geglu, apply_lora_mlp_swiglu
from axolotl.monkeypatch.lora_kernels import apply_lora_kernel_patches
from axolotl.utils.dict import DictDefault


@pytest.fixture(autouse=True)
def init_accelerate():
    """Initialize Accelerate state before tests."""
    _ = PartialState()


@pytest.fixture
def small_llama_model():
    """Create a small LLaMA model for testing."""
    config = {
        "vocab_size": 100,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
    }

    return LlamaForCausalLM(LlamaConfig(**config))


def test_swiglu_mlp_integration(small_llama_model):
    """Test SwiGLU activation in LoRA MLP context."""
    peft_config = get_peft_config(
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0,
            "bias": "none",
        }
    )
    model = PeftModelForCausalLM(small_llama_model, peft_config).to("cuda")
    cfg = DictDefault({"lora_mlp_kernel": True})

    # Apply patches
    patched_model = apply_lora_kernel_patches(model, cfg)

    # Get a sample layer
    layer = patched_model.model.model.layers[0]
    assert layer.mlp.forward.__func__ is apply_lora_mlp_swiglu

    # Test forward pass
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(
        batch_size, seq_len, model.config.hidden_size, device=model.device
    )

    # Create position IDs and embeddings
    position_ids = (
        torch.arange(seq_len, device=model.device).unsqueeze(0).expand(batch_size, -1)
    )
    cos, sin = model.model.model.rotary_emb(hidden_states, position_ids)
    position_embeddings = (cos, sin)

    # Prepare inputs for both models
    inputs = {
        "hidden_states": hidden_states,
        "attention_mask": None,
        "position_embeddings": position_embeddings,
        "output_attentions": False,
        "use_cache": False,
        "past_key_value": None,
    }

    # Compare outputs
    with torch.no_grad():
        original_output = model.model.model.layers[0](**inputs)[0]
        patched_output = layer(**inputs)[0]

    assert torch.allclose(original_output, patched_output, rtol=1e-4)


def test_geglu_model_integration():
    """Test GeGLU activation with Gemma model."""
    model = AutoModelForCausalLM.from_pretrained(
        "mhenrichsen/gemma-2b", torch_dtype=torch.float16, device_map="cuda"
    )
    peft_config = get_peft_config(
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0,
            "bias": "none",
        }
    )
    model = PeftModelForCausalLM(model, peft_config)

    cfg = DictDefault({"lora_mlp_kernel": True})
    patched_model = apply_lora_kernel_patches(model, cfg)

    # Get a sample layer
    layer = patched_model.model.model.layers[0]
    assert layer.mlp.forward.__func__ is apply_lora_mlp_geglu

    # Test input
    inputs = torch.randint(0, 100, (1, 20), device=model.device, dtype=torch.long)

    # Compare outputs
    with torch.no_grad():
        original_output = model(inputs).logits
        patched_output = patched_model(inputs).logits

    assert torch.allclose(original_output, patched_output, rtol=1e-4)


@require_read_token
@pytest.mark.parametrize("model_type", ["llama", "gemma"])
def test_kernel_patch_conditions(model_type):
    """Test conditions for kernel patching."""
    if model_type == "llama":
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    else:
        model = AutoModelForCausalLM.from_pretrained("mhenrichsen/gemma-2b")

    # Test with invalid conditions
    peft_config = get_peft_config(
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,  # Should prevent patching
            "bias": "lora_only",  # Should prevent patching
        }
    )
    model = PeftModelForCausalLM(model, peft_config)
    cfg = DictDefault({"lora_mlp_kernel": True})

    # Should log warnings and not patch
    patched_model = apply_lora_kernel_patches(model, cfg)

    # Verify no patches applied
    layer = patched_model.model.model.layers[0].mlp
    assert layer.forward.__func__ is not apply_lora_mlp_swiglu
    assert layer.forward.__func__ is not apply_lora_mlp_geglu
