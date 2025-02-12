"""Integration tests for LoRA activation and attention kernels."""
# pylint: disable=redefined-outer-name

import inspect

import pytest
import torch
from accelerate.state import PartialState
from peft import PeftModelForCausalLM, get_peft_config
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from axolotl.kernels.lora import (
    apply_lora_mlp_geglu,
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
)
from axolotl.monkeypatch.lora_kernels import (
    apply_lora_kernel_patches,
    patch_self_attn_lora,
)
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


# pylint: disable=protected-access
def test_attention_patching_integration(small_llama_model):
    """Test attention patching in integration context."""
    peft_config = get_peft_config(
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0,
            "bias": "none",
        }
    )
    model = PeftModelForCausalLM(small_llama_model, peft_config).to("cuda")

    # Get the attention class and original implementation
    attn_cls = type(model.model.model.layers[0].self_attn)
    original_forward = attn_cls.forward
    original_code = inspect.getsource(original_forward)

    # Apply patch
    patch_self_attn_lora(attn_cls)

    # Check the forward method was replaced
    assert attn_cls.forward != original_forward
    assert attn_cls.forward.__name__ == "axolotl_attn_forward"

    # Check original implementation was stored
    assert hasattr(attn_cls, "_original_forward")
    assert attn_cls._original_forward == original_code

    # Verify patched implementation is applied
    patched_forward = attn_cls.forward
    assert patched_forward.__name__ == "axolotl_attn_forward"

    # Clean up
    attn_cls.forward = original_forward
    delattr(attn_cls, "_original_forward")


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

    # Verify patches
    layer = patched_model.model.model.layers[0]
    assert layer.mlp.forward.__func__ is apply_lora_mlp_swiglu

    # Test forward pass
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(
        batch_size, seq_len, model.config.hidden_size, device=model.device
    )
    position_ids = (
        torch.arange(seq_len, device=model.device).unsqueeze(0).expand(batch_size, -1)
    )
    cos, sin = model.model.model.rotary_emb(hidden_states, position_ids)

    inputs = {
        "hidden_states": hidden_states,
        "attention_mask": None,
        "position_embeddings": (cos, sin),
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

    # Verify patches
    layer = patched_model.model.model.layers[0]
    assert layer.mlp.forward.__func__ is apply_lora_mlp_geglu

    # Test end-to-end
    inputs = torch.randint(0, 100, (1, 20), device=model.device, dtype=torch.long)
    with torch.no_grad():
        original_output = model(inputs).logits
        patched_output = patched_model(inputs).logits

    assert torch.allclose(original_output, patched_output, rtol=1e-4)


@pytest.mark.parametrize(
    "model_name,expected_activation",
    [
        ("HuggingFaceTB/SmolLM2-135M", apply_lora_mlp_swiglu),
        ("mhenrichsen/gemma-2b", apply_lora_mlp_geglu),
    ],
)
def test_model_specific_activation(model_name, expected_activation):
    """Test that each model type gets the correct activation function."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
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
    layer = patched_model.model.model.layers[0]
    assert layer.mlp.forward.__func__ is expected_activation


def test_kernel_patch_conditions():
    """Test various conditions that should prevent kernel patching."""
    test_configs = [
        # Dropout prevents patching
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
        },
        # Bias prevents patching
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0,
            "bias": "lora_only",
        },
    ]

    for config in test_configs:
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        peft_config = get_peft_config(config)
        model = PeftModelForCausalLM(model, peft_config)
        cfg = DictDefault({"lora_mlp_kernel": True})

        # Should not patch
        patched_model = apply_lora_kernel_patches(model, cfg)
        layer = patched_model.model.model.layers[0].mlp

        # Verify no patches applied
        assert layer.forward.__func__ is not apply_lora_mlp_swiglu
        assert layer.forward.__func__ is not apply_lora_mlp_geglu


def test_kernel_config_options():
    """Test that kernel configuration options are respected."""
    # Test different configurations
    test_configs = [
        (
            {"lora_mlp_kernel": True, "lora_qkv_kernel": False, "lora_o_kernel": False},
            lambda layer: (
                layer.mlp.forward.__func__ is apply_lora_mlp_swiglu
                and layer.self_attn.apply_qkv.__func__ is not apply_lora_qkv
                and layer.self_attn.apply_o.__func__ is not apply_lora_o
            ),
        ),
        (
            {"lora_mlp_kernel": False, "lora_qkv_kernel": True, "lora_o_kernel": False},
            lambda layer: (
                layer.mlp.forward.__func__ is not apply_lora_mlp_swiglu
                and layer.self_attn.apply_qkv.__func__ is apply_lora_qkv
                and layer.self_attn.apply_o.__func__ is not apply_lora_o
            ),
        ),
        (
            {"lora_mlp_kernel": False, "lora_qkv_kernel": False, "lora_o_kernel": True},
            lambda layer: (
                layer.mlp.forward.__func__ is not apply_lora_mlp_swiglu
                and layer.self_attn.apply_qkv.__func__ is not apply_lora_qkv
                and layer.self_attn.apply_o.__func__ is apply_lora_o
            ),
        ),
    ]

    for config_dict, check_fn in test_configs:
        # Create fresh model for each test
        config = {
            "vocab_size": 100,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
        }
        small_llama_model = LlamaForCausalLM(LlamaConfig(**config))

        peft_config = get_peft_config(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ],
                "lora_dropout": 0,
                "bias": "none",
            }
        )
        model = PeftModelForCausalLM(small_llama_model, peft_config).to("cuda")
        cfg = DictDefault(config_dict)
        patched_model = apply_lora_kernel_patches(model, cfg)

        # Verify only requested optimizations were applied
        for layer in patched_model.model.model.layers:
            assert check_fn(layer), f"Failed for config: {config_dict}"

        # Clean up
        del model
        del small_llama_model
        del patched_model
