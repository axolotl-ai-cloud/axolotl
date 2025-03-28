"""Integration tests for LoRA activation and attention kernels."""

# pylint: disable=redefined-outer-name

import pytest
import torch
from accelerate.state import PartialState
from peft import PeftModelForCausalLM, get_peft_config
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

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

MODEL_CONFIGS = [
    {
        "name": "openaccess-ai-collective/tiny-mistral",
        "expected_activation": apply_lora_mlp_swiglu,
        "dtype": torch.float16,
    },
    {
        "name": "Qwen/Qwen2-7B",
        "expected_activation": apply_lora_mlp_swiglu,
        "dtype": torch.float16,
    },
    {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "expected_activation": apply_lora_mlp_swiglu,
        "dtype": torch.float32,
    },
    {
        "name": "mhenrichsen/gemma-2b",
        "expected_activation": apply_lora_mlp_geglu,
        "dtype": torch.float16,
    },
]


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


def test_attention_patching_integration():
    """Test attention patching in integration context."""
    cfg = {"base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}

    # Store the original implementation
    original_forward = getattr(LlamaAttention, "forward")

    # Apply patch
    patch_self_attn_lora(cfg)

    # Get the new forward method
    patched_forward = LlamaAttention.forward

    # Check the forward method was replaced
    assert original_forward is not patched_forward
    assert patched_forward.__name__ == "axolotl_attn_forward"

    # Check original implementation was stored
    assert hasattr(LlamaAttention, "_original_forward")

    # Clean up
    setattr(LlamaAttention, "forward", original_forward)
    delattr(LlamaAttention, "_original_forward")


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
        "mhenrichsen/gemma-2b", torch_dtype=torch.float16, device_map="auto"
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


def get_lora_config():
    """Get standard LoRA configuration for testing."""
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0,
        "bias": "none",
    }


def get_test_inputs(model, seq_length=20):
    """Generate test inputs for model evaluation."""
    return torch.randint(
        0,
        model.config.vocab_size,
        (1, seq_length),
        device=model.device,
        dtype=torch.long,
    )


@pytest.mark.parametrize("model_config", MODEL_CONFIGS)
def test_model_architecture(model_config):
    """Test LoRA kernel patches across different model architectures."""
    # Load model with appropriate dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"], torch_dtype=model_config["dtype"], device_map="auto"
    )

    # Apply LoRA configuration
    peft_config = get_peft_config(get_lora_config())
    model = PeftModelForCausalLM(model, peft_config)

    # Apply kernel patches
    cfg = DictDefault({"lora_mlp_kernel": True})
    patched_model = apply_lora_kernel_patches(model, cfg)

    # Verify correct activation function
    layer = patched_model.model.model.layers[0]
    assert (
        layer.mlp.forward.__func__ is model_config["expected_activation"]
    ), f"Wrong activation for {model_config['name']}"

    # Test forward pass
    inputs = get_test_inputs(model)
    with torch.no_grad():
        original_output = model(inputs).logits
        patched_output = patched_model(inputs).logits

    # Check outputs match
    assert torch.allclose(
        original_output, patched_output, rtol=1e-4
    ), f"Outputs don't match for {model_config['name']}"


# pylint: disable=duplicate-code
def test_kernel_training_integration():
    """Test model loading with kernel patches enabled."""
    from axolotl.cli.utils import load_model_and_tokenizer

    # Create minimal config
    cfg = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "tokenizer_config": "HuggingFaceTB/SmolLM2-135M",
            "learning_rate": 0.000001,
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                }
            ],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "adapter": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_linear": True,
            "sequence_len": 1024,
            "lora_mlp_kernel": True,
            "lora_qkv_kernel": True,
            "lora_o_kernel": True,
        }
    )

    # Load model
    model, _, _ = load_model_and_tokenizer(cfg=cfg)

    # Verify correct activation function
    layer = model.model.model.layers[0]
    assert layer.mlp.forward.__func__ is apply_lora_mlp_swiglu
