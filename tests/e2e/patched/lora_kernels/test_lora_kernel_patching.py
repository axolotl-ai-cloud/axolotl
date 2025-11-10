"""Integration tests for LoRA activation and attention kernels."""

from pathlib import Path

import pytest
import torch
import yaml
from accelerate.state import PartialState
from peft import PeftModelForCausalLM, get_peft_config
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from axolotl.cli.config import load_cfg
from axolotl.kernels.lora import (
    apply_lora_mlp_geglu,
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
)
from axolotl.loaders.model import ModelLoader
from axolotl.loaders.tokenizer import load_tokenizer
from axolotl.monkeypatch.lora_kernels import (
    apply_lora_kernel_patches,
    find_self_attn_in_layer,
    get_attention_cls_from_config,
    get_layers,
    patch_self_attn_lora,
)
from axolotl.utils.dict import DictDefault

MODEL_CONFIGS = [
    {
        "name": "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        "expected_activation": apply_lora_mlp_swiglu,
        "dtype": torch.float16,
    },
    {
        "name": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "expected_activation": apply_lora_mlp_swiglu,
        "dtype": torch.float16,
    },
    {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "expected_activation": apply_lora_mlp_swiglu,
        "dtype": torch.float32,
    },
    {
        "name": "trl-internal-testing/tiny-Gemma2ForCausalLM",
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


@pytest.mark.parametrize(
    "model_name,attention_cls",
    [
        ("HuggingFaceTB/SmolLM2-135M", LlamaAttention),
        ("Qwen/Qwen3-30B-A3B", Qwen3MoeAttention),
    ],
)
def test_attention_patching_integration(model_name, attention_cls):
    """Test attention patching in integration context."""
    cfg = DictDefault({"base_model": model_name})

    # Store the original implementation
    original_forward = attention_cls.forward

    # Apply patch
    patch_self_attn_lora(cfg)

    # Get the new forward method
    patched_forward = attention_cls.forward

    # Check the forward method was replaced
    assert original_forward is not patched_forward
    assert patched_forward.__name__ == "axolotl_attn_forward"

    # Check original implementation was stored
    assert hasattr(attention_cls, "_original_forward")

    # Clean up
    attention_cls.forward = original_forward
    delattr(attention_cls, "_original_forward")


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
        "trl-internal-testing/tiny-Gemma2ForCausalLM",
        dtype=torch.float16,
        device_map="cuda:0",
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
        model_config["name"], torch_dtype=model_config["dtype"], device_map="cuda:0"
    )

    # Apply LoRA configuration
    peft_config = get_peft_config(get_lora_config())
    model = PeftModelForCausalLM(model, peft_config)

    # Apply kernel patches
    cfg = DictDefault({"lora_mlp_kernel": True})
    patched_model = apply_lora_kernel_patches(model, cfg)

    # Verify correct activation function
    layer = patched_model.model.model.layers[0]
    assert layer.mlp.forward.__func__ is model_config["expected_activation"], (
        f"Wrong activation for {model_config['name']}"
    )

    # Test forward pass
    inputs = get_test_inputs(model)
    with torch.no_grad():
        original_output = model(inputs).logits
        patched_output = patched_model(inputs).logits

    # Check outputs match
    assert torch.allclose(original_output, patched_output, rtol=1e-4), (
        f"Outputs don't match for {model_config['name']}"
    )


def test_kernel_training_integration(temp_dir):
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

    # Write cfg to yaml file
    path = Path(temp_dir) / "config.yaml"
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

    # Load config
    cfg = load_cfg(str(path))

    # Load model
    model, _, _ = load_model_and_tokenizer(cfg=cfg)

    # Verify correct activation function
    layer = model.model.model.layers[0]
    assert layer.mlp.forward.__func__ is apply_lora_mlp_swiglu


def test_kernel_training_integration_auto_enable(temp_dir):
    """Test model loading with auto-enabled kernel patches."""
    # Create minimal config without explicitly setting kernel options
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
        }
    )

    # Write cfg to yaml file
    path = Path(temp_dir) / "config.yaml"
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

    # Load config
    cfg = load_cfg(str(path))

    # Verify kernel options were auto-enabled in the config
    assert cfg.lora_mlp_kernel is True
    assert cfg.lora_qkv_kernel is True
    assert cfg.lora_o_kernel is True

    # Get the attention class before patching to check for side effects
    attention_cls = get_attention_cls_from_config(cfg)

    # Store original state before patching
    original_forward_method = attention_cls.forward

    # Load the model (this should trigger the patches)
    tokenizer = load_tokenizer(cfg)
    model, _ = ModelLoader(cfg, tokenizer).load()

    # Test side effects of patch_self_attn_lora
    assert hasattr(attention_cls, "_original_forward")
    assert attention_cls.forward != original_forward_method

    # Find at least one self-attention module and verify it has the patched methods
    found_patched_attn = False
    for layer in model.model.model.layers:
        if hasattr(layer, "self_attn"):
            self_attn = layer.self_attn
            if all(
                hasattr(self_attn, proj)
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ):
                # These methods should be added by apply_lora_kernel_patches
                assert hasattr(self_attn, "apply_qkv") and callable(self_attn.apply_qkv)
                assert hasattr(self_attn, "apply_o") and callable(self_attn.apply_o)

                found_patched_attn = True
                break

    assert found_patched_attn


def test_kernel_training_integration_dropout_non_zero(temp_dir):
    """Test model loading with dropout non-zero should not patch."""

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
            "lora_dropout": 0.1,
            "lora_target_linear": True,
            "sequence_len": 1024,
        }
    )

    # Write cfg to yaml file
    path = Path(temp_dir) / "config.yaml"
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(yaml.dump(cfg.to_dict(), Dumper=yaml.Dumper))

    # Load config
    cfg = load_cfg(str(path))

    # Get original attention class
    attention_cls = get_attention_cls_from_config(cfg)

    # Store original state before patching
    original_forward_method = attention_cls.forward

    # Load model
    model, tokenizer, _ = load_model_and_tokenizer(cfg=cfg)

    # We call modelloader as that's where the patches are applied
    # despite the fact that we're not using it to load the model
    model_loader = ModelLoader(cfg, tokenizer)

    # Apply patch
    model_loader.patch_manager._apply_self_attention_lora_patch()

    # Verify patch was not applied
    assert attention_cls.forward == original_forward_method

    # Apply apply_lora_kernel_patches
    model_loader.patch_manager._apply_lora_kernel_patch(model)

    # Verify patch was not applied
    layers = get_layers(model)
    for layer in layers:
        for self_attn in find_self_attn_in_layer(layer):
            assert not hasattr(self_attn, "apply_qkv")
            assert not hasattr(self_attn, "apply_o")
