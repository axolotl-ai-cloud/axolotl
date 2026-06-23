"""Integration tests for LoRA activation and attention kernels."""

import contextlib
from pathlib import Path

import pytest
import torch
import yaml
from accelerate.state import PartialState
from peft import LoraConfig, PeftModelForCausalLM, get_peft_config, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from axolotl.cli.config import load_cfg
from axolotl.kernels.lora import (
    apply_lora_gdn_in_proj,
    apply_lora_linear,
    apply_lora_mlp_geglu,
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
)
from axolotl.loaders.model import ModelLoader
from axolotl.loaders.tokenizer import load_tokenizer
from axolotl.monkeypatch.lora_kernels import (
    LINEAR_ATTN_IN_PROJS,
    LINEAR_ATTN_PROJS,
    apply_lora_kernel_patches,
    find_linear_attn_in_layer,
    find_self_attn_in_layer,
    get_attention_cls_from_config,
    get_layers,
    patch_self_attn_lora,
)
from axolotl.utils.dict import DictDefault

MODEL_CONFIGS = [
    {
        "name": "axolotl-ai-co/tiny-mistral-25m",
        "expected_activation": apply_lora_mlp_swiglu,
        "dtype": torch.float16,
    },
    {
        "name": "axolotl-ai-co/tiny-qwen2-129m",
        "expected_activation": apply_lora_mlp_swiglu,
        "dtype": torch.float16,
    },
    {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "expected_activation": apply_lora_mlp_swiglu,
        "dtype": torch.float32,
    },
    {
        "name": "axolotl-ai-co/tiny-gemma2-137m",
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
        "axolotl-ai-co/tiny-gemma2-137m",
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
    """Test that kernels ARE patched even with dropout and bias (now supported)."""
    test_configs = [
        # Dropout — kernels now support this
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
        },
        # Bias — kernels now support this
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

        patched_model = apply_lora_kernel_patches(model, cfg)
        layer = patched_model.model.model.layers[0].mlp

        # Verify patches ARE applied (dropout and bias are now supported)
        assert (
            layer.forward.__func__ is apply_lora_mlp_swiglu
            or layer.forward.__func__ is apply_lora_mlp_geglu
        )


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
    """Test model loading with dropout non-zero DOES patch (now supported)."""

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

    # Load model
    model, tokenizer, _ = load_model_and_tokenizer(cfg=cfg)

    model_loader = ModelLoader(cfg, tokenizer)

    # Apply patches — should succeed even with dropout > 0
    model_loader.patch_manager._apply_self_attention_lora_patch()
    model_loader.patch_manager._apply_lora_kernel_patch(model)

    # Verify patches WERE applied (dropout is now supported by kernels)
    layers = get_layers(model)
    for layer in layers:
        for self_attn in find_self_attn_in_layer(layer):
            assert hasattr(self_attn, "apply_qkv")
            assert hasattr(self_attn, "apply_o")


# ============================================================
# GatedDeltaNet (linear-attention) fused-LoRA routing
# ============================================================


def _gdn_module_with(proj_names, in_features=64, out_features=64):
    module = nn.Module()
    for name in proj_names:
        setattr(module, name, nn.Linear(in_features, out_features, bias=False))
    return module


def _gdn_layer(linear_attn=None, self_attn=False):
    layer = nn.Module()
    if linear_attn is not None:
        layer.linear_attn = linear_attn
    if self_attn:
        layer.self_attn = _gdn_module_with(["q_proj", "k_proj", "v_proj", "o_proj"])
    return layer


class TestFindLinearAttnInLayer:
    """Selects Qwen3.5 GatedDeltaNet layers only; qwen3_next (fused in_proj_qkvz/in_proj_ba) must NOT be patched."""

    def test_qwen3_5_style_is_selected(self):
        layer = _gdn_layer(_gdn_module_with(LINEAR_ATTN_PROJS))
        assert list(find_linear_attn_in_layer(layer)) == [layer.linear_attn]

    def test_qwen3_next_style_is_excluded(self):
        layer = _gdn_layer(_gdn_module_with(["in_proj_qkvz", "in_proj_ba", "out_proj"]))
        assert list(find_linear_attn_in_layer(layer)) == []

    def test_non_gdn_layer_is_excluded(self):
        assert list(find_linear_attn_in_layer(_gdn_layer(self_attn=True))) == []

    def test_out_proj_only_is_excluded(self):
        layer = _gdn_layer(_gdn_module_with(["out_proj"]))
        assert list(find_linear_attn_in_layer(layer)) == []

    def test_missing_linear_attn_attr_is_excluded(self):
        assert list(find_linear_attn_in_layer(nn.Module())) == []


def _gdn_rel_l2(actual, reference):
    return (actual - reference).norm().item() / (reference.norm().item() + 1e-12)


def _wrapped_gdn_proj(in_features=256, out_features=384, use_dora=False):
    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = nn.Linear(in_features, out_features, bias=False)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["in_proj_qkv"],
        lora_dropout=0.0,
        use_dora=use_dora,
    )
    peft_model = get_peft_model(_Holder().to("cuda").to(torch.bfloat16), config)
    proj = peft_model.base_model.model.in_proj_qkv
    proj.train()
    return proj


def _run_gdn_proj(proj, fused, in_features=256, seed=0):
    torch.manual_seed(seed)
    inputs = torch.randn(
        2, 16, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = apply_lora_linear(proj, inputs) if fused else proj(inputs)
    out.float().pow(2).mean().backward()
    grad = inputs.grad.detach().float().clone()
    proj.zero_grad(set_to_none=True)
    return out.detach().float(), grad


def test_apply_lora_linear_matches_peft_forward_and_grad():
    """Fused path matches the peft module forward and grad to bf16 float noise."""
    proj = _wrapped_gdn_proj()
    out_ref, grad_ref = _run_gdn_proj(proj, fused=False)
    out_fused, grad_fused = _run_gdn_proj(proj, fused=True)

    assert _gdn_rel_l2(out_fused, out_ref) < 1e-3
    assert _gdn_rel_l2(grad_fused, grad_ref) < 1e-3


def test_apply_lora_linear_matches_peft_with_dora():
    """The DoRA branch (magnitude scaling) routes correctly through LoRA_O."""
    proj = _wrapped_gdn_proj(use_dora=True)
    with torch.no_grad():
        for name, param in proj.named_parameters():
            if "lora_B" in name or "magnitude" in name:
                param.add_(torch.randn_like(param) * 0.01)

    out_ref, grad_ref = _run_gdn_proj(proj, fused=False)
    out_fused, grad_fused = _run_gdn_proj(proj, fused=True)

    assert _gdn_rel_l2(out_fused, out_ref) < 1e-2
    assert _gdn_rel_l2(grad_fused, grad_ref) < 1e-2


# ------------------------------------------------------------
# Fused shared-input in-projection kernel (apply_lora_gdn_in_proj)
# ------------------------------------------------------------

_GDN_IN_SIZES = {
    "in_proj_qkv": 384,
    "in_proj_z": 256,
    "in_proj_b": 16,  # starved: out == num_v_heads
    "in_proj_a": 16,
}


def _wrapped_gdn_block(targets, in_features=256, use_dora=False):
    """A peft-wrapped module holding the four GDN input projections."""

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            for name, out_features in _GDN_IN_SIZES.items():
                setattr(self, name, nn.Linear(in_features, out_features, bias=False))

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=list(targets),
        lora_dropout=0.0,
        use_dora=use_dora,
    )
    block = get_peft_model(
        _Block().to("cuda").to(torch.bfloat16), config
    ).base_model.model
    block.train()
    # Perturb lora_B / magnitude so adapters (and DoRA scaling) are non-trivial.
    with torch.no_grad():
        for name, param in block.named_parameters():
            if "lora_B" in name or "magnitude" in name:
                param.add_(torch.randn_like(param) * 0.02)
    return block


def _run_gdn_block(block, fused, targets, in_features=256, seed=0):
    names = tuple(_GDN_IN_SIZES)
    torch.manual_seed(seed)
    inputs = torch.randn(
        2, 16, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if fused:
            outs = apply_lora_gdn_in_proj(block, inputs, names)
        else:
            outs = {name: getattr(block, name)(inputs) for name in names}
    sum(o.float().pow(2).mean() for o in outs.values()).backward()
    grad_x = inputs.grad.detach().float().clone()
    grad_a = {
        name: getattr(block, name)
        .lora_A["default"]
        .weight.grad.detach()
        .float()
        .clone()
        for name in targets
    }
    block.zero_grad(set_to_none=True)
    return {k: v.detach().float() for k, v in outs.items()}, grad_x, grad_a


@pytest.mark.parametrize(
    "targets",
    [
        ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),  # all adapted
        ("in_proj_qkv", "in_proj_z"),  # subset: large only
        ("in_proj_b", "in_proj_a"),  # subset: starved only
        ("in_proj_qkv",),  # subset: single
    ],
)
def test_apply_lora_gdn_in_proj_matches_peft(targets):
    """Fused shared-input path matches independent peft forwards + grads, including
    when only a subset of the projections carry a LoRA adapter (the rest base-only)."""
    block = _wrapped_gdn_block(targets)
    out_ref, gx_ref, ga_ref = _run_gdn_block(block, fused=False, targets=targets)
    out_fused, gx_fused, ga_fused = _run_gdn_block(block, fused=True, targets=targets)

    for name in _GDN_IN_SIZES:
        assert _gdn_rel_l2(out_fused[name], out_ref[name]) < 5e-3
    assert _gdn_rel_l2(gx_fused, gx_ref) < 1e-2  # 4-way bf16 grad accumulation
    for name in targets:
        assert _gdn_rel_l2(ga_fused[name], ga_ref[name]) < 5e-3


def test_apply_lora_gdn_in_proj_matches_peft_with_dora():
    """DoRA magnitude scaling routes correctly through the fused in-projection."""
    targets = ("in_proj_qkv", "in_proj_b")
    block = _wrapped_gdn_block(targets, use_dora=True)
    out_ref, gx_ref, ga_ref = _run_gdn_block(block, fused=False, targets=targets)
    out_fused, gx_fused, ga_fused = _run_gdn_block(block, fused=True, targets=targets)

    for name in _GDN_IN_SIZES:
        assert _gdn_rel_l2(out_fused[name], out_ref[name]) < 2e-2
    assert _gdn_rel_l2(gx_fused, gx_ref) < 2e-2
    for name in targets:
        assert _gdn_rel_l2(ga_fused[name], ga_ref[name]) < 2e-2


@contextlib.contextmanager
def _fp32_matmul():
    """Force true fp32 matmuls so fp32 self-consistency checks are order-independent
    (a prior test in the suite may enable TF32, which degrades fp32 to ~1e-4)."""
    prev_cuda = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_cuda
        torch.backends.cudnn.allow_tf32 = prev_cudnn


class _FixedDropout(nn.Module):
    """Deterministic stand-in for nn.Dropout: applies a precomputed (already scaled) mask."""

    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x * self.mask


def _wrapped_gdn_block_fp32(targets, in_features=256, use_dora=False):
    """fp32 GDN in-projection block so fused-vs-reference grads compare to fp noise."""

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            for name, out_features in _GDN_IN_SIZES.items():
                setattr(self, name, nn.Linear(in_features, out_features, bias=False))

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=list(targets),
        lora_dropout=0.0,
        use_dora=use_dora,
    )
    block = get_peft_model(_Block().to("cuda"), config).base_model.model
    block.train()
    with torch.no_grad():
        for name, param in block.named_parameters():
            if "lora_B" in name or "magnitude" in name:
                param.add_(torch.randn_like(param) * 0.1)
    return block


def _gdn_in_proj_grads(block, names, targets, X, reference, mask=None):
    """Run forward+backward and return (grad_X, grad_A, grad_B). The reference path
    rebuilds the fused math with pure torch ops using the SAME shared dropout mask."""
    X = X.detach().clone().requires_grad_(True)
    if reference:
        X_drop = X * mask
        loss = 0.0
        for name in names:
            proj = getattr(block, name)
            out = X @ proj.base_layer.weight.t()
            if name in targets:
                A = proj.lora_A["default"].weight
                B = proj.lora_B["default"].weight
                out = out + proj.scaling["default"] * (X_drop @ A.t()) @ B.t()
            loss = loss + out.float().pow(2).mean()
    else:
        outs = apply_lora_gdn_in_proj(block, X, names)
        loss = sum(o.float().pow(2).mean() for o in outs.values())
    loss.backward()

    grad_x = X.grad.detach().clone()
    grad_a = {
        n: getattr(block, n).lora_A["default"].weight.grad.detach().clone()
        for n in targets
    }
    grad_b = {
        n: getattr(block, n).lora_B["default"].weight.grad.detach().clone()
        for n in targets
    }
    block.zero_grad(set_to_none=True)
    return grad_x, grad_a, grad_b


def test_apply_lora_gdn_in_proj_backward_under_dropout():
    """Gradient correctness on the ``lora_dropout > 0`` backward path (saved ``X_drop`` +
    ``grad_X_drop`` accumulation), which the dropout=0 parity tests never exercise.
    Compared against a pure-torch reference using the SAME shared dropout mask."""
    names = tuple(_GDN_IN_SIZES)
    targets = names  # all four adapted
    block = _wrapped_gdn_block_fp32(targets)

    torch.manual_seed(0)
    X = torch.randn(2, 16, 256, device="cuda")
    mask = (torch.rand_like(X) > 0.5).to(X.dtype) / 0.5  # inverted-dropout scale, p=0.5
    for name in names:
        getattr(block, name).lora_dropout["default"] = _FixedDropout(mask)

    with _fp32_matmul():
        gx_ref, ga_ref, gb_ref = _gdn_in_proj_grads(
            block, names, targets, X, reference=True, mask=mask
        )
        gx_fused, ga_fused, gb_fused = _gdn_in_proj_grads(
            block, names, targets, X, reference=False
        )

    assert _gdn_rel_l2(gx_fused, gx_ref) < 1e-4
    for name in targets:
        assert _gdn_rel_l2(ga_fused[name], ga_ref[name]) < 1e-4
        assert _gdn_rel_l2(gb_fused[name], gb_ref[name]) < 1e-4


def test_apply_lora_gdn_in_proj_dora_backward_under_dropout():
    """Self-consistency of the fused GDN DoRA + ``lora_dropout>0`` path: matches the kernel's
    intended math (base undropped, LoRA dropped, magnitude-scaled), including the ``d_mag``
    gradient."""
    targets = ("in_proj_qkv", "in_proj_b")
    names = tuple(_GDN_IN_SIZES)
    block = _wrapped_gdn_block_fp32(targets, use_dora=True)

    torch.manual_seed(0)
    x = torch.randn(2, 16, 256, device="cuda")
    mask = (torch.rand_like(x) > 0.5).to(x.dtype) / 0.5  # inverted-dropout scale, p=0.5
    for name in targets:  # only adapted projections carry lora_dropout
        getattr(block, name).lora_dropout["default"] = _FixedDropout(mask)

    def run(fused):
        inp = x.detach().clone().requires_grad_(True)
        if fused:
            outs = apply_lora_gdn_in_proj(block, inp, names)
        else:
            # kernel's intended DoRA math: base on undropped X, LoRA on dropped X,
            # mag_scale = magnitude / ||W + s*B@A||_col (norm detached, as the kernel does)
            x_drop = inp * mask
            outs = {}
            for n in names:
                proj = getattr(block, n)
                if n not in targets:  # base-only projection is a plain nn.Linear
                    outs[n] = inp @ proj.weight.t()
                    continue
                W = proj.base_layer.weight
                A = proj.lora_A["default"].weight
                B = proj.lora_B["default"].weight
                s = proj.scaling["default"]
                mag = proj.lora_magnitude_vector["default"].weight
                lora = s * (x_drop @ A.t()) @ B.t()
                weight_norm = torch.linalg.norm(W + s * (B @ A), dim=1).detach()
                outs[n] = (mag / weight_norm).unsqueeze(0) * (inp @ W.t() + lora)
        sum(o.float().pow(2).mean() for o in outs.values()).backward()
        grads = {"x": inp.grad.detach().float().clone()}
        for n in targets:
            proj = getattr(block, n)
            grads[f"A:{n}"] = (
                proj.lora_A["default"].weight.grad.detach().float().clone()
            )
            grads[f"mag:{n}"] = (
                proj.lora_magnitude_vector["default"]
                .weight.grad.detach()
                .float()
                .clone()
            )
        out = {n: outs[n].detach().float() for n in names}
        block.zero_grad(set_to_none=True)
        return out, grads

    with _fp32_matmul():
        out_ref, g_ref = run(fused=False)
        out_fused, g_fused = run(fused=True)

    for name in names:
        assert _gdn_rel_l2(out_fused[name], out_ref[name]) < 1e-4
    assert _gdn_rel_l2(g_fused["x"], g_ref["x"]) < 1e-4
    for name in targets:
        assert _gdn_rel_l2(g_fused[f"A:{name}"], g_ref[f"A:{name}"]) < 1e-4
        assert _gdn_rel_l2(g_fused[f"mag:{name}"], g_ref[f"mag:{name}"]) < 1e-4


# ------------------------------------------------------------
# End-to-end: apply_lora_kernel_patches wires the methods onto a real GDN layer
# ------------------------------------------------------------


def _build_qwen3_5_gdn_peft_model(target_modules):
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
    )
    model = Qwen3_5ForCausalLM(config)
    peft_config = get_peft_config(
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": list(target_modules),
            "lora_dropout": 0,
            "bias": "none",
        }
    )
    return PeftModelForCausalLM(model, peft_config).to("cuda")


def test_apply_lora_kernel_patches_wires_gdn_layer():
    """apply_lora_kernel_patches attaches the fused in-proj + out_proj methods onto a
    peft-wrapped GatedDeltaNet layer, and they match the unpatched peft projections."""
    model = _build_qwen3_5_gdn_peft_model(LINEAR_ATTN_PROJS)
    cfg = DictDefault({"lora_qkv_kernel": True, "lora_o_kernel": True})

    linear_attn = get_layers(model)[0].linear_attn
    ref = {
        name: getattr(linear_attn, name) for name in (*LINEAR_ATTN_IN_PROJS, "out_proj")
    }

    apply_lora_kernel_patches(model, cfg)

    assert hasattr(linear_attn, "apply_in_proj_fused")
    assert hasattr(linear_attn, "apply_out_proj")

    # in-projections share the hidden-size input; out_proj consumes value_dim.
    x_in = torch.randn(
        2, 8, model.config.hidden_size, device="cuda", dtype=torch.float32
    )
    x_out = torch.randn(
        2, 8, ref["out_proj"].in_features, device="cuda", dtype=torch.float32
    )
    with torch.no_grad():
        fused = linear_attn.apply_in_proj_fused(x_in)
        for name in LINEAR_ATTN_IN_PROJS:
            assert torch.allclose(fused[name], ref[name](x_in), rtol=1e-4, atol=1e-4)
        assert torch.allclose(
            linear_attn.apply_out_proj(x_out),
            ref["out_proj"](x_out),
            rtol=1e-4,
            atol=1e-4,
        )


def test_apply_lora_kernel_patches_skips_gdn_without_adapters():
    """No in-projection adapter -> no fused method attached (model left untouched)."""
    model = _build_qwen3_5_gdn_peft_model(["out_proj"])
    cfg = DictDefault({"lora_qkv_kernel": True, "lora_o_kernel": True})

    apply_lora_kernel_patches(model, cfg)

    linear_attn = get_layers(model)[0].linear_attn
    assert not hasattr(linear_attn, "apply_in_proj_fused")
    assert hasattr(linear_attn, "apply_out_proj")  # out_proj still routed


@pytest.mark.parametrize(
    "lora_qkv_kernel, lora_o_kernel, in_proj_patched, out_proj_patched",
    [
        (True, False, True, False),  # qkv flag drives in-projections only
        (False, True, False, True),  # o flag drives out_proj only
    ],
)
def test_apply_lora_kernel_patches_gdn_gates_independently(
    lora_qkv_kernel, lora_o_kernel, in_proj_patched, out_proj_patched
):
    """GDN in_proj follows lora_qkv_kernel and out_proj follows lora_o_kernel,
    matching how self-attn qkv/o gate independently."""
    model = _build_qwen3_5_gdn_peft_model(LINEAR_ATTN_PROJS)
    cfg = DictDefault(
        {"lora_qkv_kernel": lora_qkv_kernel, "lora_o_kernel": lora_o_kernel}
    )

    apply_lora_kernel_patches(model, cfg)

    linear_attn = get_layers(model)[0].linear_attn
    assert hasattr(linear_attn, "apply_in_proj_fused") is in_proj_patched
    assert hasattr(linear_attn, "apply_out_proj") is out_proj_patched
