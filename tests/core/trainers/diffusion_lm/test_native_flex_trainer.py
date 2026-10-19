"""GPU trainer-boundary contracts for native packed diffusion."""

import importlib

import pytest
import torch
from datasets import Dataset
from transformers import DiffusionGemmaConfig

from axolotl.core.trainers.diffusion_lm.collator import DiffusionCollator
from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer
from axolotl.core.training_args import AxolotlTrainingArguments
from axolotl.model_support.diffusion_gemma.modeling import (
    AxolotlDiffusionGemmaForBlockDiffusion,
)
from axolotl.utils.dict import DictDefault


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("steps", [1, 2], ids=["k1", "k2"])
def test_gemma_flex_trainer_compute_loss_backwards_padded_two_document_batch(
    tmp_path, dtype, steps, monkeypatch
):
    config = DiffusionGemmaConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 256,
            "layer_types": ["full_attention"],
            "per_layer_config": (
                {} if dtype is torch.bfloat16 else {"0": {"head_dim": 16}}
            ),
            "num_experts": 2,
            "top_k_experts": 1,
            "moe_intermediate_size": 64,
            "pad_token_id": 0,
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "patch_size": 16,
            "position_embedding_size": 16,
        },
        canvas_length=8,
    )
    device = torch.device("cuda")
    model = (
        AxolotlDiffusionGemmaForBlockDiffusion._from_config(
            config, attn_implementation="flex_attention"
        )
        .to(device=device, dtype=dtype)
        .train()
    )
    batch = (
        DiffusionCollator(0, 3)
        .build_batch(
            [
                {"input_ids": [2, 3, 4, 5, 6], "labels": [-100, -100, 4, 5, 6]},
                {
                    "input_ids": [2, 7, 8, 9, 10, 11],
                    "labels": [-100, -100, -100, 9, 10, 11],
                },
            ]
        )
        .to(device)
    )
    trainer = AxolotlDiffusionTrainer(
        model=model,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            report_to=[],
            remove_unused_columns=False,
            bf16=dtype is torch.bfloat16,
        ),
        train_dataset=Dataset.from_list([{"example": 0}]),
        data_collator=lambda _: batch.__dict__,
    )
    trainer.axolotl_cfg = DictDefault(
        {
            "attn_implementation": "flex_attention",
            "flex_attn_compile_kwargs": {
                "fwd_BLOCK_M": 16,
                "fwd_BLOCK_N": 16,
                "bwd_BLOCK_M1": 16,
                "bwd_BLOCK_N1": 16,
                "bwd_BLOCK_M2": 16,
                "bwd_BLOCK_N2": 16,
                "fwd_num_stages": 1,
                "bwd_num_stages": 1,
            },
            "diffusion_lm": {
                "from_causal_lm": False,
                "self_conditioning": {"p": 0.0},
                "unroll": {"k_max": steps},
            },
        }
    )
    trainer.model.to(device=device, dtype=dtype)
    monkeypatch.setattr(
        AxolotlDiffusionTrainer,
        "_sample_native_unroll_steps",
        staticmethod(lambda _k_max, _device: steps),
    )
    trainer.post_set_axolotl_cfg()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype is torch.bfloat16):
        loss = trainer.compute_loss(model, batch.__dict__)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("steps", [1, 2], ids=["k1", "k2"])
def test_nemotron_flex_trainer_compute_loss_backward(tmp_path, monkeypatch, steps):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from axolotl.model_support.nemotron_diffusion.compat import (
        resolve_nemotron_model_class,
    )

    from tests.native_source_fixtures import native_source_fixture_path

    source = native_source_fixture_path("nemotron")
    if source is None:
        pytest.skip("native Nemotron source fixture unavailable")
    source = str(source)
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
        source,
        local_files_only=True,
    )
    config = config_class(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        mask_token_id=100,
        dlm_paradigm="bidirectional",
        rope_parameters={
            "llama_4_scaling_beta": 1.0,
            "original_max_position_embeddings": 1,
        },
        use_cache=False,
    )
    device = torch.device("cuda")
    model = resolve_nemotron_model_class(source)(config).to(device).train()
    model.config._attn_implementation = "flex_attention"
    model.encoder.config._attn_implementation = "flex_attention"
    received_options = []
    source_module = importlib.import_module(
        type(model.encoder.layers[0].self_attn).__mro__[1].__module__
    )
    original_flex = source_module.ALL_ATTENTION_FUNCTIONS["flex_attention"]
    monkeypatch.setitem(
        source_module.ALL_ATTENTION_FUNCTIONS,
        "flex_attention",
        lambda *args, **kwargs: (
            received_options.append(kwargs.get("kernel_options"))
            or original_flex(*args, **kwargs)
        ),
    )
    batch = {
        "input_ids": torch.tensor([[3, 4, 5, 6, 7, 8]], device=device),
        "document_ids": torch.tensor([[17, 17, 17, 91, 91, 91]], device=device),
        "semantic_validity": torch.ones((1, 6), device=device, dtype=torch.bool),
        "canvas_loss_mask": torch.ones((1, 6), device=device, dtype=torch.bool),
        "canvas_corruptible_mask": torch.ones((1, 6), device=device, dtype=torch.bool),
    }
    trainer = AxolotlDiffusionTrainer(
        model=model,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            report_to=[],
            remove_unused_columns=False,
        ),
        train_dataset=Dataset.from_list([{"x": 0}]),
    )
    trainer.axolotl_cfg = DictDefault(
        {
            "attn_implementation": "flex_attention",
            "flex_attn_compile_kwargs": {
                "fwd_BLOCK_M": 16,
                "fwd_BLOCK_N": 16,
                "bwd_BLOCK_M1": 16,
                "bwd_BLOCK_N1": 16,
                "bwd_BLOCK_M2": 16,
                "bwd_BLOCK_N2": 16,
                "fwd_num_stages": 1,
                "bwd_num_stages": 1,
            },
            "diffusion_lm": {"from_causal_lm": False, "unroll": {"k_max": steps}},
        }
    )
    monkeypatch.setattr(
        AxolotlDiffusionTrainer,
        "_sample_native_unroll_steps",
        staticmethod(lambda _k, _d: steps),
    )
    trainer.post_set_axolotl_cfg()
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert len(received_options) == steps
    assert all(
        options == trainer.axolotl_cfg.flex_attn_compile_kwargs
        for options in received_options
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dream_flex_trainer_compute_loss_backward(tmp_path, monkeypatch):
    from transformers import AutoConfig

    from axolotl.model_support.dream import _model_class, compat

    from tests.native_source_fixtures import native_source_fixture_path

    source = native_source_fixture_path("dream")
    if source is None:
        pytest.skip("native Dream source fixture unavailable")
    config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    for name, value in {
        "vocab_size": 32,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 128,
        "max_window_layers": 1,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "mask_token_id": 2,
        "use_cache": False,
    }.items():
        setattr(config, name, value)
    config._name_or_path = str(source)
    device = torch.device("cuda")
    model = (
        _model_class()
        .from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="flex_attention",
        )
        .to(device)
        .train()
    )
    received_options = []
    original_flex = compat._dream_flex_attention
    monkeypatch.setattr(
        compat,
        "_dream_flex_attention",
        lambda *args, **kwargs: (
            received_options.append(
                kwargs.get("kernel_options", args[4] if len(args) > 4 else None)
            )
            or original_flex(*args, **kwargs)
        ),
    )
    batch = {
        "input_ids": torch.tensor([[3, 4, 5, 6, 7, 8]], device=device),
        "document_ids": torch.tensor([[17, 17, 17, 91, 91, 91]], device=device),
        "semantic_validity": torch.ones((1, 6), device=device, dtype=torch.bool),
        "canvas_loss_mask": torch.ones((1, 6), device=device, dtype=torch.bool),
        "canvas_corruptible_mask": torch.ones((1, 6), device=device, dtype=torch.bool),
    }
    trainer = AxolotlDiffusionTrainer(
        model=model,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            report_to=[],
            remove_unused_columns=False,
        ),
        train_dataset=Dataset.from_list([{"x": 0}]),
    )
    trainer.axolotl_cfg = DictDefault(
        {
            "attn_implementation": "flex_attention",
            "flex_attn_compile_kwargs": {"fwd_BLOCK_M": 16, "fwd_BLOCK_N": 16},
            "diffusion_lm": {"from_causal_lm": False, "unroll": {"k_max": 1}},
        }
    )
    trainer.post_set_axolotl_cfg()
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert received_options == [trainer.axolotl_cfg.flex_attn_compile_kwargs]
