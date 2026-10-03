"""GPU trainer-boundary contracts for native packed diffusion."""

import importlib

import pytest
import torch
from datasets import Dataset

from axolotl.core.training_args import AxolotlTrainingArguments
from axolotl.integrations.diffusion.lm.trainer import AxolotlDiffusionTrainer
from axolotl.utils.dict import DictDefault


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("steps", [1, 2], ids=["k1", "k2"])
def test_nemotron_flex_trainer_compute_loss_backward(tmp_path, monkeypatch, steps):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from tests.native_source_fixtures import native_source_fixture_path
    from axolotl.model_support.nemotron_diffusion.compat import (
        resolve_nemotron_model_class,
    )

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
            "diffusion": {"from_causal_lm": False, "unroll": {"k_max": steps}},
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
