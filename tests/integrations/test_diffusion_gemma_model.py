"""End-to-end smoke test: tiny DiffusionGemma + collator + block-diffusion step.

Exercises the full data->model->loss->backward path on real model code (CPU,
random init), without the HF Trainer harness.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")


def _tiny_model():
    from transformers import (
        DiffusionGemmaConfig,
        DiffusionGemmaForBlockDiffusion,
        DiffusionGemmaTextConfig,
        Gemma4VisionConfig,
    )

    torch.manual_seed(0)
    text_cfg = DiffusionGemmaTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=2,
        sliding_window=8,
        max_position_embeddings=256,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
    )
    vis_cfg = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=16,
        num_channels=3,
    )
    cfg = DiffusionGemmaConfig(
        text_config=text_cfg, vision_config=vis_cfg, canvas_length=8
    )
    return DiffusionGemmaForBlockDiffusion(cfg).to(torch.float32)


class _Tok:
    pad_token_id = 0
    bos_token_id = 1


def _batch(canvas_length=8):
    from axolotl.integrations.diffusion_gemma.collator import CanvasCollator

    coll = CanvasCollator(
        _Tok(), canvas_length=canvas_length, seed=0, block_selection="first"
    )
    feats = [
        {
            "input_ids": [1, 5, 6, 7, 20, 21, 22, 23],
            "labels": [-100, -100, -100, -100, 20, 21, 22, 23],
        },
        {"input_ids": [1, 8, 9, 30, 31], "labels": [-100, -100, -100, 30, 31]},
    ]
    return coll(feats)


@pytest.mark.parametrize("corruption", ["uniform", "mask"])
def test_block_diffusion_step_forward_backward(corruption):
    from axolotl.integrations.diffusion_gemma.diffusion import DiffusionObjectiveConfig
    from axolotl.integrations.diffusion_gemma.trainer import block_diffusion_step

    model = _tiny_model()
    model.train()
    batch = _batch()
    cfg = DiffusionObjectiveConfig(
        vocab_size=model.config.get_text_config().vocab_size,
        corruption=corruption,
        mask_token_id=(127 if corruption == "mask" else None),
        self_conditioning_prob=0.0,
    )
    gen = torch.Generator().manual_seed(0)
    loss, metrics, outputs = block_diffusion_step(model, batch, cfg, generator=gen)

    assert outputs.logits.shape == (2, 8, 128)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert "diffusion/token_ce" in metrics


def test_self_conditioning_path_runs():
    from axolotl.integrations.diffusion_gemma.diffusion import DiffusionObjectiveConfig
    from axolotl.integrations.diffusion_gemma.trainer import block_diffusion_step

    model = _tiny_model()
    model.train()
    batch = _batch()
    cfg = DiffusionObjectiveConfig(
        vocab_size=model.config.get_text_config().vocab_size,
        self_conditioning_prob=1.0,  # force the self-conditioning branch
    )
    gen = torch.Generator().manual_seed(1)
    loss, metrics, _ = block_diffusion_step(model, batch, cfg, generator=gen)
    assert metrics["diffusion/self_conditioned"] == 1.0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_multimodal_kwargs_forwarded_to_encoder():
    """pixel_values / mm_token_type_ids / image_position_ids reach model.forward."""
    from types import SimpleNamespace

    from axolotl.integrations.diffusion_gemma.diffusion import DiffusionObjectiveConfig
    from axolotl.integrations.diffusion_gemma.trainer import block_diffusion_step

    seen = {}

    class _RecordingModel:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            b, seq = kwargs["decoder_input_ids"].shape
            return SimpleNamespace(logits=torch.randn(b, seq, 16))

    batch = {
        "input_ids": torch.tensor([[1, 99, 7, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "canvas_labels": torch.tensor([[5, 6, 0, 0]]),
        "canvas_loss_mask": torch.tensor([[1, 1, 0, 0]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 0, 0]]),
        "pixel_values": torch.zeros(1, 3, 16, 16),
        "image_position_ids": torch.zeros(1, 2, dtype=torch.long),
    }
    cfg = DiffusionObjectiveConfig(vocab_size=16, self_conditioning_prob=0.0)
    block_diffusion_step(
        _RecordingModel(), batch, cfg, generator=torch.Generator().manual_seed(0)
    )
    assert "pixel_values" in seen and seen["pixel_values"].shape == (1, 3, 16, 16)
    assert "mm_token_type_ids" in seen and "image_position_ids" in seen


def test_padded_prefix_is_masked_from_decoder():
    """A batch with differing prefix lengths must still produce a finite loss."""
    from axolotl.integrations.diffusion_gemma.diffusion import DiffusionObjectiveConfig
    from axolotl.integrations.diffusion_gemma.trainer import block_diffusion_step

    model = _tiny_model()
    model.eval()
    batch = _batch()
    # second example has a shorter prefix -> right padding in input_ids
    assert (
        batch["attention_mask"].sum(dim=1) != batch["attention_mask"].shape[1]
    ).any()
    cfg = DiffusionObjectiveConfig(
        vocab_size=model.config.get_text_config().vocab_size,
        self_conditioning_prob=0.0,
    )
    loss, _, _ = block_diffusion_step(
        model, batch, cfg, generator=torch.Generator().manual_seed(2)
    )
    assert torch.isfinite(loss)
