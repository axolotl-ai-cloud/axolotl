"""Opt-in native-source coverage for Nemotron Diffusion VLM."""

import copy
import os

import pytest
import torch
from transformers import PixtralVisionConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from axolotl.integrations.diffusion.lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.model_support.nemotron_diffusion.compat import (
    resolve_nemotron_vlm_model_class,
)


@pytest.mark.slow
def test_native_vlm_image_forward_selected_and_packed_parity():
    source = os.environ.get("AXOLOTL_NEMOTRON_VLM_SOURCE")
    if not source:
        pytest.skip("set AXOLOTL_NEMOTRON_VLM_SOURCE to run native VLM coverage")
    config_class = get_class_from_dynamic_module(
        "configuration_nemotron_labs_diffusion_vlm.NemotronLabsDiffusionVLMConfig",
        source,
        local_files_only=True,
    )
    config = config_class.from_pretrained(source)
    config.vocab_size = 128
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 1
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.max_position_embeddings = 128
    config.mask_token_id = 100
    config.dlm_paradigm = "bidirectional"
    config.complementary_mask = True
    config.use_cache = False
    config.vision_config = PixtralVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_channels=3,
        patch_size=2,
        image_size=8,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config.spatial_merge_size = 2
    config.rope_parameters = dict(config.rope_scaling)
    config._attn_implementation = "eager"
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(4)
    model = resolve_nemotron_vlm_model_class(source)(config).to(device).train()
    ids = torch.tensor([[1, 18, 19, 21, 2]], device=device)
    pixels = torch.randn(1, 3, 4, 4, device=device)
    image_sizes = torch.tensor([[4, 4]], device=device)
    output = model(input_ids=ids, pixel_values=pixels, image_sizes=image_sizes)
    output.logits[..., :4].sum().backward()
    selected = model(
        input_ids=ids,
        pixel_values=pixels,
        image_sizes=image_sizes,
        axolotl_selected_logits=(
            torch.tensor([[0]], device=device),
            torch.tensor([[4]], device=device),
        ),
    )
    assert selected.logits.shape == (1, 1, 128)

    model.eval()
    with torch.no_grad():
        first = model(
            input_ids=ids, pixel_values=pixels, image_sizes=image_sizes
        ).logits
        changed = model(
            input_ids=ids,
            pixel_values=pixels + 0.25,
            image_sizes=image_sizes,
        ).logits
        assert not torch.allclose(first, changed)
        other_ids = torch.tensor([[3, 18, 19, 21, 4]], device=device)
        other_pixels = torch.randn(1, 3, 4, 4, device=device)
        other_sizes = torch.tensor([[4, 4]], device=device)
        right = model(
            input_ids=other_ids,
            pixel_values=other_pixels,
            image_sizes=other_sizes,
        ).logits
        packed_ids = torch.cat((ids, other_ids), dim=1)
        visible = torch.zeros((1, 1, 10, 10), dtype=torch.bool, device=device)
        visible[:, :, :5, :5] = True
        visible[:, :, 5:, 5:] = True
        packed = model(
            input_ids=packed_ids,
            pixel_values=torch.cat((pixels, other_pixels), dim=0),
            image_sizes=torch.cat((image_sizes, other_sizes), dim=0),
            attention_mask=visible,
            position_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], device=device),
        ).logits
    torch.testing.assert_close(packed[:, :5], first, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(packed[:, 5:], right, atol=1e-5, rtol=1e-5)

    flex_config = copy.deepcopy(config)
    flex_config._attn_implementation = "flex_attention"
    flex = resolve_nemotron_vlm_model_class(source)(flex_config).to(device).eval()
    flex.load_state_dict(model.state_dict())
    with torch.no_grad():
        flex_single = flex(
            input_ids=ids, pixel_values=pixels, image_sizes=image_sizes
        ).logits
        flex_packed = flex(
            input_ids=packed_ids,
            pixel_values=torch.cat((pixels, other_pixels), dim=0),
            image_sizes=torch.cat((image_sizes, other_sizes), dim=0),
            attention_mask=visible,
            position_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], device=device),
        ).logits
    torch.testing.assert_close(flex_single, first, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(flex_packed, packed, atol=1e-4, rtol=1e-4)

    backend = FullSequenceBackend(
        mask_token_id=config.mask_token_id, attention_backend="flex_attention"
    )
    single_canvas = backend.pack(
        ids,
        torch.zeros_like(ids),
        torch.ones_like(ids, dtype=torch.bool),
    )
    right_canvas = backend.pack(
        other_ids,
        torch.zeros_like(other_ids),
        torch.ones_like(other_ids, dtype=torch.bool),
    )
    packed_canvas = backend.pack(
        packed_ids,
        torch.tensor([[17] * 5 + [91] * 5], device=device),
        torch.ones_like(packed_ids, dtype=torch.bool),
    )
    assert type(packed_canvas["attention_mask"]).__name__ == "BlockMask"
    flex.train()
    block_single = flex(
        input_ids=single_canvas["input_ids"],
        pixel_values=pixels,
        image_sizes=image_sizes,
        attention_mask=single_canvas["attention_mask"],
        position_ids=single_canvas["position_ids"],
    ).logits
    block_right = flex(
        input_ids=right_canvas["input_ids"],
        pixel_values=other_pixels,
        image_sizes=other_sizes,
        attention_mask=right_canvas["attention_mask"],
        position_ids=right_canvas["position_ids"],
    ).logits
    block_packed = flex(
        input_ids=packed_canvas["input_ids"],
        pixel_values=torch.cat((pixels, other_pixels), dim=0),
        image_sizes=torch.cat((image_sizes, other_sizes), dim=0),
        attention_mask=packed_canvas["attention_mask"],
        position_ids=packed_canvas["position_ids"],
    ).logits
    torch.testing.assert_close(block_packed[:, :5], block_single[:, :5])
    torch.testing.assert_close(block_packed[:, 5:10], block_right[:, :5])
    block_packed[:, :10].square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in flex.parameters()
    )
