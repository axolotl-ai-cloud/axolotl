"""Model-family shape tests for the packed multimodal collator."""

from __future__ import annotations

import numpy as np

from axolotl.utils.collators import MultiModalBatchSamplerDataCollatorForSeq2Seq

from tests.mm_packing_utils import PadTokenizer


def _collator():
    return MultiModalBatchSamplerDataCollatorForSeq2Seq(
        tokenizer=PadTokenizer(), padding=True, return_tensors="pt"
    )


def _text(n, offset):
    return {
        "input_ids": list(range(offset, offset + n)),
        "labels": list(range(offset, offset + n)),
        "attention_mask": [1] * n,
        "position_ids": list(range(n)),
    }


def test_idefics3_pixel_values_keep_batch_dim():
    # Tiled VLM: per-row pixel_values (1, n_img, 3, H, W); model needs 5D back.
    r1 = {
        **_text(3, 10),
        "pixel_values": np.ones((1, 2, 3, 8, 8), np.float32),
        "pixel_attention_mask": np.ones((2, 8, 8), np.int64),
    }
    r2 = {
        **_text(2, 20),
        "pixel_values": np.full((1, 3, 3, 8, 8), 2.0, np.float32),
        "pixel_attention_mask": np.ones((3, 8, 8), np.int64),
    }
    batch = _collator()([[r1, r2]])
    assert tuple(batch["pixel_values"].shape) == (1, 5, 3, 8, 8)
    assert tuple(batch["pixel_attention_mask"].shape) == (1, 5, 8, 8)
    assert batch["input_ids"].shape == (1, 5)


def test_idefics3_micro_batch_size_gt_1_pads_images():
    # mbs>1 -> B packs per call; tiled pixel_values pad to the batch-max image count.
    r1 = {
        **_text(3, 10),
        "pixel_values": np.ones((1, 2, 3, 8, 8), np.float32),
        "pixel_attention_mask": np.ones((2, 8, 8), np.int64),
    }
    r2 = {
        **_text(2, 20),
        "pixel_values": np.full((1, 3, 3, 8, 8), 2.0, np.float32),
        "pixel_attention_mask": np.ones((3, 8, 8), np.int64),
    }
    batch = _collator()([[r1], [r2]])
    assert tuple(batch["pixel_values"].shape) == (2, 3, 3, 8, 8)
    assert tuple(batch["pixel_attention_mask"].shape) == (2, 3, 8, 8)
    assert batch["input_ids"].shape == (2, 3)
    # first pack had 2 images -> its 3rd image slot is zero padding
    assert batch["pixel_values"][0, 2].abs().sum() == 0


def test_llava_style_pixel_values_no_spurious_batch_dim():
    # 4D per-row (n_img, C, H, W) must stay 4D (concat images), not gain a batch dim.
    r1 = {**_text(2, 10), "pixel_values": np.ones((1, 3, 8, 8), np.float32)}
    r2 = {**_text(2, 20), "pixel_values": np.ones((2, 3, 8, 8), np.float32)}
    batch = _collator()([[r1, r2]])
    assert tuple(batch["pixel_values"].shape) == (3, 3, 8, 8)


def test_qwen2vl_flat_patches_concat():
    # Qwen2-VL's 2D flat patches must stay 2D (concat along axis 0), not gain a batch dim.
    embed_dim = 8
    p1, p2 = 1 * 4 * 4, 1 * 2 * 6
    r1 = {
        **_text(3, 10),
        "pixel_values": np.ones((p1, embed_dim), np.float32),
        "image_grid_thw": np.array([[1, 4, 4]], np.int64),
    }
    r2 = {
        **_text(2, 20),
        "pixel_values": np.full((p2, embed_dim), 2.0, np.float32),
        "image_grid_thw": np.array([[1, 2, 6]], np.int64),
    }
    batch = _collator()([[r1, r2]])
    assert batch["pixel_values"].ndim == 2
    assert tuple(batch["pixel_values"].shape) == (p1 + p2, embed_dim)
    assert tuple(batch["image_grid_thw"].shape) == (2, 3)
    assert batch["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 6]]


def test_pixtral_ragged_stacked_per_row_pads_and_stacks():
    # Pixtral ships (n_img, C, H, W) with per-sample H/W; differing-resolution
    # samples must pad to the pack-wide max and stack.
    r1 = {**_text(3, 10), "pixel_values": np.ones((1, 3, 96, 64), np.float32)}
    r2 = {**_text(2, 20), "pixel_values": np.full((1, 3, 64, 128), 2.0, np.float32)}
    batch = _collator()([[r1, r2]])
    assert tuple(batch["pixel_values"].shape) == (2, 3, 96, 128)
    assert batch["pixel_values"][0, 0, 0, 0] == 1
    assert batch["pixel_values"][1, 0, 0, 0] == 2
    assert batch["pixel_values"][1, 0, 80, 80] == 0  # r2 was only 64x128


def test_pixtral_ragged_list_of_images_single_row():
    # A single sample carrying a Python list of differently-sized images.
    r1 = {
        **_text(4, 10),
        "pixel_values": [
            np.ones((3, 512, 768), np.float32),
            np.full((3, 256, 1024), 2.0, np.float32),
        ],
    }
    batch = _collator()([[r1]])
    assert tuple(batch["pixel_values"].shape) == (2, 3, 512, 1024)
    assert batch["pixel_values"][0, 0, 0, 0] == 1
    assert batch["pixel_values"][1, 0, 0, 0] == 2
    assert batch["pixel_values"][0, 0, 0, 900] == 0  # img0 width was 768 -> padded


def test_pixtral_ragged_image_sizes_concat_no_batch_dim():
    # image_sizes must collate to (num_images, 2), the layout the model crops with.
    r1 = {
        **_text(3, 10),
        "pixel_values": np.ones((1, 3, 96, 64), np.float32),
        "image_sizes": np.array([[96, 64]], np.int64),
    }
    r2 = {
        **_text(2, 20),
        "pixel_values": np.full((1, 3, 64, 128), 2.0, np.float32),
        "image_sizes": np.array([[64, 128]], np.int64),
    }
    batch = _collator()([[r1, r2]])
    assert tuple(batch["image_sizes"].shape) == (2, 2)
    assert batch["image_sizes"].tolist() == [[96, 64], [64, 128]]


def test_pixtral_ragged_micro_batch_size_gt_1_pads_across_packs():
    # mbs>1 -> separate packs; ragged pixel_values pad to the batch-wide max H/W.
    r1 = {**_text(3, 10), "pixel_values": np.ones((1, 3, 96, 64), np.float32)}
    r2 = {**_text(2, 20), "pixel_values": np.full((1, 3, 64, 128), 2.0, np.float32)}
    batch = _collator()([[r1], [r2]])
    assert tuple(batch["pixel_values"].shape) == (2, 3, 96, 128)
    assert batch["pixel_values"][0, 0, 0, 0] == 1
    assert batch["pixel_values"][1, 0, 0, 0] == 2


def test_pixtral_ragged_list_of_images_pads_and_concats_image_sizes():
    # A plain list of variable-resolution images must pad to 4D; a list breaks the vision tower.
    r1 = {
        **_text(4, 10),
        "pixel_values": [
            np.ones((3, 64, 96), np.float32),
            np.full((3, 128, 64), 2.0, np.float32),
        ],
        "image_sizes": np.array([[64, 96], [128, 64]], np.int64),
    }
    batch = _collator()([[r1]])
    assert batch["pixel_values"].ndim == 4
    assert tuple(batch["pixel_values"].shape) == (2, 3, 128, 96)
    assert batch["pixel_values"][0, 0, 0, 0] == 1
    assert batch["pixel_values"][1, 0, 0, 0] == 2
    assert tuple(batch["image_sizes"].shape) == (2, 2)
    assert batch["image_sizes"].tolist() == [[64, 96], [128, 64]]
