"""Tests for the DiffusionGemma canvas collator."""

import torch

from axolotl.integrations.diffusion_gemma.collator import CanvasCollator


class _Tok:
    pad_token_id = 0
    bos_token_id = 1


def _collator(canvas_length=8, **kw):
    return CanvasCollator(_Tok(), canvas_length=canvas_length, seed=0, **kw)


def test_basic_prefix_canvas_split():
    # prompt = [1,2,3] (labels -100), answer = [4,5,6] (labels = ids)
    feat = {"input_ids": [1, 2, 3, 4, 5, 6], "labels": [-100, -100, -100, 4, 5, 6]}
    batch = _collator(canvas_length=8, block_selection="first")([feat])
    assert batch["input_ids"].shape == (1, 3)
    assert batch["input_ids"][0].tolist() == [1, 2, 3]
    assert batch["canvas_labels"].shape == (1, 8)
    # canvas holds the 3 answer tokens then padding
    assert batch["canvas_labels"][0, :3].tolist() == [4, 5, 6]
    assert batch["canvas_loss_mask"][0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]


def test_prefix_padding_and_attention_mask():
    feats = [
        {"input_ids": [1, 2, 9], "labels": [-100, -100, 9]},
        {"input_ids": [1, 2, 3, 4, 9], "labels": [-100, -100, -100, -100, 9]},
    ]
    batch = _collator(canvas_length=4, block_selection="first")(feats)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1]


def test_long_answer_first_block_truncates_to_canvas():
    answer = list(range(10, 30))  # 20 answer tokens
    feat = {"input_ids": [1, 2] + answer, "labels": [-100, -100] + answer}
    batch = _collator(canvas_length=8, block_selection="first")([feat])
    assert batch["canvas_labels"].shape == (1, 8)
    assert batch["canvas_labels"][0].tolist() == answer[:8]
    assert batch["canvas_loss_mask"][0].sum().item() == 8


def test_random_block_extends_prefix():
    answer = list(range(10, 26))  # 16 tokens -> 2 blocks at L=8
    feat = {"input_ids": [1] + answer, "labels": [-100] + answer}
    # seed chosen so we can just assert shape invariants across many draws
    coll = _collator(canvas_length=8, block_selection="random")
    seen_prefix_lens = set()
    for _ in range(20):
        b = coll([feat])
        seen_prefix_lens.add(b["input_ids"].shape[1])
        # canvas is always a contiguous slice of the answer
        canvas = b["canvas_labels"][0][b["canvas_loss_mask"][0].bool()].tolist()
        assert canvas == answer[: len(canvas)] or canvas == answer[8 : 8 + len(canvas)]
    # both block 0 (prefix len 1) and block 1 (prefix len 9) should appear
    assert len(seen_prefix_lens) == 2


def test_no_masked_prompt_falls_back_to_bos_prefix():
    feat = {"input_ids": [5, 6, 7], "labels": [5, 6, 7]}
    batch = _collator(canvas_length=8, block_selection="first")([feat])
    assert batch["input_ids"][0].tolist() == [1]  # bos
    assert batch["canvas_labels"][0, :3].tolist() == [5, 6, 7]


def test_dtypes_are_long():
    feat = {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}
    batch = _collator(canvas_length=4)([feat])
    for k in ("input_ids", "attention_mask", "canvas_labels", "canvas_loss_mask"):
        assert batch[k].dtype == torch.long


def test_multimodal_mm_token_type_ids_sliced_to_prefix():
    # prompt has two image tokens (mm=1) then text; answer = [50, 51]
    feat = {
        "input_ids": [1, 99, 99, 7, 50, 51],
        "labels": [-100, -100, -100, -100, 50, 51],
        "mm_token_type_ids": [0, 1, 1, 0, 0, 0],
        "pixel_values": torch.zeros(2, 3, 16, 16),
    }
    batch = _collator(canvas_length=4, block_selection="first")([feat])
    # prefix is the 4 prompt tokens; mm ids sliced to match
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape
    assert batch["mm_token_type_ids"][0].tolist() == [0, 1, 1, 0]
    assert batch["pixel_values"].shape == (2, 3, 16, 16)


def test_multimodal_pixel_values_concatenated_across_batch():
    feats = [
        {
            "input_ids": [1, 99, 7, 50],
            "labels": [-100, -100, -100, 50],
            "mm_token_type_ids": [0, 1, 0, 0],
            "pixel_values": torch.zeros(1, 3, 16, 16),
        },
        {
            "input_ids": [1, 99, 99, 7, 60],
            "labels": [-100, -100, -100, -100, 60],
            "mm_token_type_ids": [0, 1, 1, 0, 0],
            "pixel_values": torch.zeros(2, 3, 16, 16),
        },
    ]
    batch = _collator(canvas_length=4, block_selection="first")(feats)
    # image features gathered in row-major order: 1 + 2 = 3 images
    assert batch["pixel_values"].shape == (3, 3, 16, 16)
