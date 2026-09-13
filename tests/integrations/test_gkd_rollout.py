"""Tests for the GKD Axis B rollout prompt-extraction helper."""

import torch

from axolotl.integrations.gkd.rollout import extract_prompt_batch


def test_right_padded_variable_length_prompts():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 20, 21, 0],  # prompt [10,11,12], completion [20,21], pad
            [30, 31, 40, 41, 42, 43],  # prompt [30,31], completion [40,41,42,43]
        ]
    )
    labels = torch.tensor(
        [
            [-100, -100, -100, 20, 21, -100],
            [-100, -100, 40, 41, 42, 43],
        ]
    )
    attn = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])

    prompt_ids, prompt_mask = extract_prompt_batch(
        input_ids, labels, attn, pad_token_id=0
    )

    # left-padded to the longest prompt (len 3)
    assert prompt_ids.tolist() == [[10, 11, 12], [0, 30, 31]]
    assert prompt_mask.tolist() == [[1, 1, 1], [0, 1, 1]]


def test_left_padding_within_prompt_is_stripped():
    # padding_side="left": leading pads sit before the prompt and must be dropped.
    input_ids = torch.tensor([[0, 0, 5, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, -100, -100, 7, 8]])
    attn = torch.tensor([[0, 0, 1, 1, 1, 1]])

    prompt_ids, prompt_mask = extract_prompt_batch(
        input_ids, labels, attn, pad_token_id=0
    )

    assert prompt_ids.tolist() == [[5, 6]]
    assert prompt_mask.tolist() == [[1, 1]]


def test_no_completion_uses_full_row_as_prompt():
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.full((1, 4), -100)
    attn = torch.ones(1, 4, dtype=torch.long)

    prompt_ids, prompt_mask = extract_prompt_batch(
        input_ids, labels, attn, pad_token_id=0
    )

    assert prompt_ids.tolist() == [[1, 2, 3, 4]]
    assert prompt_mask.tolist() == [[1, 1, 1, 1]]


def test_none_attention_mask_treats_all_attended():
    input_ids = torch.tensor([[1, 2, 3, 9]])
    labels = torch.tensor([[-100, -100, 3, 9]])

    prompt_ids, prompt_mask = extract_prompt_batch(
        input_ids, labels, None, pad_token_id=0
    )

    assert prompt_ids.tolist() == [[1, 2]]
    assert prompt_mask.tolist() == [[1, 1]]


def test_preserves_dtype_and_device():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 3, 4]])
    prompt_ids, prompt_mask = extract_prompt_batch(
        input_ids, labels, None, pad_token_id=7
    )
    assert prompt_ids.dtype == input_ids.dtype
    assert prompt_mask.dtype == torch.long
