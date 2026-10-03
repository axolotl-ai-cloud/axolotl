import json
from pathlib import Path

import pytest
import torch

from axolotl.integrations.diffusion.lm.attention import (
    decoder_flex_block_mask,
    decoder_prefix_canvas_mask,
    encoder_causal_mask,
)


def test_sliding_masks_window_and_canvas_bidirectionality():
    docs = torch.zeros((1, 5), dtype=torch.long)
    valid = torch.ones_like(docs, dtype=torch.bool)
    positions = torch.arange(5).unsqueeze(0)
    encoder = encoder_causal_mask(docs, valid, positions, sliding_window=2)
    assert encoder[0, 0, 4, 3]
    assert not encoder[0, 0, 4, 2]
    canvas_docs = torch.zeros((1, 2), dtype=torch.long)
    canvas_positions = torch.tensor([[4, 5]])
    decoder = decoder_prefix_canvas_mask(
        docs,
        canvas_docs,
        valid,
        torch.ones_like(canvas_docs, dtype=torch.bool),
        torch.tensor([5]),
        positions,
        canvas_positions,
        sliding_window=2,
    )
    assert not decoder[0, 0, 1, 0]
    assert decoder[0, 0, 0, 6]
    assert decoder[0, 0, 1, 5]


def test_decoder_masks_match_pinned_nemo_selected_block_fixture():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures" / "nemo_packed_attention_f21252.json"
        ).read_text()
    )
    encoder_docs = torch.tensor([[0] * 9 + [1] * 5])
    canvas_docs = torch.tensor([[0] * 3 + [1] * 3])
    encoder_positions = torch.tensor([list(range(9)) + list(range(5))])
    canvas_positions = torch.tensor([[5, 6, 7, 1, 2, 3]])
    valid_encoder = torch.ones_like(encoder_docs, dtype=torch.bool)
    valid_canvas = torch.ones_like(canvas_docs, dtype=torch.bool)
    arguments = (
        encoder_docs,
        canvas_docs,
        valid_encoder,
        valid_canvas,
        torch.tensor(fixture["decoder_prefix_lengths"]),
        encoder_positions,
        canvas_positions,
    )
    full = decoder_prefix_canvas_mask(*arguments)
    sliding = decoder_prefix_canvas_mask(
        *arguments, sliding_window=fixture["sliding_window"]
    )
    torch.testing.assert_close(full[0, 0], torch.tensor(fixture["full_mask"]))
    torch.testing.assert_close(sliding[0, 0], torch.tensor(fixture["sliding_mask"]))


@pytest.mark.parametrize("window", [None, 3])
def test_decoder_flex_selected_block_visibility(window):
    docs = torch.tensor([[0] * 8 + [1] * 4 + [-1] * 2])
    positions = torch.tensor([list(range(8)) + list(range(4)) + [0, 0]])
    canvas_docs = torch.tensor([[0, 0, 1, 1, -1]])
    prefixes = [5, 2]
    block = decoder_flex_block_mask(
        docs,
        canvas_docs,
        docs >= 0,
        canvas_docs >= 0,
        torch.tensor(prefixes),
        positions,
        sliding_window=window,
        logical_ids=torch.tensor([0, 1]),
    )
    query_length = canvas_docs.shape[1]
    encoder_length = docs.shape[1]
    key_length = encoder_length + query_length
    query = torch.arange(query_length)[:, None].expand(-1, key_length)
    key = torch.arange(key_length)[None, :].expand(query_length, -1)
    actual = block.mask_mod(
        torch.zeros_like(query), torch.zeros_like(query), query, key
    )
    expected = torch.zeros((query_length, key_length), dtype=torch.bool)
    for q, document in enumerate(canvas_docs[0].tolist()):
        if document < 0:
            continue
        prefix = prefixes[document]
        for k in range(key_length):
            if k < encoder_length:
                position = int(positions[0, k])
                expected[q, k] = (
                    int(docs[0, k]) == document
                    and position < prefix
                    and (window is None or position >= prefix - window + 1)
                )
            else:
                expected[q, k] = int(canvas_docs[0, k - encoder_length]) == document
    torch.testing.assert_close(actual, expected)
