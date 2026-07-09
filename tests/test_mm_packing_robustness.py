"""Robustness tests for multimodal (VLM) sample packing: over-long media rows must not be
silently truncated, and streaming/pretraining packing routes to the buffered packer."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.data.utils import truncate_long_seq
from axolotl.utils.dict import DictDefault


class TestTruncateLongSeqMultimodal:
    """truncate_long_seq guards multimodal rows."""

    def test_multimodal_row_over_length_raises(self):
        sample = {
            "input_ids": [list(range(10))],
            "labels": [list(range(10))],
            "pixel_values": [[[0.0, 1.0], [2.0, 3.0]]],
            "length": [10],
        }
        with pytest.raises(ValueError, match="(?i)multimodal"):
            truncate_long_seq(sample, sequence_len=4)

    def test_image_grid_thw_over_length_raises(self):
        sample = {
            "input_ids": [list(range(10))],
            "image_grid_thw": [[1, 2, 2]],
        }
        with pytest.raises(ValueError, match="(?i)desync"):
            truncate_long_seq(sample, sequence_len=4)

    def test_text_only_row_still_truncates(self):
        # No media columns -> unchanged behavior: over-long text rows get sliced.
        sample = {
            "input_ids": [list(range(10))],
            "labels": [list(range(10))],
            "length": [10],
        }
        out = truncate_long_seq(sample, sequence_len=4)
        assert out["input_ids"][0] == [0, 1, 2, 3]
        assert out["labels"][0] == [0, 1, 2, 3]
        assert out["length"][0] == 4

    def test_multimodal_row_within_length_untouched(self):
        sample = {
            "input_ids": [list(range(4))],
            "pixel_values": [[[0.0, 1.0]]],
        }
        out = truncate_long_seq(sample, sequence_len=8)
        assert out["input_ids"][0] == [0, 1, 2, 3]

    def test_mixed_batch_text_truncates_media_raises(self):
        # Text row is over-length (fine), media row is over-length (must raise).
        sample = {
            "input_ids": [list(range(10)), list(range(10))],
            "pixel_values": [None, [[0.0, 1.0]]],
        }
        with pytest.raises(ValueError):
            truncate_long_seq(sample, sequence_len=4)


def _cfg(**extra):
    return DictDefault(
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
        learning_rate=1e-3,
        datasets=[
            {
                "path": "HuggingFaceH4/llava-instruct-mix-vsft",
                "type": "chat_template",
            }
        ],
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        sequence_len=2048,
        **extra,
    )


class TestMultimodalPackingStreaming:
    """check_mm_sample_packing_streaming in DatasetValidationMixin.

    Streaming multimodal packing is served by the buffered packer, so it
    validates; the `pretraining_dataset` route lacks that packer and raises.
    """

    def test_mm_packing_with_streaming_ok(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            streaming=True,
            max_steps=100,
        )
        out = validate_config(cfg)
        assert out.remove_unused_columns is False

    def test_mm_packing_with_pretraining_dataset_raises(self):
        # pretraining_dataset routes to the text streaming loader (no buffered MM
        # packer), which would silently drop media -> blocked at validation.
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            pretraining_dataset=[{"path": "allenai/c4", "name": "en"}],
            streaming=True,
            max_steps=100,
        )
        with pytest.raises(Exception, match="(?i)pretraining_dataset"):
            validate_config(cfg)

    def test_mm_eval_packing_with_streaming_ok(self):
        cfg = _cfg(
            is_multimodal=True,
            eval_sample_packing=True,
            streaming=True,
            max_steps=100,
        )
        validate_config(cfg)

    def test_non_mm_packing_streaming_unaffected(self):
        # No processor_type / is_multimodal -> streaming validator is a no-op here.
        cfg = _cfg(sample_packing=True, streaming=True, max_steps=100)
        out = validate_config(cfg)  # must not raise
        assert out.streaming is True

    def test_mm_packing_non_streaming_unaffected(self):
        cfg = _cfg(processor_type="AutoProcessor", sample_packing=True)
        out = validate_config(cfg)  # must not raise
        assert out.remove_unused_columns is False


def test_cross_attention_mask_packing_rejected():
    """mllama-style cross_attention_mask in a multi-sample pack must raise."""
    import numpy as np
    import pytest
    import torch

    from axolotl.utils.collators import MultiModalBatchSamplerDataCollatorForSeq2Seq

    class _Tok:
        padding_side = "right"
        pad_token_id = 0

        def pad(self, feats, **_k):
            L = max(len(f["input_ids"]) for f in feats)
            return {
                k: torch.tensor([list(f[k]) + [0] * (L - len(f[k])) for f in feats])
                for k in set().union(*(f.keys() for f in feats))
            }

    def row(off):
        return {
            "input_ids": [off, off + 1],
            "labels": [off, off + 1],
            "attention_mask": [1, 1],
            "position_ids": [0, 1],
            "pixel_values": np.ones((1, 1, 4, 3, 8, 8), np.float32),
            "cross_attention_mask": np.ones((2, 1, 4), np.int64),
        }

    col = MultiModalBatchSamplerDataCollatorForSeq2Seq(
        tokenizer=_Tok(), padding=True, return_tensors="pt"
    )
    with pytest.raises(ValueError, match="cross-attention"):
        col([[row(10), row(20)]])
