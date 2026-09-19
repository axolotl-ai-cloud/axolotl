"""Tests for multimodal sample packing."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from axolotl.core.builders.causal import HFCausalTrainerBuilder
from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy
from axolotl.utils.collators import MultiModalBatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.samplers.balanced import (
    BALANCED_GREEDY,
    FIRST_FIT_DECREASING,
    balanced_greedy_pack_group,
    default_sample_packing_strategy,
)
from axolotl.utils.samplers.multipack import MultipackBatchSampler
from axolotl.utils.trainer import add_position_ids, filter_sequences_by_length

from tests.mm_packing_utils import PadTokenizer as _Tokenizer


class _ProcessorPrompter:
    processor = object()
    roles: dict[str, str] = {}
    chat_template = None
    chat_template_msg_variables: set[str] = set()
    message_field_training = None
    message_field_training_detail = None
    message_property_mappings = {"role": "role", "content": "content"}
    field_messages = "messages"
    field_system = "system"
    drop_system_message = False
    template_thinking_key = "reasoning_content"

    def build_prompt(self, turns, add_generation_prompt=False, **_kwargs):
        has_image = any(
            isinstance(turn.get("content"), list)
            and any(part.get("type") == "image" for part in turn["content"])
            for turn in turns
        )
        if add_generation_prompt:
            return {"input_ids": [1, 32000, 32000], "attention_mask": [1, 1, 1]}
        if has_image:
            return {
                "input_ids": [1, 32000, 32000, 32000, 32000, 32000, 32000, 2],
                "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1],
                "pixel_values": [[[[1.0]]]],
            }
        return {
            "input_ids": [1, 32000, 32000, 32000, 32000, 2],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "pixel_values": [[[[1.0]]]],
        }


def test_default_sample_packing_strategy_is_balanced_for_multimodal():
    assert default_sample_packing_strategy(True) == BALANCED_GREEDY
    assert default_sample_packing_strategy(False) == FIRST_FIT_DECREASING


def test_balanced_greedy_pack_group_pairs_long_and_short_samples():
    lengths = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)

    bins = balanced_greedy_pack_group(lengths, 0, bin_capacity=10, bin_size=200)

    assert len(bins) == 5
    assert sorted(idx for packed in bins for idx in packed) == list(range(len(lengths)))
    assert all(sum(lengths[idx] for idx in packed) <= 10 for packed in bins)
    assert any(set(packed) == {1, 8} for packed in bins)
    assert any(set(packed) == {4, 5} for packed in bins)


def test_multipack_sampler_accepts_balanced_strategy():
    lengths = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
    sampler = MultipackBatchSampler(
        sampler=list(range(len(lengths))),
        lengths=lengths,
        batch_size=1,
        batch_max_len=10,
        group_size=100,
        bin_size=200,
        drop_last=False,
        packing_strategy=BALANCED_GREEDY,
    )

    packs = [pack for batch in sampler.generate_batches() for pack in batch]

    assert any(set(pack) == {1, 8} for pack in packs)
    assert all(sum(lengths[idx] for idx in pack) <= 10 for pack in packs)


def test_processor_backed_chat_template_emits_post_processor_length():
    strategy = ChatTemplateStrategy(
        _ProcessorPrompter(),
        _Tokenizer(),
        train_on_inputs=False,
        sequence_len=128,
        train_on_eos=None,
        train_on_eot=None,
    )

    row = strategy.tokenize_prompt(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image"},
                    ],
                },
                {"role": "assistant", "content": "image"},
            ]
        }
    )

    assert row["length"] == len(row["input_ids"]) == 8


def test_builder_uses_packed_multimodal_collator_for_sample_packing():
    cfg = SimpleNamespace(
        model_config_type="llama",
        plugins=None,
        processor_type="AutoProcessor",
        reward_model=False,
    )
    training_args = SimpleNamespace(
        pretraining=False,
        sample_packing=True,
        eval_sample_packing=False,
    )
    builder = HFCausalTrainerBuilder(
        cfg=cfg, model=object(), tokenizer=_Tokenizer(), processor=object()
    )

    collator = builder.build_collator(training_args, padding=True)

    assert isinstance(collator, MultiModalBatchSamplerDataCollatorForSeq2Seq)


def test_explicit_packing_length_filters_and_survives_position_ids():
    sample = {"input_ids": [[1, 2], [1, 2, 3]], "length": [5, 3]}

    assert filter_sequences_by_length(sample, sequence_len=4) == [False, True]
    assert add_position_ids(sample)["length"] == [5, 3]


def test_multimodal_packed_collator_preserves_media_shapes():
    # One packed group (mbs=1). Tiled VLM (Idefics3/SmolVLM) pixel_values keep their
    # 5D (batch, num_images, C, H, W) layout with images concatenated across the pack.
    collator = MultiModalBatchSamplerDataCollatorForSeq2Seq(
        tokenizer=_Tokenizer(),
        padding=True,
        return_tensors="pt",
    )
    features = [
        [
            {
                "input_ids": [11, 12, 13],
                "labels": [-100, 12, 13],
                "attention_mask": [1, 1, 1],
                "position_ids": [0, 1, 2],
                "pixel_values": np.ones((1, 1, 3, 2, 2), dtype=np.float32),
                "image_grid_thw": [1, 2, 3],
                "token_span_mask": np.ones((3, 2), dtype=np.int64),
            },
            {
                "input_ids": [21, 22],
                "labels": [-100, 22],
                "attention_mask": [1, 1],
                "position_ids": [0, 1],
                "pixel_values": np.full((1, 1, 3, 2, 2), 2.0, dtype=np.float32),
                "image_grid_thw": [1, 1, 2],
                "token_span_mask": np.zeros((2, 2), dtype=np.int64),
            },
            {
                "input_ids": [31, 32, 33, 34],
                "labels": [-100, -100, 33, 34],
                "attention_mask": [1, 1, 1, 1],
                "position_ids": [0, 1, 2, 3],
                "pixel_values": np.full((1, 1, 3, 2, 2), 3.0, dtype=np.float32),
                "image_grid_thw": [1, 2, 2],
                "token_span_mask": np.ones((4, 2), dtype=np.int64),
            },
        ],
    ]

    batch = collator(features)

    assert batch["input_ids"].shape == (1, 9)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 2, 2, 3, 3, 3, 3]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 0, 1, 2, 3]]
    assert batch["pixel_values"].shape == (1, 3, 3, 2, 2)
    assert batch["image_grid_thw"].tolist() == [[1, 2, 3], [1, 1, 2], [1, 2, 2]]
    assert batch["token_span_mask"].shape == (1, 9, 2)


def test_multimodal_packed_collator_concatenates_patch_pixel_values():
    collator = MultiModalBatchSamplerDataCollatorForSeq2Seq(
        tokenizer=_Tokenizer(),
        padding=True,
        return_tensors="pt",
    )
    features = [
        [
            {
                "input_ids": [11],
                "labels": [11],
                "attention_mask": [1],
                "position_ids": [0],
                "pixel_values": np.ones((2, 3), dtype=np.float32),
                "image_grid_thw": [1, 1, 2],
            },
            {
                "input_ids": [21],
                "labels": [21],
                "attention_mask": [1],
                "position_ids": [0],
                "pixel_values": np.full((1, 3), 2.0, dtype=np.float32),
                "image_grid_thw": [1, 1, 1],
            },
        ]
    ]

    batch = collator(features)

    assert batch["pixel_values"].shape == (3, 3)
    assert batch["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 1]]
