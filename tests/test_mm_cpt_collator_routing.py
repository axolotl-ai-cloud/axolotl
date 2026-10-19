"""build_collator MM CPT routing + real streaming .map end-to-end tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor

from axolotl.core.builders.causal import HFCausalTrainerBuilder
from axolotl.utils.collators import DataCollatorForSeq2Seq
from axolotl.utils.collators.mm_pretrain import MultiModalPretrainDataCollator
from axolotl.utils.data.streaming import wrap_streaming_dataset
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline

_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"


@pytest.fixture(scope="module", name="smolvlm_processor")
@enable_hf_offline
def fixture_smolvlm_processor(
    download_smolvlm_500m_instruct_model,  # pylint: disable=unused-argument
):
    return AutoProcessor.from_pretrained(_SMOLVLM)


def _make_builder(smolvlm_processor, cfg_overrides=None):
    builder = HFCausalTrainerBuilder.__new__(HFCausalTrainerBuilder)
    builder.tokenizer = smolvlm_processor.tokenizer
    builder.processor = smolvlm_processor
    cfg = {
        "processor_type": "AutoProcessor",
        "pretraining_dataset": [{"path": "some/ds", "type": "multimodal_pretrain"}],
        "sequence_len": 512,
    }
    cfg.update(cfg_overrides or {})
    builder.cfg = DictDefault(cfg)
    return builder


def test_build_collator_routes_mm_cpt_to_mm_collator(smolvlm_processor):
    """The pretraining branch must hand MM CPT runs the MM collator."""
    builder = _make_builder(smolvlm_processor)
    collator = builder.build_collator(SimpleNamespace(pretraining=True))
    assert isinstance(collator, MultiModalPretrainDataCollator)


def test_build_collator_text_pretraining_unaffected(smolvlm_processor):
    """Non-MM pretraining keeps the text collator path."""
    builder = _make_builder(
        smolvlm_processor,
        cfg_overrides={
            "pretraining_dataset": [{"path": "some/ds", "type": "pretrain"}],
            "pretraining_sample_concatenation": False,
        },
    )
    collator = builder.build_collator(SimpleNamespace(pretraining=True))
    assert isinstance(collator, DataCollatorForSeq2Seq)
    assert not isinstance(collator, MultiModalPretrainDataCollator)


def test_build_collator_no_processor_falls_through(smolvlm_processor):
    """MM entry without a processor must not reach the MM collator."""
    builder = _make_builder(smolvlm_processor)
    builder.processor = None
    builder.cfg = DictDefault(
        dict(builder.cfg) | {"pretraining_sample_concatenation": False}
    )
    collator = builder.build_collator(SimpleNamespace(pretraining=True))
    assert not isinstance(collator, MultiModalPretrainDataCollator)


def test_streaming_map_pipeline_end_to_end(smolvlm_processor, tmp_path):
    """3-row jsonl + 2 tiny PNGs through the real datasets .map pipeline and the
    build_collator-routed collator: one real batch out."""
    imgs = []
    for i in range(2):
        p = tmp_path / f"img_{i}.png"
        arr = np.random.default_rng(i).integers(0, 255, (32, 32, 3)).astype("uint8")
        Image.fromarray(arr).save(p)
        imgs.append(p.name)

    rows = [
        {"text": "<image>\ncaption one", "images": [imgs[0]]},
        {"text": "plain text row", "images": []},
        {"text": "<image>\ncaption two", "images": [imgs[1]]},
    ]
    data_file = tmp_path / "shard.jsonl"
    data_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    entry = {
        "path": str(data_file),
        "type": "multimodal_pretrain",
        "text_column": "text",
        "image_column": "images",
        "image_base_dir": str(tmp_path),
    }
    cfg = DictDefault(
        {
            "processor_type": "AutoProcessor",
            "pretraining_dataset": [entry],
            "sequence_len": 2048,
            "sample_packing": False,
            "shuffle_merged_datasets": False,
            "streaming_multipack_buffer_size": 8,
            "pretraining_sample_concatenation": False,
            "seed": 42,
        }
    )

    dataset = load_dataset(
        "json", data_files=str(data_file), split="train", streaming=True
    )
    wrapped = wrap_streaming_dataset(
        dataset,
        smolvlm_processor.tokenizer,
        cfg,
        ds_wrapper_fn=None,
        processor=smolvlm_processor,
        pretraining_config=entry,
    )
    encoded_rows = list(wrapped)
    assert len(encoded_rows) == 3
    for row in encoded_rows:
        assert "_mm_text" in row and "images" in row

    builder = HFCausalTrainerBuilder.__new__(HFCausalTrainerBuilder)
    builder.tokenizer = smolvlm_processor.tokenizer
    builder.processor = smolvlm_processor
    builder.cfg = cfg
    collator = builder.build_collator(SimpleNamespace(pretraining=True))
    assert isinstance(collator, MultiModalPretrainDataCollator)

    batch = collator.torch_call(encoded_rows)
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["input_ids"].shape[0] == 3
    assert "pixel_values" in batch
    image_token_id = collator.image_token_spec.image_token_id
    assert int((batch["labels"] == image_token_id).sum().item()) == 0
    assert int((batch["input_ids"] == image_token_id).sum().item()) > 0
