"""Routing tests for `skip_prepare_dataset` multimodal sample packing."""

from __future__ import annotations

from unittest.mock import Mock, patch

from datasets import Dataset

from axolotl.utils.data.mm_streaming import BufferedMultimodalPacker
from axolotl.utils.data.sft import prepare_datasets
from axolotl.utils.dict import DictDefault


def _mm_cfg(**extra):
    return DictDefault(
        {
            "processor_type": "AutoProcessor",
            "sample_packing": True,
            "skip_prepare_dataset": True,
            "sequence_len": 256,
            "datasets": [
                {"path": "test/dataset", "type": "chat_template", "split": "train"}
            ],
            **extra,
        }
    )


@patch("axolotl.processing_strategies.get_processing_strategy")
@patch("axolotl.utils.data.sft.load_dataset")
def test_skip_prepare_mm_routes_to_buffered_packer(mock_load, mock_strategy):
    # A tiny materialized (non-streaming) map dataset stands in for a real download.
    mock_load.return_value = Dataset.from_dict({"messages": [[], []]})
    mock_strategy.return_value = Mock()

    train_dataset, eval_dataset, total_num_steps, prompters = prepare_datasets(
        _mm_cfg(), Mock(), processor=Mock()
    )

    assert isinstance(train_dataset, BufferedMultimodalPacker)
    assert eval_dataset is None
    assert total_num_steps == -1
    assert prompters == []
    # The raw dataset must be loaded materialized (streaming=False), not streamed.
    assert mock_load.call_args.kwargs["streaming"] is False


@patch("axolotl.utils.data.sft._prepare_standard_dataset")
def test_skip_prepare_mm_without_packing_uses_eager_path(mock_standard):
    mock_standard.return_value = (Mock(), None, 100, [])
    cfg = _mm_cfg(sample_packing=False)

    prepare_datasets(cfg, Mock(), processor=Mock())

    mock_standard.assert_called_once()


@patch("axolotl.utils.data.sft._prepare_standard_dataset")
def test_skip_prepare_without_processor_uses_eager_path(mock_standard):
    mock_standard.return_value = (Mock(), None, 100, [])
    cfg = _mm_cfg()

    prepare_datasets(cfg, Mock(), processor=None)

    mock_standard.assert_called_once()
