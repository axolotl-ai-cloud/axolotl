"""Config-validation tests for multimodal (VLM) sample packing."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


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


class TestMultimodalSamplePacking:
    """check_mm_sample_packing in DatasetValidationMixin."""

    def test_mm_packing_with_skip_prepare_ok_buffered_path(self):
        # skip_prepare_dataset now routes to the buffered MM packer (no raise).
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            skip_prepare_dataset=True,
            remove_unused_columns=False,
        )
        out = validate_config(cfg)
        assert out.remove_unused_columns is False

    def test_mm_eval_packing_with_skip_prepare_ok(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            eval_sample_packing=True,
            skip_prepare_dataset=True,
        )
        out = validate_config(cfg)
        assert out.remove_unused_columns is False

    def test_mm_packing_via_is_multimodal_flag_ok(self):
        cfg = _cfg(
            is_multimodal=True,
            sample_packing=True,
            skip_prepare_dataset=True,
        )
        out = validate_config(cfg)
        assert out.sample_packing is True

    def test_mm_packing_streaming_ok_buffered_path(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            streaming=True,
            max_steps=100,
        )
        out = validate_config(cfg)
        assert out.remove_unused_columns is False

    def test_mm_packing_streaming_skip_prepare_ok(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            streaming=True,
            skip_prepare_dataset=True,
            max_steps=100,
        )
        out = validate_config(cfg)
        assert out.remove_unused_columns is False

    def test_mm_packing_without_skip_prepare_ok_and_sets_remove_unused(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
        )
        out = validate_config(cfg)
        assert out.remove_unused_columns is False

    def test_mm_packing_skip_prepare_false_explicit_ok(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            skip_prepare_dataset=False,
            remove_unused_columns=False,
        )
        out = validate_config(cfg)
        assert out.remove_unused_columns is False

    def test_mm_packing_remove_unused_true_raises(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            remove_unused_columns=True,
        )
        with pytest.raises(Exception, match="(?i)remove_unused_columns"):
            validate_config(cfg)

    def test_non_mm_packing_unaffected(self):
        # No processor_type / is_multimodal -> the MM packing validator is a no-op.
        cfg = _cfg(sample_packing=True, skip_prepare_dataset=True)
        out = validate_config(cfg)
        assert out.sample_packing is True


class TestMultimodalPretrainingPacking:
    """check_mm_sample_packing_streaming + streaming-MM num_workers handling."""

    def test_mm_packing_with_pretraining_dataset_raises(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            pretraining_dataset=[{"path": "some/pretrain", "type": "pretrain"}],
            max_steps=100,
        )
        with pytest.raises(Exception, match="(?i)pretraining_dataset"):
            validate_config(cfg)

    def test_mm_eval_packing_with_pretraining_dataset_raises(self):
        cfg = _cfg(
            is_multimodal=True,
            eval_sample_packing=True,
            pretraining_dataset=[{"path": "some/pretrain", "type": "pretrain"}],
            max_steps=100,
        )
        with pytest.raises(Exception, match="(?i)pretraining_dataset"):
            validate_config(cfg)

    def test_mm_streaming_packing_still_ok(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            streaming=True,
            max_steps=100,
        )
        out = validate_config(cfg)
        assert out.remove_unused_columns is False

    def test_streaming_mm_packing_forces_num_workers_zero(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            streaming=True,
            max_steps=100,
        )
        out = validate_config(cfg, {"n_gpu": 8}, {"torch_version": "2.6.0"})
        assert out.dataloader_num_workers == 0
        assert out.dataloader_prefetch_factor is None

    def test_non_mm_pretraining_dataset_unaffected(self):
        cfg = _cfg(
            sample_packing=True,
            pretraining_dataset=[{"path": "some/pretrain", "type": "pretrain"}],
            max_steps=100,
        )
        out = validate_config(cfg)
        assert out.sample_packing is True
