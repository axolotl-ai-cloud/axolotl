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

    def test_mm_packing_with_skip_prepare_raises(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
            skip_prepare_dataset=True,
            remove_unused_columns=False,
        )
        with pytest.raises(Exception, match="(?i)pre-tokeniz"):
            validate_config(cfg)

    def test_mm_eval_packing_with_skip_prepare_raises(self):
        # eval_sample_packing alone also triggers the requirement
        cfg = _cfg(
            processor_type="AutoProcessor",
            eval_sample_packing=True,
            skip_prepare_dataset=True,
        )
        with pytest.raises(Exception, match="(?i)skip_prepare_dataset"):
            validate_config(cfg)

    def test_mm_packing_via_is_multimodal_flag_raises(self):
        cfg = _cfg(
            is_multimodal=True,
            sample_packing=True,
            skip_prepare_dataset=True,
        )
        with pytest.raises(Exception, match="(?i)multimodal sample packing"):
            validate_config(cfg)

    def test_mm_packing_without_skip_prepare_ok_and_sets_remove_unused(self):
        cfg = _cfg(
            processor_type="AutoProcessor",
            sample_packing=True,
        )
        out = validate_config(cfg)  # must not raise
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
        out = validate_config(cfg)  # must not raise
        assert out.sample_packing is True
