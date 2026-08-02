from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.core.trainers.base import (
    AxolotlTrainer,
    _save_pretrained_supports_is_main_process,
)


def _make_trainer(tmp_path, is_main_process=True):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    trainer = AxolotlTrainer.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    trainer.axolotl_cfg = None
    trainer.model = LlamaForCausalLM(config)
    trainer.accelerator = SimpleNamespace(is_main_process=is_main_process)
    trainer.processing_class = None
    trainer.data_collator = None
    return trainer


def test_save_pretrained_supports_is_main_process():
    assert _save_pretrained_supports_is_main_process() is True


def test_save_forwards_is_main_process(tmp_path):
    trainer = _make_trainer(tmp_path)
    with patch.object(LlamaForCausalLM, "save_pretrained", new=MagicMock()) as m:
        trainer._save()
    m.assert_called_once_with(str(tmp_path), state_dict=None, is_main_process=True)


def test_save_forwards_is_main_process_false(tmp_path):
    trainer = _make_trainer(tmp_path, is_main_process=False)
    with patch.object(LlamaForCausalLM, "save_pretrained", new=MagicMock()) as m:
        trainer._save()
    m.assert_called_once_with(str(tmp_path), state_dict=None, is_main_process=False)


def test_save_skips_is_main_process_when_unsupported(tmp_path):
    trainer = _make_trainer(tmp_path)
    with (
        patch(
            "axolotl.core.trainers.base._save_pretrained_supports_is_main_process",
            return_value=False,
        ),
        patch.object(LlamaForCausalLM, "save_pretrained", new=MagicMock()) as m,
    ):
        trainer._save()
    m.assert_called_once_with(str(tmp_path), state_dict=None)
