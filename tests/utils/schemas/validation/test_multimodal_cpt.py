"""Multimodal CPT config validation gates."""

from __future__ import annotations

import logging

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def _mm_cpt_cfg(min_base_cfg, **overrides) -> DictDefault:
    base = DictDefault(
        **(
            min_base_cfg
            | {
                "datasets": None,
                "pretraining_dataset": [
                    {
                        "path": "some/ds",
                        "type": "multimodal_pretrain",
                        "image_column": "images",
                    }
                ],
                "streaming": True,
                "max_steps": 10,
                "processor_type": "AutoProcessor",
                "sequence_len": 2048,
            }
        )
    )
    return base | DictDefault(overrides)


class TestMultimodalCPTGates:
    def test_missing_processor_type_raises(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pop("processor_type", None)
        with pytest.raises(ValueError, match="processor_type"):
            validate_config(cfg)

    def test_sample_packing_rejected(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg, sample_packing=True)
        with pytest.raises(ValueError, match="sample_packing"):
            validate_config(cfg)

    def test_chat_template_rejected(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg, chat_template="tokenizer_default")
        with pytest.raises(ValueError, match="chat_template"):
            validate_config(cfg)

    def test_multiple_pretraining_dataset_entries_rejected(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pretraining_dataset.append({"path": "other/ds", "type": "pretrain"})
        with pytest.raises(ValueError, match="exactly one `pretraining_dataset`"):
            validate_config(cfg)

    def test_multimodal_entry_in_non_first_slot_rejected(self, min_base_cfg):
        cfg = DictDefault(
            **(
                min_base_cfg
                | {
                    "datasets": None,
                    "pretraining_dataset": [
                        {"path": "text/ds", "type": "pretrain"},
                        {
                            "path": "mm/ds",
                            "type": "multimodal_pretrain",
                            "image_column": "images",
                        },
                    ],
                    "streaming": True,
                    "max_steps": 10,
                    "processor_type": "AutoProcessor",
                    "sequence_len": 2048,
                }
            )
        )
        with pytest.raises(ValueError, match="exactly one `pretraining_dataset`"):
            validate_config(cfg)

    def test_valid_cfg_passes_and_disables_remove_unused_columns(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        validated = validate_config(cfg)
        assert validated.remove_unused_columns is False
        pd = validated.pretraining_dataset[0]
        assert pd.type == "multimodal_pretrain"
        assert pd.image_column == "images"

    def test_multimodal_flag_triggers_gates(self, min_base_cfg):
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pretraining_dataset[0]["type"] = "pretrain"
        cfg.pretraining_dataset[0]["multimodal"] = True
        cfg.pop("processor_type", None)
        with pytest.raises(ValueError, match="processor_type"):
            validate_config(cfg)

    def test_non_mm_pretraining_dataset_unaffected(self, min_base_cfg):
        cfg = DictDefault(
            **(
                min_base_cfg
                | {
                    "datasets": None,
                    "pretraining_dataset": [{"path": "some/ds", "type": "pretrain"}],
                    "streaming": True,
                    "max_steps": 10,
                    "sequence_len": 2048,
                }
            )
        )
        validate_config(cfg)

    def test_mm_eval_dataset_keys_preserved_through_validation(self, min_base_cfg):
        """MM-specific keys on a test_datasets entry survive validate_config."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/ds",
                    "type": "multimodal_pretrain",
                    "text_column": "eval_text",
                    "image_column": "eval_imgs",
                    "image_base_dir": "/eval/images",
                    "image_token": "<my_img>",
                }
            ],
        )
        validated = validate_config(cfg)
        td = validated.test_datasets[0]
        assert td["text_column"] == "eval_text"
        assert td["image_column"] == "eval_imgs"
        assert td["image_base_dir"] == "/eval/images"
        assert td["image_token"] == "<my_img>"

    def test_mm_eval_dataset_via_multimodal_flag(self, min_base_cfg):
        """`multimodal: true` (without type='multimodal_pretrain') opts an eval entry into MM."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/ds",
                    "multimodal": True,
                    "image_column": "imgs2",
                }
            ],
        )
        validated = validate_config(cfg)
        td = validated.test_datasets[0]
        assert td["image_column"] == "imgs2"
        assert td["multimodal"] is True

    def test_non_mm_eval_entry_does_not_match_mm_model(self, min_base_cfg):
        """SFT eval entries (no MM markers) still validate as SFTDataset."""
        cfg = DictDefault(
            **(
                min_base_cfg
                | {
                    "test_datasets": [
                        {"path": "eval/ds", "type": "alpaca", "split": "test"}
                    ],
                    "sequence_len": 2048,
                }
            )
        )
        validated = validate_config(cfg)
        td = validated.test_datasets[0]
        assert "message_property_mappings" in td
        assert td["type"] == "alpaca"

    def test_mm_eval_rejects_mismatched_image_base_dir(self, min_base_cfg):
        """Multiple MM eval entries with different image_base_dir are rejected."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/a",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/images/a",
                },
                {
                    "path": "eval/b",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/images/b",
                },
            ],
        )
        with pytest.raises(ValueError, match="image_base_dir"):
            validate_config(cfg)

    def test_mm_eval_rejects_mismatched_image_token(self, min_base_cfg):
        """Multiple MM eval entries with different image_token overrides are rejected."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/a",
                    "type": "multimodal_pretrain",
                    "image_token": "<img_a>",
                },
                {
                    "path": "eval/b",
                    "type": "multimodal_pretrain",
                    "image_token": "<img_b>",
                },
            ],
        )
        with pytest.raises(ValueError, match="image_token"):
            validate_config(cfg)

    def test_mm_eval_accepts_matching_image_base_dir(self, min_base_cfg):
        """Multiple MM eval entries sharing image_base_dir validate cleanly."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {
                    "path": "eval/a",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/images/shared",
                },
                {
                    "path": "eval/b",
                    "type": "multimodal_pretrain",
                    "image_base_dir": "/images/shared",
                },
            ],
        )
        validated = validate_config(cfg)
        assert len(validated.test_datasets) == 2

    def test_mm_eval_accepts_all_unset_image_settings(self, min_base_cfg):
        """Multiple MM eval entries with image_base_dir / image_token unset everywhere validate."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {"path": "eval/a", "type": "multimodal_pretrain"},
                {"path": "eval/b", "type": "multimodal_pretrain"},
            ],
        )
        validated = validate_config(cfg)
        assert len(validated.test_datasets) == 2

    def test_mixed_modality_test_datasets_rejected_at_validation(self, min_base_cfg):
        """A test_datasets list mixing MM and non-MM entries fails at config-load."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[
                {"path": "eval/a", "type": "multimodal_pretrain"},
                {"path": "eval/b", "type": "alpaca", "split": "test"},
            ],
        )
        with pytest.raises(ValueError) as exc:
            validate_config(cfg)
        msg = str(exc.value)
        assert "Mixing multimodal and non-multimodal" in msg
        assert "test_datasets" in msg
        assert "share modality" in msg

    def test_mm_test_datasets_with_text_training_rejected(self, min_base_cfg):
        """MM test_datasets paired with non-MM training fails at config-load."""
        cfg = DictDefault(
            **(
                min_base_cfg
                | {
                    "datasets": None,
                    "pretraining_dataset": [{"path": "text/ds", "type": "pretrain"}],
                    "test_datasets": [
                        {"path": "eval/a", "type": "multimodal_pretrain"}
                    ],
                    "streaming": True,
                    "max_steps": 10,
                    "sequence_len": 2048,
                    "processor_type": "AutoProcessor",
                }
            )
        )
        with pytest.raises(ValueError) as exc:
            validate_config(cfg)
        msg = str(exc.value)
        assert "Multimodal `test_datasets`" in msg
        assert "multimodal CPT training" in msg
        assert "multimodal_pretrain" in msg

    def test_text_test_datasets_with_mm_training_rejected(self, min_base_cfg):
        """Non-MM test_datasets paired with MM training fails at config-load."""
        cfg = _mm_cpt_cfg(
            min_base_cfg,
            test_datasets=[{"path": "eval/a", "type": "alpaca", "split": "test"}],
        )
        with pytest.raises(ValueError) as exc:
            validate_config(cfg)
        msg = str(exc.value)
        assert "Multimodal CPT training" in msg
        assert "multimodal `test_datasets`" in msg
        assert "multimodal_pretrain" in msg

    def test_remove_unused_columns_auto_set_emits_info_log(
        self, min_base_cfg, caplog, monkeypatch
    ):
        """Auto-setting `remove_unused_columns: false` for MM CPT logs an INFO record naming the previous value."""
        # `axolotl` logger has propagate=False (logging_config.py); flip it so
        # caplog's root handler receives the record.
        monkeypatch.setattr(logging.getLogger("axolotl"), "propagate", True)
        cfg = _mm_cpt_cfg(min_base_cfg)
        cfg.pop("remove_unused_columns", None)
        with caplog.at_level(logging.INFO, logger="axolotl.utils.schemas.validation"):
            validated = validate_config(cfg)
        assert validated.remove_unused_columns is False
        matches = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "Auto-set" in r.getMessage()
        ]
        assert matches, (
            "expected an INFO record about auto-setting remove_unused_columns"
        )
        msg = matches[0].getMessage()
        assert "remove_unused_columns" in msg
        assert "previous value: None" in msg

    def test_remove_unused_columns_already_false_does_not_log(
        self, min_base_cfg, caplog, monkeypatch
    ):
        """When the user already set `remove_unused_columns: false`, no auto-set log fires."""
        monkeypatch.setattr(
            logging.getLogger("axolotl.utils.schemas.validation"), "propagate", True
        )
        cfg = _mm_cpt_cfg(min_base_cfg, remove_unused_columns=False)
        with caplog.at_level(logging.INFO, logger="axolotl.utils.schemas.validation"):
            validate_config(cfg)
        assert not any(
            "Auto-set" in r.getMessage() and "remove_unused_columns" in r.getMessage()
            for r in caplog.records
        )
