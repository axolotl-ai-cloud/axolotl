"""Tests for the flash-attn availability validator."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestFlashAttnAvailabilityValidator:
    """attn_implementation: flash_attention_2/3 requires a loadable flash-attn build."""

    @pytest.fixture(autouse=True)
    def _gpu_present(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    @staticmethod
    def _force_availability(monkeypatch, value: bool):
        import transformers.utils

        monkeypatch.setattr(
            transformers.utils, "is_flash_attn_2_available", lambda **_: value
        )
        monkeypatch.setattr(
            transformers.utils, "is_flash_attn_3_available", lambda **_: value
        )

    def test_fa2_unavailable_raises(self, min_base_cfg, monkeypatch):
        self._force_availability(monkeypatch, False)
        cfg = min_base_cfg | DictDefault(attn_implementation="flash_attention_2")
        with pytest.raises(ValueError, match="no\\s+flash-attn build is available"):
            validate_config(cfg)

    def test_fa3_unavailable_raises(self, min_base_cfg, monkeypatch):
        self._force_availability(monkeypatch, False)
        cfg = min_base_cfg | DictDefault(attn_implementation="flash_attention_3")
        with pytest.raises(ValueError, match="no\\s+flash-attn build is available"):
            validate_config(cfg)

    def test_legacy_flash_attention_flag_unavailable_raises(
        self, min_base_cfg, monkeypatch
    ):
        self._force_availability(monkeypatch, False)
        cfg = min_base_cfg | DictDefault(flash_attention=True)
        with pytest.raises(ValueError, match="no\\s+flash-attn build is available"):
            validate_config(cfg)

    def test_fa2_available_passes(self, min_base_cfg, monkeypatch):
        self._force_availability(monkeypatch, True)
        cfg = min_base_cfg | DictDefault(attn_implementation="flash_attention_2")
        validated = validate_config(cfg)
        assert validated.attn_implementation == "flash_attention_2"

    def test_no_cuda_skips_check(self, min_base_cfg, monkeypatch):
        self._force_availability(monkeypatch, False)
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        cfg = min_base_cfg | DictDefault(attn_implementation="flash_attention_2")
        validated = validate_config(cfg)
        assert validated.attn_implementation == "flash_attention_2"

    def test_sdpa_skips_check(self, min_base_cfg, monkeypatch):
        self._force_availability(monkeypatch, False)
        cfg = min_base_cfg | DictDefault(attn_implementation="sdpa")
        validated = validate_config(cfg)
        assert validated.attn_implementation == "sdpa"
