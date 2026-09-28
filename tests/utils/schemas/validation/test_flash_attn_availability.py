"""Tests for the flash-attn availability validator."""

from unittest.mock import Mock

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
        import kernels
        import transformers.utils

        monkeypatch.setattr(
            transformers.utils, "is_flash_attn_2_available", lambda **_: value
        )
        monkeypatch.setattr(
            transformers.utils, "is_flash_attn_3_available", lambda **_: value
        )
        if not value:
            # a hub lookup that succeeds overrides transformers' verdict, so
            # "unavailable" has to fail there as well
            def no_build(*_, **__):
                raise FileNotFoundError("Cannot find a build variant")

            monkeypatch.setattr(kernels, "get_kernel", no_build)

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

    @pytest.mark.parametrize("attn_version", [2, 3])
    @pytest.mark.parametrize("has_build", [True, False])
    def test_hub_retry_with_unavailable_publisher_status(
        self, min_base_cfg, monkeypatch, attn_version, has_build
    ):
        import kernels
        import transformers.integrations.hub_kernels as hub_kernels
        import transformers.utils

        checker = Mock(return_value=False)
        monkeypatch.setattr(
            transformers.utils, f"is_flash_attn_{attn_version}_available", checker
        )
        monkeypatch.setattr(hub_kernels, "get_attn_kernel_version", lambda _: 1)

        def get_kernel(repo_id, *, version, trust_remote_code=False):
            if not trust_remote_code:
                raise ValueError("could not verify publisher trust status")
            if not has_build:
                raise FileNotFoundError("Cannot find a build variant")
            return object()

        monkeypatch.setattr(kernels, "get_kernel", get_kernel)
        attn_implementation = f"flash_attention_{attn_version}"
        cfg = min_base_cfg | DictDefault(attn_implementation=attn_implementation)
        if has_build:
            validated = validate_config(cfg)
            assert validated.attn_implementation == attn_implementation
            checker.cache_clear.assert_called_once_with()
        else:
            with pytest.raises(ValueError, match="Cannot find a build variant"):
                validate_config(cfg)
            checker.cache_clear.assert_not_called()
