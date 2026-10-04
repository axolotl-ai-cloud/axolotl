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

        import axolotl.utils.schemas.validation as validation

        monkeypatch.setattr(
            transformers.utils, "is_flash_attn_2_available", lambda **_: value
        )
        monkeypatch.setattr(
            transformers.utils, "is_flash_attn_3_available", lambda **_: value
        )
        if not value:
            # a hub lookup that succeeds overrides transformers' verdict, so
            # "unavailable" has to fail there as well, online and from the cache
            def no_build(*_, **__):
                raise FileNotFoundError("Cannot find a build variant")

            monkeypatch.setattr(kernels, "get_kernel", no_build)
            monkeypatch.setattr(validation, "_get_kernel_from_cache", no_build)

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

        import axolotl.utils.schemas.validation as validation

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

        def no_cached_build(*_, **__):
            raise FileNotFoundError("no loadable cached snapshot")

        monkeypatch.setattr(validation, "_get_kernel_from_cache", no_cached_build)
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

    @pytest.mark.parametrize("cached", [True, False])
    def test_hub_network_failure_falls_back_to_cache(
        self, min_base_cfg, monkeypatch, cached
    ):
        import kernels
        import transformers.integrations.hub_kernels as hub_kernels

        import axolotl.utils.schemas.validation as validation

        self._force_availability(monkeypatch, False)
        monkeypatch.setattr(hub_kernels, "get_attn_kernel_version", lambda _: 1)

        def online(*_, **__):
            raise ConnectionError("Connection reset by peer")

        def from_cache(repo_id, version):
            assert version == 1
            if not cached:
                raise FileNotFoundError("Cannot find a local snapshot")
            return object()

        monkeypatch.setattr(kernels, "get_kernel", online)
        monkeypatch.setattr(validation, "_get_kernel_from_cache", from_cache)
        cfg = min_base_cfg | DictDefault(attn_implementation="flash_attention_2")
        if cached:
            validated = validate_config(cfg)
            assert validated.attn_implementation == "flash_attention_2"
        else:
            with pytest.raises(ValueError, match="Connection reset by peer"):
                validate_config(cfg)

    @pytest.mark.parametrize("cached", [True, False])
    def test_version_lookup_failure_falls_back_to_cache(
        self, min_base_cfg, monkeypatch, cached
    ):
        """Resolving the pinned version lists hub refs, so it fails offline as well."""
        import kernels
        import transformers.integrations.hub_kernels as hub_kernels

        import axolotl.utils.schemas.validation as validation

        self._force_availability(monkeypatch, False)

        def offline(_):
            raise ConnectionError("Name or service not known")

        def never(*_, **__):
            raise AssertionError("get_kernel must not run without a version")

        def from_cache(repo_id, version):
            assert version is None
            if not cached:
                raise FileNotFoundError("Cannot find a local snapshot")
            return object()

        monkeypatch.setattr(hub_kernels, "get_attn_kernel_version", offline)
        monkeypatch.setattr(kernels, "get_kernel", never)
        monkeypatch.setattr(validation, "_get_kernel_from_cache", from_cache)
        cfg = min_base_cfg | DictDefault(attn_implementation="flash_attention_2")
        if cached:
            assert validate_config(cfg).attn_implementation == "flash_attention_2"
        else:
            with pytest.raises(ValueError, match="Name or service not known"):
                validate_config(cfg)

    def test_cache_lookup_prefers_version_ref_then_newest_build(
        self, monkeypatch, tmp_path
    ):
        import kernels
        from huggingface_hub import constants

        import axolotl.utils.schemas.validation as validation

        repo = tmp_path / "kernels--kernels-community--flash-attn2"
        for sha in ("aaa", "bbb"):
            (repo / "snapshots" / sha / "build").mkdir(parents=True)
        (repo / "snapshots" / "ccc").mkdir()  # no build dir: never a candidate
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("KERNELS_CACHE", raising=False)
        loaded = []
        monkeypatch.setattr(
            kernels,
            "get_local_kernel",
            lambda path, **_: loaded.append(path.name) or object(),
        )

        validation._get_kernel_from_cache("kernels-community/flash-attn2", 3)
        assert loaded[-1] in {"aaa", "bbb"}

        (repo / "refs").mkdir()
        (repo / "refs" / "v3").write_text("bbb")
        validation._get_kernel_from_cache("kernels-community/flash-attn2", 3)
        assert loaded[-1] == "bbb"

    def test_cache_lookup_without_snapshot_raises(self, monkeypatch, tmp_path):
        from huggingface_hub import constants

        import axolotl.utils.schemas.validation as validation

        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.delenv("KERNELS_CACHE", raising=False)
        with pytest.raises(FileNotFoundError, match="no loadable cached snapshot"):
            validation._get_kernel_from_cache("kernels-community/flash-attn2", 3)
