"""Tests for FA4 backend selection: quack gate, warning suppression, native upgrade."""

import warnings
from unittest.mock import MagicMock

import pytest

import axolotl.monkeypatch.attention.flash_attn_4 as fa4
from axolotl.loaders.model import ModelLoader


class TestQuackVersionGate:
    @pytest.mark.parametrize(
        "installed, expected_ok",
        [
            ("0.5.0", False),
            ("0.5.3", False),
            ("0.6.0", True),
            ("0.6.1", True),
            ("1.0.0", True),
        ],
    )
    def test_version_boundary(self, monkeypatch, installed, expected_ok):
        monkeypatch.setattr("importlib.metadata.version", lambda _pkg: installed)
        ok, ver = fa4._quack_supported()
        assert ok is expected_ok
        assert ver == installed

    def test_absent_quack_is_treated_ok(self, monkeypatch):
        from importlib.metadata import PackageNotFoundError

        def _raise(_pkg):
            raise PackageNotFoundError(_pkg)

        monkeypatch.setattr("importlib.metadata.version", _raise)
        assert fa4._quack_supported() == (True, None)


class TestConfigureFa4:
    def test_suppresses_aux_data_warning(self, monkeypatch):
        monkeypatch.setattr(fa4, "_quack_supported", lambda: (True, None))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            fa4.configure_fa4()
            warnings.warn(
                "Argument aux_data (position 17) cannot be converted to a JitArgument",
                UserWarning,
                stacklevel=2,
            )
            warnings.warn("unrelated warning", UserWarning, stacklevel=2)
        messages = [str(w.message) for w in rec]
        assert not any("aux_data" in m for m in messages)
        assert any("unrelated" in m for m in messages)

    def test_warns_on_stale_quack(self, monkeypatch):
        monkeypatch.setattr(fa4, "_quack_supported", lambda: (False, "0.5.3"))
        mock_log = MagicMock()
        monkeypatch.setattr(fa4, "LOG", mock_log)
        with warnings.catch_warnings():
            fa4.configure_fa4()
        mock_log.warning.assert_called_once()
        assert "0.5.3" in str(mock_log.warning.call_args)


class TestResolveFlashAttention4:
    @staticmethod
    def _loader():
        loader = ModelLoader.__new__(ModelLoader)
        loader.model_config = object()
        return loader

    @pytest.mark.parametrize("impl", ["sdpa", "eager", "xformers"])
    def test_non_flash_passthrough(self, monkeypatch, impl):
        fa4_usable = MagicMock()
        monkeypatch.setattr(fa4, "fa4_usable", fa4_usable)
        assert self._loader()._resolve_flash_attention_4(impl) == impl
        fa4_usable.assert_not_called()

    def test_explicit_fa4_kept_without_capability_check(self, monkeypatch):
        monkeypatch.setattr(
            fa4, "fa4_usable", MagicMock(side_effect=AssertionError("must not check"))
        )
        configure = MagicMock()
        monkeypatch.setattr(fa4, "configure_fa4", configure)
        assert (
            self._loader()._resolve_flash_attention_4("flash_attention_4")
            == "flash_attention_4"
        )
        configure.assert_called_once()

    def test_fa2_upgraded_when_usable(self, monkeypatch):
        monkeypatch.setattr(fa4, "fa4_usable", lambda _mc: True)
        configure = MagicMock()
        monkeypatch.setattr(fa4, "configure_fa4", configure)
        assert (
            self._loader()._resolve_flash_attention_4("flash_attention_2")
            == "flash_attention_4"
        )
        configure.assert_called_once()

    def test_fa2_kept_when_not_usable(self, monkeypatch):
        monkeypatch.setattr(fa4, "fa4_usable", lambda _mc: False)
        configure = MagicMock()
        monkeypatch.setattr(fa4, "configure_fa4", configure)
        assert (
            self._loader()._resolve_flash_attention_4("flash_attention_2")
            == "flash_attention_2"
        )
        configure.assert_not_called()
