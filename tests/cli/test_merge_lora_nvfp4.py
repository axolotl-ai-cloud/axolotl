# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Routing test for the NVFP4 MoE branch of the LoRA merge CLI (step 4).

Pins two things on a triton-less / CUDA-less host:
  * ``is_nvfp4_moe_checkpoint`` is True for a synthetic dsv4 NVFP4 base dir and False for a
    plain dir (no index / non-NVFP4 config);
  * ``_do_merge_lora_efficient`` routes to ``write_merged_nvfp4_checkpoint`` when detection is
    True and to ``merge_lora_sharded_efficient`` when detection is False (no regression).

``merge_lora.py`` is loaded BY FILE PATH with ``axolotl.cli.config`` stubbed (its real import
chain pulls in numba, which is incompatible with the test host's numpy) so the genuine routing
source under test runs unchanged. The writer + its two sibling deps are likewise loaded by file
path (the scattermoe_lora package ``__init__`` imports triton) and injected under the writer's
real module name so the CLI's lazy ``from ...nvfp4_lora_merge_writer import ...`` resolves to
them. Gated on torchao + safetensors, NOT CUDA."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import pytest

pytest.importorskip("torchao", reason="torchao required for NVFP4 requant")
pytest.importorskip("safetensors", reason="safetensors required for checkpoint IO")


def _load_by_path(mod_name: str, path: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scattermoe_lib(filename: str) -> str:
    import axolotl

    return os.path.join(
        os.path.dirname(axolotl.__file__),
        "integrations",
        "kernels",
        "libs",
        "scattermoe_lora",
        filename,
    )


# Reuse the writer test's synthetic-checkpoint helpers (which load the writer + siblings by file
# path and inject them) so this test never re-derives the NVFP4 synthesis.
_WRITER_TEST = _load_by_path(
    "_merge_lora_nvfp4_writer_test",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "integrations",
        "kernels",
        "scattermoe_lora",
        "test_nvfp4_lora_merge_writer.py",
    ),
)
WRITER_MOD = _WRITER_TEST.writer_mod


def _merge_lora_path() -> str:
    import axolotl

    return os.path.join(os.path.dirname(axolotl.__file__), "cli", "merge_lora.py")


def _load_merge_lora_module():
    """Load ``cli/merge_lora.py`` by file path with its numba-poisoned config import stubbed and
    the file-path-loaded writer injected under its real module name (so the CLI's lazy import of
    the writer resolves without executing the triton-importing scattermoe_lora package init).

    The package stub + writer injection persist for the whole test run: the CLI imports the
    writer LAZILY inside the routing helper, so they must still be in ``sys.modules`` when a test
    calls ``_do_merge_lora_efficient``. Only the config stub is temporary."""
    pkg = "axolotl.integrations.kernels.libs.scattermoe_lora"
    writer_name = pkg + ".nvfp4_lora_merge_writer"

    if not isinstance(sys.modules.get(pkg), types.ModuleType) or not hasattr(
        sys.modules.get(pkg), "__path__"
    ):
        pkg_stub = types.ModuleType(pkg)
        pkg_stub.__path__ = [os.path.dirname(_scattermoe_lib("__init__.py"))]
        sys.modules[pkg] = pkg_stub
    sys.modules[writer_name] = WRITER_MOD

    saved_cfg = sys.modules.get("axolotl.cli.config")
    cfg_stub = types.ModuleType("axolotl.cli.config")
    cfg_stub.load_cfg = lambda *a, **k: None  # unused by the routing path under test
    sys.modules["axolotl.cli.config"] = cfg_stub
    try:
        mod = _load_by_path("axolotl.cli.merge_lora", _merge_lora_path())
    finally:
        if saved_cfg is None:
            sys.modules.pop("axolotl.cli.config", None)
        else:
            sys.modules["axolotl.cli.config"] = saved_cfg
    return mod


MERGE_LORA = _load_merge_lora_module()


def _plain_dir(path: str) -> str:
    """A non-NVFP4 base dir: no index, a config.json with a non-NVFP4 quant algo."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": "llama"}, f)
    with open(os.path.join(path, "hf_quant_config.json"), "w") as f:
        json.dump({"quantization": {"quant_algo": "FP8"}}, f)
    return path


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_is_nvfp4_moe_checkpoint_true_for_dsv4(tmp_path):
    base = str(tmp_path / "base")
    _WRITER_TEST._make_base_checkpoint(base)
    assert WRITER_MOD.is_nvfp4_moe_checkpoint(base) is True


def test_is_nvfp4_moe_checkpoint_false_for_plain_dir(tmp_path):
    assert (
        WRITER_MOD.is_nvfp4_moe_checkpoint(_plain_dir(str(tmp_path / "plain"))) is False
    )


def test_is_nvfp4_moe_checkpoint_false_for_missing_dir(tmp_path):
    assert WRITER_MOD.is_nvfp4_moe_checkpoint(str(tmp_path / "does_not_exist")) is False


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


class _Cfg(dict):
    """Minimal attribute-access config stand-in for the routing path."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None


def test_efficient_merge_routes_to_nvfp4_writer(tmp_path, monkeypatch):
    base = str(tmp_path / "base")
    _WRITER_TEST._make_base_checkpoint(base)
    adapter = str(tmp_path / "adapter")
    _WRITER_TEST._make_adapter(adapter, r=4, alpha=8.0)

    calls = {"writer": [], "sharded": []}
    monkeypatch.setattr(
        WRITER_MOD,
        "write_merged_nvfp4_checkpoint",
        lambda **kw: calls["writer"].append(kw),
    )
    monkeypatch.setattr(
        MERGE_LORA,
        "merge_lora_sharded_efficient",
        lambda *a, **kw: calls["sharded"].append(kw),
    )

    cfg = _Cfg(
        base_model=base, lora_model_dir=adapter, output_dir=str(tmp_path / "out")
    )
    MERGE_LORA._do_merge_lora_efficient(cfg=cfg)

    assert len(calls["writer"]) == 1
    assert calls["sharded"] == []
    assert calls["writer"][0]["base_repo"] == base
    assert calls["writer"][0]["adapter_dir"] == adapter
    assert calls["writer"][0]["output_dir"] == str(tmp_path / "out" / "merged")


def test_efficient_merge_routes_to_sharded_for_plain_base(tmp_path, monkeypatch):
    base = _plain_dir(str(tmp_path / "plain"))
    adapter = str(tmp_path / "adapter")
    os.makedirs(adapter, exist_ok=True)

    calls = {"writer": [], "sharded": []}
    monkeypatch.setattr(
        WRITER_MOD,
        "write_merged_nvfp4_checkpoint",
        lambda **kw: calls["writer"].append(kw),
    )
    monkeypatch.setattr(
        MERGE_LORA,
        "merge_lora_sharded_efficient",
        lambda *a, **kw: calls["sharded"].append(kw),
    )

    cfg = _Cfg(
        base_model=base, lora_model_dir=adapter, output_dir=str(tmp_path / "out")
    )
    MERGE_LORA._do_merge_lora_efficient(cfg=cfg)

    assert calls["writer"] == []
    assert len(calls["sharded"]) == 1


# ---------------------------------------------------------------------------
# no-regression on the REAL production import path (no package/writer stub)
# ---------------------------------------------------------------------------


def _load_merge_lora_module_no_stub(monkeypatch):
    """Load a fresh ``cli/merge_lora.py`` with ONLY the numba-poisoned config import stubbed.

    Unlike ``_load_merge_lora_module`` this injects NO scattermoe_lora package/writer stub, so the
    CLI's lazy ``from ...nvfp4_lora_merge_writer import ...`` hits the REAL package __init__ (which
    imports triton). This exercises the production import path that the stub-injecting loader hides,
    pinning that a missing writer degrades to the standard sharded path instead of crashing."""
    pkg = "axolotl.integrations.kernels.libs.scattermoe_lora"
    writer_name = pkg + ".nvfp4_lora_merge_writer"
    # Drop any package/writer stub a sibling test left in sys.modules so the real import is forced.
    monkeypatch.delitem(sys.modules, pkg, raising=False)
    monkeypatch.delitem(sys.modules, writer_name, raising=False)

    cfg_stub = types.ModuleType("axolotl.cli.config")
    cfg_stub.load_cfg = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "axolotl.cli.config", cfg_stub)
    return _load_by_path("axolotl.cli.merge_lora_real", _merge_lora_path())


def test_plain_merge_falls_through_when_writer_import_unavailable(
    tmp_path, monkeypatch
):
    """On a triton-less host the writer package import fails; a non-NVFP4 base must still route to
    the standard sharded path (no regression) rather than crash on ``import triton``."""
    pytest.importorskip("torch")
    if importlib.util.find_spec("triton") is not None:
        pytest.skip(
            "triton present; this regression only manifests on triton-less hosts"
        )

    mod = _load_merge_lora_module_no_stub(monkeypatch)

    calls = {"sharded": []}
    monkeypatch.setattr(
        mod,
        "merge_lora_sharded_efficient",
        lambda *a, **kw: calls["sharded"].append(kw),
    )

    base = _plain_dir(str(tmp_path / "plain"))
    adapter = str(tmp_path / "adapter")
    os.makedirs(adapter, exist_ok=True)
    cfg = _Cfg(
        base_model=base, lora_model_dir=adapter, output_dir=str(tmp_path / "out")
    )

    mod._do_merge_lora_efficient(cfg=cfg)

    assert len(calls["sharded"]) == 1


def test_try_merge_nvfp4_moe_returns_false_when_writer_import_unavailable(monkeypatch):
    """Pin the guard directly: when the writer package import fails (triton absent),
    ``_try_merge_nvfp4_moe`` returns False instead of propagating ImportError."""
    if importlib.util.find_spec("triton") is not None:
        pytest.skip(
            "triton present; this regression only manifests on triton-less hosts"
        )

    mod = _load_merge_lora_module_no_stub(monkeypatch)
    cfg = _Cfg(base_model="/nonexistent-not-nvfp4")
    assert mod._try_merge_nvfp4_moe(cfg=cfg) is False
