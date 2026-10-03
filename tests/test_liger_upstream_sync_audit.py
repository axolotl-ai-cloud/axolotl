"""Unit tests for the liger-upstream-sync skill's audit tool.

Covers the static parser (`parse_plugin`), the pure diff logic (`classify`), the
hardening guards (decoy dispatch `if`, non-literal override set), and the
exit-code contract (0 clean / 1 findings / 2 could-not-audit). The audit lives
under the skill directory, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents/skills/liger-upstream-sync/scripts/audit_liger_sync.py"
)
_spec = importlib.util.spec_from_file_location("liger_audit", _AUDIT_PATH)
assert _spec and _spec.loader
audit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = audit  # dataclass annotation resolution needs this
_spec.loader.exec_module(audit)


# Mirrors the real plugin's dispatch shape: an `if` head testing both the table
# and the override set, a nested rope-default tuple in the body (must be ignored),
# incremental override mutation, and a decoy `.parameters` probe.
PLUGIN_SRC = """
def pre_model_load(cfg):
    axolotl_override_liger_fn = {"qwen3_5"}
    axolotl_override_liger_fn |= {"qwen3_5_moe"}
    axolotl_override_liger_fn.add("extra_override")

    if (
        cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN
        and cfg.model_config_type not in axolotl_override_liger_fn
    ):
        if rope_value is None and cfg.model_config_type in ("qwen2_vl", "qwen3_vl"):
            rope_value = True
        if "rope" in liger_fn_sig.parameters:
            kwargs["rope"] = rope_value
        if "swiglu" in liger_fn_sig.parameters:
            kwargs["swiglu"] = cfg.liger_glu_activation
        if "foo" in other_obj.parameters:
            pass
    elif cfg.model_config_type == "qwen3_5":
        pass
    elif cfg.model_config_type == "llama4":
        pass
    elif cfg.model_config_type in ("gemma4_unified", "gemma4_unified_text"):
        pass
    elif cfg.model_config_type == "jamba":
        pass
"""


def _write(tmp_path: Path, src: str) -> Path:
    path = tmp_path / "plugin.py"
    path.write_text(src, encoding="utf-8")
    return path


def _fn() -> None: ...


def _fn_glu(glu: bool = False) -> None: ...


# --- parse_plugin -----------------------------------------------------------


def test_parse_extracts_override_elif_and_relied(tmp_path: Path) -> None:
    facts = audit.parse_plugin(_write(tmp_path, PLUGIN_SRC))
    # `=`, `|=`, and `.add()` are all captured
    assert facts.override_set == {"qwen3_5", "qwen3_5_moe", "extra_override"}
    # only dispatch-ladder elif tests; the rope-default tuple in the body is NOT included
    assert facts.elif_types == {
        "qwen3_5",
        "llama4",
        "gemma4_unified",
        "gemma4_unified_text",
        "jamba",
    }
    assert "qwen2_vl" not in facts.elif_types
    # only `liger_fn_sig.parameters` probes; the decoy `other_obj.parameters` is excluded
    assert facts.relied_params == {"rope", "swiglu"}
    assert not facts.warnings


def test_parse_ignores_decoy_if_before_head(tmp_path: Path) -> None:
    src = (
        "def f(cfg):\n"
        '    axolotl_override_liger_fn = {"qwen3_5"}\n'
        '    if "warmup" in MODEL_TYPE_TO_APPLY_LIGER_FN:\n'
        "        pass\n"
        "    if (cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN"
        " and cfg.model_config_type not in axolotl_override_liger_fn):\n"
        "        pass\n"
        '    elif cfg.model_config_type == "llama4":\n'
        "        pass\n"
    )
    facts = audit.parse_plugin(_write(tmp_path, src))
    assert facts.elif_types == {"llama4"}
    assert not facts.warnings


def test_parse_warns_on_non_literal_override(tmp_path: Path) -> None:
    src = (
        "def f(cfg):\n"
        "    axolotl_override_liger_fn = _build_overrides()\n"
        "    if (cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN"
        " and cfg.model_config_type not in axolotl_override_liger_fn):\n"
        "        pass\n"
        '    elif cfg.model_config_type == "llama4":\n'
        "        pass\n"
    )
    facts = audit.parse_plugin(_write(tmp_path, src))
    assert facts.override_set == set()
    assert facts.warnings  # reliability warning raised


def test_parse_warns_on_partial_literal_override(tmp_path: Path) -> None:
    # a set mixing a literal and a non-literal element must still warn (don't trust
    # the partial read), per the broadened unresolved check
    src = (
        "def f(cfg):\n"
        '    axolotl_override_liger_fn = {"qwen3_5", SOME_CONST}\n'
        "    if (cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN"
        " and cfg.model_config_type not in axolotl_override_liger_fn):\n"
        "        pass\n"
        '    elif cfg.model_config_type == "qwen3_5":\n'
        "        pass\n"
    )
    facts = audit.parse_plugin(_write(tmp_path, src))
    assert "qwen3_5" in facts.override_set  # the literal part is still captured
    assert facts.warnings  # but the non-literal element raises a warning


# --- classify ---------------------------------------------------------------


def test_classify_buckets() -> None:
    facts = audit.PluginFacts(
        override_set={"qwen3_5"},
        elif_types={"qwen3_5", "llama4", "jamba"},
        relied_params={"swiglu"},
    )
    upstream = {"qwen3_5": _fn, "llama4": _fn, "other": _fn_glu}
    result = audit.classify(facts, upstream)

    assert result.shadowed == ["llama4"]  # in table, hand-patched, not overridden
    assert result.override_ok == ["qwen3_5"]  # native + routed
    assert result.override_stale == [] and result.override_unrouted == []
    assert result.custom == ["jamba"]  # not in table
    # "other" exposes a `glu` param the plugin (which checks `swiglu`) no longer forwards
    assert [t for t, _ in result.drift] == ["other"]
    assert result.review_count == 2  # shadowed + drift; ok/custom don't count


def test_classify_detects_bump_and_override_suppresses_it() -> None:
    elif_types = {"qwen3_5", "qwen3_5_moe", "llama4"}
    bumped = {"qwen3_5": _fn, "qwen3_5_moe": _fn, "llama4": _fn, "llama": _fn}

    # bump applied but override set NOT updated -> the new natives are flagged
    forgot = audit.classify(audit.PluginFacts(elif_types=elif_types), bumped)
    assert "qwen3_5" in forgot.shadowed and "qwen3_5_moe" in forgot.shadowed

    # with the override set updated, the bumped types are suppressed; llama4 remains
    fixed = audit.classify(
        audit.PluginFacts(
            override_set={"qwen3_5", "qwen3_5_moe"}, elif_types=elif_types
        ),
        bumped,
    )
    assert "qwen3_5" not in fixed.shadowed
    assert fixed.shadowed == ["llama4"]


def test_classify_override_health() -> None:
    facts = audit.PluginFacts(
        override_set={"stale_type", "unrouted_type"},
        elif_types={"jamba"},
    )
    # unrouted_type is native but has no elif; stale_type is not native at all
    upstream = {"unrouted_type": _fn}
    result = audit.classify(facts, upstream)
    assert result.override_stale == ["stale_type"]
    assert result.override_unrouted == ["unrouted_type"]
    assert result.review_count == 2


# --- exit-code contract (via main) ------------------------------------------


def test_exit_2_when_plugin_missing(tmp_path: Path) -> None:
    assert audit.main(["--plugin", str(tmp_path / "nope.py")]) == 2


def test_exit_2_when_plugin_unparseable(tmp_path: Path) -> None:
    broken = _write(tmp_path, "def f(:\n    pass\n")
    assert audit.main(["--plugin", str(broken)]) == 2


def test_exit_1_on_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "load_upstream", lambda: ({"llama4": _fn}, "test"))
    # llama4 is hand-patched and not overridden -> shadowed -> findings -> exit 1
    assert audit.main(["--plugin", _write(tmp_path, PLUGIN_SRC).__fspath__()]) == 1


def test_exit_0_when_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = (
        "def f(cfg):\n"
        '    axolotl_override_liger_fn = {"qwen3_5"}\n'
        "    if (cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN"
        " and cfg.model_config_type not in axolotl_override_liger_fn):\n"
        "        pass\n"
        '    elif cfg.model_config_type == "qwen3_5":\n'
        "        pass\n"
        '    elif cfg.model_config_type == "jamba":\n'
        "        pass\n"
    )
    # qwen3_5 native + overridden + routed (ok); jamba custom; no drift -> exit 0
    monkeypatch.setattr(audit, "load_upstream", lambda: ({"qwen3_5": _fn}, "test"))
    assert audit.main(["--plugin", _write(tmp_path, src).__fspath__()]) == 0


# --- source mode (--liger-source) -------------------------------------------

_MONKEY_PATCH_SRC = (
    "def _f(**kwargs):\n    pass\n\n"
    'MODEL_TYPE_TO_APPLY_LIGER_FN = {"llama": _f, "qwen3": _f, "qwen3_5": _f}\n'
)


def test_dispatch_table_keys_and_load_source(tmp_path: Path) -> None:
    src = tmp_path / "monkey_patch.py"
    src.write_text(_MONKEY_PATCH_SRC, encoding="utf-8")
    assert audit.dispatch_table_keys(_MONKEY_PATCH_SRC) == {"llama", "qwen3", "qwen3_5"}
    table, label = audit.load_source(src)
    assert table is not None and set(table) == {"llama", "qwen3", "qwen3_5"}
    assert all(v is None for v in table.values())  # keys-only, no fn objects
    assert "keys-only" in label


def test_dispatch_table_keys_bails_on_non_literal_keys() -> None:
    # all-literal table is read in full
    assert audit.dispatch_table_keys(
        'MODEL_TYPE_TO_APPLY_LIGER_FN = {"llama": 1, "qwen3": 1}\n'
    ) == {"llama", "qwen3"}
    # dict unpacking (`{**base, ...}`) -> key is None -> unreadable (empty), not partial
    assert (
        audit.dispatch_table_keys(
            'BASE = {"a": 1}\nMODEL_TYPE_TO_APPLY_LIGER_FN = {**BASE, "llama": 1}\n'
        )
        == set()
    )
    # a computed/non-literal key -> unreadable
    assert (
        audit.dispatch_table_keys('MODEL_TYPE_TO_APPLY_LIGER_FN = {"llama": 1, K: 2}\n')
        == set()
    )


def test_load_source_resolves_package_dir(tmp_path: Path) -> None:
    pkg = tmp_path / "liger_kernel" / "transformers"
    pkg.mkdir(parents=True)
    (pkg / "monkey_patch.py").write_text(_MONKEY_PATCH_SRC, encoding="utf-8")
    (tmp_path / "liger_kernel" / "__init__.py").write_text('__version__ = "9.9.9"\n')
    table, label = audit.load_source(tmp_path)  # a directory, not a file
    assert table is not None and "qwen3_5" in table
    assert "9.9.9" in label


def test_classify_skips_drift_when_disabled() -> None:
    facts = audit.PluginFacts(elif_types={"jamba"}, relied_params={"swiglu"})
    upstream = {"other": _fn_glu}  # would normally flag glu drift
    assert audit.classify(facts, upstream, check_drift=False).drift == []
    assert audit.classify(facts, upstream, check_drift=True).drift  # control


def test_main_liger_source_skips_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "monkey_patch.py"
    src.write_text(
        'MODEL_TYPE_TO_APPLY_LIGER_FN = {"qwen3_5": None}\n', encoding="utf-8"
    )
    plugin = (
        "def f(cfg):\n"
        '    axolotl_override_liger_fn = {"qwen3_5"}\n'
        "    if (cfg.model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN"
        " and cfg.model_config_type not in axolotl_override_liger_fn):\n"
        '        if "swiglu" in liger_fn_sig.parameters:\n'
        "            pass\n"
        '    elif cfg.model_config_type == "qwen3_5":\n'
        "        pass\n"
        '    elif cfg.model_config_type == "jamba":\n'
        "        pass\n"
    )
    # qwen3_5 overridden+routed (ok), jamba custom, no shadow, drift skipped -> exit 0
    code = audit.main(
        ["--plugin", _write(tmp_path, plugin).__fspath__(), "--liger-source", str(src)]
    )
    assert code == 0
    assert "skipped" in capsys.readouterr().out  # [4] reports keys-only skip


def test_load_source_returns_none_on_unreadable_table(tmp_path: Path) -> None:
    src = tmp_path / "monkey_patch.py"
    src.write_text("X = 1\n", encoding="utf-8")  # no dispatch-table dict literal
    table, _ = audit.load_source(src)
    assert table is None  # unreadable -> not a misleading 0-type table
    # main() treats it as could-not-audit, not as "0 findings"
    code = audit.main(
        [
            "--plugin",
            _write(tmp_path, PLUGIN_SRC).__fspath__(),
            "--liger-source",
            str(src),
        ]
    )
    assert code == 2


def test_main_liger_source_missing_returns_2(tmp_path: Path) -> None:
    code = audit.main(
        [
            "--plugin",
            _write(tmp_path, PLUGIN_SRC).__fspath__(),
            "--liger-source",
            str(tmp_path / "absent"),
        ]
    )
    assert code == 2
