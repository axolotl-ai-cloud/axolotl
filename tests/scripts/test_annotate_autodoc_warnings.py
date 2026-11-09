"""Checks for reporting quartodoc docstring warnings as GitHub annotations."""

import importlib.util
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "annotate_autodoc_warnings",
    Path(__file__).resolve().parents[2] / "docs/scripts/annotate_autodoc_warnings.py",
)
assert SPEC is not None and SPEC.loader is not None
annotate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotate)

LOG = """\
src/axolotl/cli/main.py:183: No type or annotation for parameter 'kwargs'
some unrelated quartodoc output
src/axolotl/utils/collators/dpo.py:29: Parameter 'pad_token_id' does not appear in the function signature
"""


def test_parses_only_griffe_warning_lines():
    assert annotate.parse_warnings(LOG) == [
        (
            "src/axolotl/cli/main.py",
            183,
            "No type or annotation for parameter 'kwargs'",
        ),
        (
            "src/axolotl/utils/collators/dpo.py",
            29,
            "Parameter 'pad_token_id' does not appear in the function signature",
        ),
    ]


def test_workflow_commands_escape_and_cap():
    warnings = [("src/a,b.py", line, "100% | broken") for line in range(12)]
    commands = annotate.workflow_commands(warnings)

    assert len(commands) == annotate.MAX_ANNOTATIONS + 1
    assert commands[0] == (
        "::warning file=src/a%2Cb.py,line=0,title=Docstring::100%25 | broken"
    )
    assert commands[-1] == (
        "::notice title=Docstring::2 more docstring warning(s) in the job summary"
    )


def test_main_writes_summary_and_succeeds(tmp_path, monkeypatch, capsys):
    log = tmp_path / "quartodoc-build.log"
    log.write_text(LOG)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(sys, "argv", ["annotate", str(log)])

    annotate.main()

    out = capsys.readouterr().out
    assert out.count("::warning file=") == 2
    assert "2 docstring warning(s)" in out
    text = summary.read_text()
    assert "| `src/axolotl/cli/main.py:183` |" in text
    assert "autodoc-log" in text


def test_summary_without_warnings():
    assert "No docstring warnings" in annotate.summary([])
