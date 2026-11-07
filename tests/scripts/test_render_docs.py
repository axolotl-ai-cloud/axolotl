"""Checks for combining independent Quarto website renders."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "render_docs", Path(__file__).resolve().parents[2] / "docs/scripts/render_docs.py"
)
assert SPEC is not None and SPEC.loader is not None
render_docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(render_docs)


def make_site(path, page, text):
    path.mkdir()
    (path / page).write_text(text)
    (path / "search.json").write_text(json.dumps([{"href": page, "text": text}]))
    return path


def test_merge_preserves_home_page_and_all_search_entries(tmp_path):
    first = make_site(tmp_path / "first", "index.html", "Home")
    second = make_site(tmp_path / "second", "api.html", "API")
    (second / "index.html").write_text("Redirect to API")
    for site in (first, second):
        (site / "shared.css").write_text("body {}")

    output = tmp_path / "merged"
    render_docs.merge_sites([first, second], output, home_worker=0)

    assert (output / "index.html").read_text() == "Home"
    assert (output / "api.html").read_text() == "API"
    assert (output / "shared.css").read_text() == "body {}"
    assert {
        entry["href"] for entry in json.loads((output / "search.json").read_text())
    } == {
        "index.html",
        "api.html",
    }


def test_merge_rejects_conflicting_shared_assets(tmp_path):
    first = make_site(tmp_path / "first", "index.html", "Home")
    second = make_site(tmp_path / "second", "api.html", "API")
    (first / "shared.css").write_text("first")
    (second / "shared.css").write_text("second")

    with pytest.raises(ValueError, match="Conflicting output: shared.css"):
        render_docs.merge_sites([first, second], tmp_path / "merged", home_worker=0)


def test_merge_rejects_conflicting_search_entries(tmp_path):
    first = make_site(tmp_path / "first", "index.html", "Home")
    second = make_site(tmp_path / "second", "index.html", "Different home")

    with pytest.raises(ValueError, match="Conflicting search entry: index.html"):
        render_docs.merge_sites([first, second], tmp_path / "merged", home_worker=0)


def test_partition_covers_markdown_notebooks_and_quarto_once():
    inputs = [
        "index.qmd",
        "FAQS.md",
        "docs/api/a.qmd",
        "docs/api/b.qmd",
        "docs/guide.qmd",
        "docs/models/a.qmd",
        "examples/demo.ipynb",
    ]
    assignments = render_docs.partition_inputs(inputs, 4)
    targets = [target for worker in assignments for target in worker]
    for name in inputs:
        assert (
            sum(name == target or name.startswith(target + "/") for target in targets)
            == 1
        )
    assert "docs/api" in targets
    assert "index.qmd" in targets
