"""Custom quartodoc renderer for the Axolotl API reference."""

from typing import Optional

from plum import dispatch
from quartodoc import MdRenderer, layout
from quartodoc._griffe_compat import docstrings as ds


class Renderer(MdRenderer):
    """Renderer that keeps whole attribute docstrings in the summary table.

    Attributes never get an expanded section of their own, so the summary table is
    the only place their docstring appears; the default renderer truncates it to the
    first line.
    """

    style = "axolotl"

    @dispatch
    def summarize(
        self,
        el: layout.DocAttribute,
        path: Optional[str] = None,
        shorten: bool = False,
    ):
        if path is None:
            link = f"[{el.name}](#{el.anchor})"
        else:
            link = f"[{el.name}]({path}.qmd#{el.anchor})"

        doc = el.obj.docstring
        parts = doc.parsed if doc is not None else []
        if parts and isinstance(parts[0], ds.DocstringSectionText):
            description = parts[0].value
        else:
            description = ""

        return self._summary_row(link, description)
