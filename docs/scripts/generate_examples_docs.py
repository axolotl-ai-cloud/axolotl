"""
auto generate example docs from allowlist
"""

import re
import shutil
import sys
from pathlib import Path

import yaml

# Paths
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]  # repo root (docs/scripts -> docs -> ROOT)
EXAMPLES_DIR = ROOT / "examples"
OUTPUT_DIR = ROOT / "docs" / "examples"
ALLOWLIST_YML = THIS.parent / "examples-allowlist.yml"


# utilities
def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-_/]+", "-", name.strip())
    s = s.replace("/", "-")
    s = re.sub(r"-+", "-", s).strip("-").lower()
    return s or "example"


def read_allowlist():
    with open(ALLOWLIST_YML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("examples", [])
    if not isinstance(items, list):
        raise ValueError("`examples` must be a list in examples-allowlist.yml")
    return items


def find_readme(folder: Path) -> Path | None:
    for name in ("README.md", "Readme.md", "readme.md"):
        p = folder / name
        if p.exists():
            return p
    return None


def first_h1(md: str) -> str | None:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def rewrite_and_copy_assets(
    md: str, src_dir: Path, dest_assets_root: Path, slug: str
) -> str:
    """
    Copy local image assets referenced in markdown to
    docs/examples/assets/<slug>/... and rewrite the links.
    """
    dest_assets = dest_assets_root / slug

    def repl(m):
        url = m.group(1).strip()
        if re.match(r"^(https?:)?//", url):
            return m.group(0)  # leave remote URLs
        # normalize path
        src_path = (src_dir / url).resolve()
        if not src_path.exists():
            return m.group(0)  # leave as-is if not found
        rel = src_path.relative_to(src_dir)
        dest_path = dest_assets / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        new_rel = f"assets/{slug}/{rel.as_posix()}"
        return m.group(0).replace(url, new_rel)

    return IMG_RE.sub(repl, md)


def write_qmd(out_path: Path, title: str, body_md: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fm = f"---\ntitle: {title!r}\nformat:\n  html:\n    toc: true\n---\n\n"
    out_path.write_text(fm + body_md, encoding="utf-8")


def main():
    allow = read_allowlist()
    if not EXAMPLES_DIR.exists():
        print(f"[WARN] {EXAMPLES_DIR} not found", file=sys.stderr)
        return

    (OUTPUT_DIR / "assets").mkdir(parents=True, exist_ok=True)

    generated = []
    for item in allow:
        src_dir = EXAMPLES_DIR / item
        if not src_dir.exists() or not src_dir.is_dir():
            print(f"[WARN] Skipping {item} (not a directory)", file=sys.stderr)
            continue

        readme = find_readme(src_dir)
        if not readme:
            print(f"[WARN] Skipping {item} (no README.md)", file=sys.stderr)
            continue

        md = readme.read_text(encoding="utf-8")
        slug = slugify(item)
        title = first_h1(md) or f"Example: {item}"
        md = rewrite_and_copy_assets(md, src_dir, OUTPUT_DIR / "assets", slug)
        write_qmd(OUTPUT_DIR / f"{slug}.qmd", title, md)
        generated.append(slug)

    # Optional: index page
    if generated:
        generated.sort()
        listing = "\n".join([f"- [{s}](./{s}.qmd)" for s in generated])
        index_md = "# Examples\n\nBelow are the curated examples:\n\n" + listing + "\n"
        write_qmd(OUTPUT_DIR / "index.qmd", "Examples", index_md)


if __name__ == "__main__":
    main()
