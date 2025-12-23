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


def slugify(name: str) -> str:
    """Convert a name to a slug (lowercase, hyphens for spaces)."""
    s = re.sub(r"[^a-zA-Z0-9\s\-]+", "", name.strip())
    s = re.sub(r"\s+", "-", s).strip("-").lower()
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


def remove_first_h1(md: str) -> tuple[str, str | None]:
    """
    Remove the first H1 from markdown and return (modified_md, h1_title).
    The H1 is removed since we use the frontmatter title instead.
    """
    lines = md.splitlines()
    result = []
    h1_title = None
    skipped_first = False

    for line in lines:
        if not skipped_first and line.startswith("# "):
            h1_title = line[2:].strip()
            skipped_first = True
            continue
        result.append(line)

    return "\n".join(result), h1_title


IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def rewrite_and_copy_assets(md: str, src_dir: Path, dest_assets_root: Path) -> str:
    """
    Copy local image assets referenced in markdown to
    docs/examples/assets/... and rewrite the links.
    """
    dest_assets = dest_assets_root / "assets"

    def repl(m):
        url = m.group(1).strip()
        if re.match(r"^(https?:)?//", url):
            return m.group(0)  # leave remote URLs
        src_path = (src_dir / url).resolve()
        if not src_path.exists():
            return m.group(0)  # leave as-is if not found
        rel = src_path.relative_to(src_dir)
        # Create a unique asset path based on source directory name
        asset_name = src_dir.name.replace("/", "-")
        dest_path = dest_assets / asset_name / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        new_rel = f"assets/{asset_name}/{rel.as_posix()}"
        return m.group(0).replace(url, new_rel)

    return IMG_RE.sub(repl, md)


def rewrite_readme_links(
    md: str,
    src_dir: Path,
    examples_dir: Path,
    parent_index_only: set,
    current_src_path: str,
    allowlist_entries: set,
    current_output_path: str,
) -> str:
    """
    Rewrite links between README.md files to point to the correct .qmd files.
    """

    def repl(m):
        text = m.group(1)
        url = m.group(2).strip()

        # Skip remote URLs and anchor links
        if re.match(r"^(https?:)?//", url) or url.startswith("#"):
            return m.group(0)

        # Skip non-markdown files
        if not url.lower().endswith(".md"):
            return m.group(0)

        # Resolve the target path
        try:
            target_path = (src_dir / url).resolve()

            # Check if target is outside examples_dir
            try:
                rel_path = target_path.relative_to(examples_dir)
            except ValueError:
                # Target is outside examples_dir, leave as-is
                return m.group(0)

            parts = list(rel_path.parts)

            # Determine the output path for the target
            if len(parts) > 0 and parts[-1].lower() in ("readme.md", "readme"):
                # This is a README link
                if len(parts) == 1:
                    # Link to root README -> index.qmd
                    target_output = "index.qmd"
                elif len(parts) == 2:
                    if parts[0] == ".":
                        # Current directory README
                        target_output = "index.qmd"
                    else:
                        # subdir/README.md
                        parent_dir = parts[0]
                        if parent_dir in parent_index_only:
                            target_output = f"{parent_dir}/index.qmd"
                        else:
                            target_output = f"{parent_dir}.qmd"
                else:
                    # Deeper nesting: parent/subdir/README.md
                    # Build the full path like "parent/subdir"
                    full_path = "/".join(parts[:-1])  # Remove README.md
                    # Check if this exact path is in allowlist
                    if full_path in allowlist_entries:
                        # This is a sub-entry with its own entry -> use .qmd
                        target_output = f"{full_path}.qmd"
                    elif parts[0] == ".":
                        # ./subdir/README.md -> check if subdir has own entry
                        subdir = parts[1]
                        if subdir in parent_index_only:
                            target_output = f"{subdir}/index.qmd"
                        else:
                            target_output = f"{subdir}.qmd"
                    else:
                        # parent/subdir where parent doesn't have own entry
                        target_output = f"{full_path}/index.qmd"
            else:
                # Regular .md file -> convert to .qmd, keep path structure
                target_output = "/".join(parts)[:-2] + "qmd"

            # Compute relative path from current output file to target
            current_parts = current_output_path.split("/")
            target_parts = target_output.split("/")

            # Special case: if current is a subdir file and target is a single-component file at root
            # Example: current="magistral/vision", target="magistral.qmd"
            if len(current_parts) > 1 and len(target_parts) == 1:
                # Current is in subdir, target is at root level
                # Go up to root: ../ for each level
                up_count = len(current_parts) - 1
                rel_parts = [".."] * up_count + [target_parts[0]]
                new_url = "/".join(rel_parts)
            else:
                # Find common prefix
                i = 0
                while (
                    i < min(len(current_parts) - 1, len(target_parts))
                    and current_parts[i] == target_parts[i]
                ):
                    i += 1

                # Build relative path: go up (../) then down to target
                up_count = len(current_parts) - 1 - i
                rel_parts = [".."] * up_count + target_parts[i:]

                if not rel_parts or rel_parts == [".."]:
                    # Points to same directory or parent
                    new_url = "/".join(rel_parts) if rel_parts else "."
                else:
                    new_url = "/".join(rel_parts)

            return f"[{text}]({new_url})"
        except (ValueError, IndexError):
            return m.group(0)

    return LINK_RE.sub(repl, md)


def write_qmd(out_path: Path, title: str, body_md: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fm = f"---\ntitle: {title!r}\nformat:\n  html:\n    toc: true\n---\n\n"
    out_path.write_text(fm + body_md, encoding="utf-8")


def update_quarto_yml(generated: list[tuple[str, str, str]]):
    """
    Update _quarto.yml with the generated example files in the correct order.
    This keeps the sidebar in sync with the allowlist.
    """
    quarto_yml = ROOT / "_quarto.yml"
    if not quarto_yml.exists():
        print(f"[WARN] {quarto_yml} not found, skipping update", file=sys.stderr)
        return

    content = quarto_yml.read_text(encoding="utf-8")

    # Pattern to match just the Model Guides contents
    pattern = r'(        - section: "Model Guides"\n          contents:)([^\n]*|.*?)(?=\n        - (?:section|title):|\n\nformat:)'

    def replacement(match):
        prefix = match.group(1)
        lines = "\n".join(
            [
                f"            - docs/examples/{path}.qmd"
                for path, name, title in generated
            ]
        )
        return prefix + "\n" + lines + "\n"

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    if new_content != content:
        quarto_yml.write_text(new_content, encoding="utf-8")
        print(f"Updated {quarto_yml}")
    else:
        print(f"No changes needed for {quarto_yml}")


def main():
    allow = read_allowlist()
    if not EXAMPLES_DIR.exists():
        print(f"[WARN] {EXAMPLES_DIR} not found", file=sys.stderr)
        return

    (OUTPUT_DIR / "assets").mkdir(parents=True, exist_ok=True)

    # First pass: identify which parents have their own entry vs only sub-entries
    parent_entries = set()  # Parents that have their own entry
    parent_with_subs = set()  # Parents that have sub-entries
    allowlist_entries = set()  # All entries in allowlist

    for item in allow:
        if isinstance(item, str):
            name = item
        else:
            name = item.get("name")

        allowlist_entries.add(name)

        if "/" in name:
            parent = name.split("/")[0]
            parent_with_subs.add(parent)
        else:
            parent_entries.add(name)

    # Parents with subs that DON'T have their own entry -> use index.qmd
    parent_index_only = parent_with_subs - parent_entries

    generated = []
    seen_dirs = set()  # Track which parent directories we've created index for

    for item in allow:
        if isinstance(item, str):
            name = item
            title = None
        else:
            name = item.get("name")
            title = item.get("title")

        if not name:
            print(f"[WARN] Skipping item without name: {item}", file=sys.stderr)
            continue

        src_dir = EXAMPLES_DIR / name
        if not src_dir.exists() or not src_dir.is_dir():
            print(f"[WARN] Skipping {name} (not a directory)", file=sys.stderr)
            continue

        readme = find_readme(src_dir)
        if not readme:
            print(f"[WARN] Skipping {name} (no README.md)", file=sys.stderr)
            continue

        md = readme.read_text(encoding="utf-8")

        # Determine output path first (needed for link rewriting)
        parts = name.split("/")
        if len(parts) == 1:
            # Simple case: no subdirectory
            parent = parts[0]
            # Check if this parent should use index.qmd instead
            if parent in parent_index_only:
                out_path = OUTPUT_DIR / parent / "index.qmd"
                sidebar_path = f"{parent}/index"
            else:
                out_path = OUTPUT_DIR / f"{parts[0]}.qmd"
                sidebar_path = parts[0]
        else:
            # Has subdirectory: e.g., magistral/think
            parent = parts[0]
            child = "-".join(parts[1:])  # handle nested subdirs
            out_path = OUTPUT_DIR / parent / f"{child}.qmd"
            sidebar_path = f"{parent}/{child}"

        # Remove the first H1 (we use frontmatter title instead)
        md, _ = remove_first_h1(md)
        # Rewrite links between README files
        md = rewrite_readme_links(
            md,
            src_dir,
            EXAMPLES_DIR,
            parent_index_only,
            name,
            allowlist_entries,
            sidebar_path,
        )
        md = rewrite_and_copy_assets(md, src_dir, OUTPUT_DIR)

        # Handle parent index generation for sub-entries
        if len(parts) > 1:
            # Has subdirectory: e.g., magistral/think
            parent = parts[0]

            # Create index.qmd for parent if not already done and parent doesn't have own entry
            if parent not in seen_dirs and parent in parent_index_only:
                parent_readme = find_readme(EXAMPLES_DIR / parent)
                if parent_readme:
                    parent_md = parent_readme.read_text(encoding="utf-8")
                    parent_md, _ = remove_first_h1(parent_md)
                    parent_md = rewrite_readme_links(
                        parent_md,
                        EXAMPLES_DIR / parent,
                        EXAMPLES_DIR,
                        parent_index_only,
                        parent,
                        allowlist_entries,
                        f"{parent}/index",
                    )
                    parent_md = rewrite_and_copy_assets(
                        parent_md, EXAMPLES_DIR / parent, OUTPUT_DIR
                    )
                    parent_title = parent.replace("-", " ").replace("/", " ").title()
                    write_qmd(
                        OUTPUT_DIR / parent / "index.qmd", parent_title, parent_md
                    )
                    generated.append((f"{parent}/index", parent, parent_title))
                    seen_dirs.add(parent)

        if not title:
            title = name.replace("/", " ").replace("-", " ").title()

        write_qmd(out_path, title, md)
        generated.append((sidebar_path, name, title))

    # Index page - preserve allowlist order
    if generated:
        listing = "\n".join(
            [f"- [{title}]({path}.qmd)" for path, name, title in generated]
        )
        index_md = (
            "# Model Guides\n\nBelow are the curated examples for training various model architectures:\n\n"
            + listing
            + "\n"
        )
        index_fm = "---\nformat:\n  html:\n    toc: true\n---\n\n"
        (OUTPUT_DIR / "index.qmd").write_text(index_fm + index_md, encoding="utf-8")

        # Auto-update _quarto.yml to keep sidebar in sync
        update_quarto_yml(generated)


if __name__ == "__main__":
    main()
