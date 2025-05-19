#!/usr/bin/env python
"""
Script to update all test files to use the standardized logging approach.
"""

import os
import re
import sys
from pathlib import Path


def update_file(file_path, dry_run=False):
    """Update a file to use the standardized logging approach."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Keep track of changes
    changes_made = False

    # Replace the import if it's a standalone import
    import_pattern = r"import\s+logging\s*\n"
    if re.search(import_pattern, content):
        new_content = re.sub(
            import_pattern, "from axolotl.utils.logging import get_logger\n", content
        )
        changes_made = new_content != content
        content = new_content

    # Replace the logger initialization
    logger_pattern = r'LOG\s*=\s*logging\.getLogger\([\'"]([^\'"]+)[\'"]\)'
    if re.search(logger_pattern, content):
        new_content = re.sub(logger_pattern, r'LOG = get_logger("\1")', content)
        changes_made = changes_made or (new_content != content)
        content = new_content

    # Remove logging.basicConfig if present
    basicconfig_pattern = r"logging\.basicConfig\([^\)]+\)\s*\n"
    if re.search(basicconfig_pattern, content):
        new_content = re.sub(basicconfig_pattern, "", content)
        changes_made = changes_made or (new_content != content)
        content = new_content

    if changes_made and not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    return changes_made


def find_and_update_files(base_dir, dry_run=False):
    """Find and update all test files that use logging."""
    updated_files = []
    skipped_files = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if (
                    "import logging" in content
                    or "logging.getLogger" in content
                    or "logging.basicConfig" in content
                ):
                    if "from axolotl.utils.logging import get_logger" in content:
                        if "import logging" in content:
                            # Both imports present, probably needs manual inspection
                            skipped_files.append(file_path)
                        else:
                            # Already using the standardized logger
                            pass
                    else:
                        if update_file(file_path, dry_run):
                            updated_files.append(file_path)

    return updated_files, skipped_files


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        sys.argv.remove("--dry-run")

    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "tests"

    updated_files, skipped_files = find_and_update_files(base_dir, dry_run)

    if dry_run:
        print(f"DRY RUN: Would update {len(updated_files)} files:")
    else:
        print(f"Updated {len(updated_files)} files:")

    for file in updated_files:
        rel_path = os.path.relpath(file, os.getcwd())
        print(f"  - {rel_path}")

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files (need manual inspection):")
        for file in skipped_files:
            rel_path = os.path.relpath(file, os.getcwd())
            print(f"  - {rel_path}")
