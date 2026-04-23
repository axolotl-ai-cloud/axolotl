#!/usr/bin/env python3
"""Build a disposable Hugging Face Kernel Hub package for ScatterMoE LoRA.

This script does not move or edit the in-tree Axolotl kernel sources. It copies
``src/axolotl/integrations/kernels/libs/scattermoe_lora`` into an ignored
build directory and emits a universal HF kernels project that can be pushed to
the Hub.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGE_NAME = "scattermoe_lora"
BUILD_VARIANT = "torch-universal"
DEFAULT_REPO_ID = "kernels-community/scattermoe-lora"
HF_REPO_TYPE = "kernel"
HF_KERNEL_URL_PREFIX = "https://hf.co/kernels"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    REPO_ROOT / "src" / "axolotl" / "integrations" / "kernels" / "libs" / PACKAGE_NAME
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "hf-kernels" / PACKAGE_NAME

EXCLUDED_DIRS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
EXCLUDED_FILE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.so",
    ".DS_Store",
}

TEXT_REPLACEMENTS = {
    "from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant import": (
        "from .selective_dequant import"
    ),
    "from axolotl.integrations.kernels.libs.scattermoe_lora.selective_dequant_kernel import": (
        "from .selective_dequant_kernel import"
    ),
    "from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops import": (
        "from .ops import"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy Axolotl's ScatterMoE LoRA Triton kernels into a disposable "
            "HF Kernel Hub universal package."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"ScatterMoE LoRA source package to copy. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination build/dist directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF Hub repo id to write into build.toml. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Kernel major version written to build.toml and metadata.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--no-source-layout",
        action="store_true",
        help="Only write the shippable build/ tree, not torch-ext/ sources.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "Upload the generated universal kernel package with huggingface_hub. "
            "This bypasses kernel-builder and is intended for pure Python/Triton "
            "universal kernels."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the HF Hub repo as private when used with --upload.",
    )
    parser.add_argument(
        "--skip-version-branch",
        action="store_true",
        help="With --upload, only upload main and skip the v<version> branch.",
    )
    return parser.parse_args()


def should_skip_file(path: Path) -> bool:
    return any(
        fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_FILE_PATTERNS
    )


def iter_source_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirs, filenames in os.walk(source_dir):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for filename in sorted(filenames):
            path = Path(root) / filename
            if not should_skip_file(path):
                files.append(path)
    return files


def content_hash(source_dir: Path) -> str:
    digest = hashlib.sha1()
    for path in iter_source_files(source_dir):
        rel = path.relative_to(source_dir).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:10]


def git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def transform_python_source(text: str, rel_path: Path, op_namespace: str) -> str:
    for old, new in TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)

    if rel_path.as_posix() == "gemma4_experts.py":
        text = text.replace(
            "    from axolotl.integrations.kernels.constants import resolve_experts_class",
            (
                "    raise RuntimeError(\n"
                '        "patch_gemma4_scattermoe is only available from the in-tree Axolotl "\n'
                '        "integration. Use register_scattermoe_experts() with the standalone "\n'
                '        "HF kernel package."\n'
                "    )"
            ),
        )

    return text.replace("scattermoe::", f"{op_namespace}::")


def copy_package(source_dir: Path, package_dir: Path, op_namespace: str) -> None:
    for source in iter_source_files(source_dir):
        rel_path = source.relative_to(source_dir)
        destination = package_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if source.suffix == ".py":
            text = source.read_text(encoding="utf-8")
            text = transform_python_source(text, rel_path, op_namespace)
            destination.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(source, destination)

    write_ops_module(package_dir / "_ops.py", op_namespace)


def write_ops_module(path: Path, op_namespace: str) -> None:
    path.write_text(
        "\n".join(
            [
                "import torch",
                "",
                f"ops = torch.ops.{op_namespace}",
                "",
                "",
                "def add_op_namespace_prefix(op_name: str) -> str:",
                f'    return f"{op_namespace}::{{op_name}}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_build_toml(path: Path, repo_id: str, version: int) -> None:
    lines = [
        "[general]",
        f'name = "{PACKAGE_NAME}"',
        "universal = true",
        f"version = {version}",
        "",
    ]
    if repo_id:
        lines.extend(
            [
                "[general.hub]",
                f'repo-id = "{repo_id}"',
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_flake(path: Path) -> None:
    path.write_text(
        """{
  description = "Flake for scattermoe_lora kernel";

  inputs = {
    builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      builder,
    }:
    builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
""",
        encoding="utf-8",
    )


def write_readme(path: Path, repo_id: str, source_hash: str, op_namespace: str) -> None:
    repo_display = repo_id or "<your-org>/scattermoe-lora"
    path.write_text(
        f"""---
library_name: kernels
license: apache-2.0
tags:
- kernel
- kernels
---

# ScatterMoE LoRA

Standalone Hugging Face Kernel Hub package for Axolotl's ScatterMoE LoRA Triton kernels.

This package is generated from Axolotl's in-tree `scattermoe_lora` sources and is exported as a universal kernel because the implementation is Python/Triton rather than a precompiled C++/CUDA extension.

```python
from kernels import get_kernel

scattermoe_lora = get_kernel("{repo_display}")
```

Export metadata:

- source package: `src/axolotl/integrations/kernels/libs/scattermoe_lora`
- source revision: `{git_revision()}`
- source content hash: `{source_hash}`
- torch custom op namespace: `{op_namespace}`

The generated `build/torch-universal/{PACKAGE_NAME}` directory is the shippable Hub artifact. `torch-ext/{PACKAGE_NAME}` is included so `kernel-builder build-and-copy` can regenerate the universal build tree if desired.
""",
        encoding="utf-8",
    )


def write_metadata(path: Path, version: int) -> None:
    path.write_text(
        json.dumps({"version": version}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"{output_dir} already exists. Re-run with --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def build_package(args: argparse.Namespace) -> Path:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source package does not exist: {source_dir}")
    if not (source_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"source package is missing __init__.py: {source_dir}")

    source_hash = content_hash(source_dir)
    op_namespace = f"_{PACKAGE_NAME}_{source_hash}"

    prepare_output_dir(output_dir, args.force)

    write_build_toml(output_dir / "build.toml", args.repo_id, args.version)
    write_flake(output_dir / "flake.nix")
    write_readme(output_dir / "README.md", args.repo_id, source_hash, op_namespace)

    if not args.no_source_layout:
        copy_package(source_dir, output_dir / "torch-ext" / PACKAGE_NAME, op_namespace)

    build_package_dir = output_dir / "build" / BUILD_VARIANT / PACKAGE_NAME
    copy_package(source_dir, build_package_dir, op_namespace)
    write_metadata(build_package_dir.parent / "metadata.json", args.version)

    return output_dir


def upload_package(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.repo_id:
        raise ValueError("--repo-id is required when using --upload")

    try:
        from huggingface_hub import HfApi, constants as hf_constants
    except ImportError as exc:
        raise RuntimeError(
            "--upload requires huggingface_hub. Install it or run the upload "
            "manually with the Hugging Face CLI."
        ) from exc

    try:
        hub_version = metadata.version("huggingface_hub")
    except metadata.PackageNotFoundError:
        hub_version = "unknown"

    accepted_repo_types = getattr(
        hf_constants,
        "REPO_TYPES_WITH_KERNEL",
        getattr(hf_constants, "REPO_TYPES", ()),
    )
    if HF_REPO_TYPE not in accepted_repo_types:
        raise RuntimeError(
            "Your huggingface_hub installation does not support "
            f"repo_type={HF_REPO_TYPE!r} (found huggingface_hub {hub_version}). "
            f"Upgrade this interpreter with: {sys.executable} -m pip install --upgrade "
            "'huggingface_hub>=1.10.0'"
        )

    # huggingface_hub 1.11.0 has partial kernel support: create_repo accepts
    # "kernel", but upload_folder/create_commit still validate against the
    # older REPO_TYPES list. Extend it in-process so those helpers use the
    # /api/kernels/... endpoints until upstream broadens that check.
    if HF_REPO_TYPE not in hf_constants.REPO_TYPES:
        hf_constants.REPO_TYPES.append(HF_REPO_TYPE)

    api = HfApi()
    try:
        repo_id = api.create_repo(
            repo_id=args.repo_id,
            repo_type=HF_REPO_TYPE,
            private=args.private,
            exist_ok=True,
        ).repo_id
    except ValueError as exc:
        if "Invalid repo type" in str(exc):
            raise RuntimeError(
                "huggingface_hub rejected repo_type='kernel'. "
                f"This usually means the command is running with an older Hub "
                f"client than expected (found huggingface_hub {hub_version} at "
                f"{sys.executable}). Upgrade with: {sys.executable} -m pip "
                "install --upgrade 'huggingface_hub>=1.10.0'"
            ) from exc
        raise

    delete_patterns = [
        "build/**",
        "torch-ext/**",
        "build.toml",
        "flake.nix",
        "README.md",
    ]

    api.upload_folder(
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        folder_path=output_dir,
        revision="main",
        delete_patterns=delete_patterns,
        commit_message="Upload ScatterMoE LoRA universal kernel",
    )
    print(f"Uploaded main branch: {HF_KERNEL_URL_PREFIX}/{repo_id}")

    if args.skip_version_branch:
        return

    version_branch = f"v{args.version}"
    api.create_branch(
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        branch=version_branch,
        revision="main",
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        folder_path=output_dir,
        revision=version_branch,
        delete_patterns=delete_patterns,
        commit_message=f"Upload ScatterMoE LoRA universal kernel {version_branch}",
    )
    print(
        f"Uploaded version branch: "
        f"{HF_KERNEL_URL_PREFIX}/{repo_id}/tree/{version_branch}"
    )


def main() -> int:
    args = parse_args()
    try:
        output_dir = build_package(args)
        if args.upload:
            upload_package(args, output_dir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote ScatterMoE LoRA HF kernel package to: {output_dir}")
    print(f"Shippable artifact: {output_dir / 'build' / BUILD_VARIANT / PACKAGE_NAME}")
    if args.upload:
        print(f'Load it with: get_kernel("{args.repo_id}", version={args.version})')
        print(f"Uploaded as Hugging Face repo_type={HF_REPO_TYPE!r}.")
        return 0

    print("Next step:")
    print("  upload this universal Python/Triton kernel directly:")
    print(
        f"    python3 {Path(__file__).as_posix()} "
        f"--repo-id {args.repo_id} --force --upload"
    )
    if shutil.which("kernel-builder") is None:
        print("  optional: install kernel-builder for full Nix-based builds:")
        print(
            "    curl -fsSL "
            "https://raw.githubusercontent.com/huggingface/kernels/main/install.sh "
            "| bash"
        )
    else:
        print("  optional: upload with kernel-builder:")
        print(f"    cd {output_dir}")
        print("    kernel-builder build-and-upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
