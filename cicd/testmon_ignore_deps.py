#!/usr/bin/env python
"""Print the installed distributions testmon should leave out of its environment check.

testmon drops its whole baseline when any installed distribution changes at major.minor,
and the CI lanes install ~400 mostly unpinned packages, so a tzdata or openai release was
enough to re-run every test. Only the libraries below still invalidate the baseline.

Usage: pytest --testmon -o "testmon_ignore_dependencies=$(python cicd/testmon_ignore_deps.py)"
"""

from __future__ import annotations

import importlib.metadata
import re
import sys

TRACKED = frozenset(
    {
        "pytest",
        # core ML stack
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "safetensors",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "numpy",
        "pydantic",
        "jinja2",
        # kernels and quantization
        "torchao",
        "bitsandbytes",
        "liger-kernel",
        "cut-cross-entropy",
        "kernels",
        "triton",
        "xformers",
        "fla-core",
        "flash-linear-attention",
        # axolotl companions and training-path extras
        "axolotl-contribs-lgpl",
        "axolotl-contribs-mit",
        "axolotl-ringmaster",
        "mistral-common",
        "schedulefree",
    }
)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    installed = {}
    for dist in importlib.metadata.distributions():
        # testmon matches the raw metadata Name, so that is what gets printed
        name = dist.metadata["Name"]
        if name:
            installed[name] = dist.version
    tracked = sorted(n for n in installed if normalize(n) in TRACKED)
    print(
        "testmon: environment check covers "
        + ", ".join(f"{n} {installed[n]}" for n in tracked),
        file=sys.stderr,
    )
    print(" ".join(sorted(n for n in installed if normalize(n) not in TRACKED)))


if __name__ == "__main__":
    main()
