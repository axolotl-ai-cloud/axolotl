# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The package imports without axolotl, transformers or TRL."""

import subprocess
import sys

MODULES = [
    "tree_recurrence",
    "tree_recurrence.components",
    "tree_recurrence.integrations.hub_kernels",
    "tree_recurrence.integrations.transformers",
    "tree_recurrence.integrations.trl",
    "tree_recurrence.reference",
]


def test_imports_pull_in_no_framework():
    code = (
        "import sys\n"
        + "".join(f"import {module}\n" for module in MODULES)
        + "loaded = sorted({name.split('.')[0] for name in sys.modules}"
        " & {'axolotl', 'transformers', 'trl', 'kernels', 'fla', 'mamba_ssm'})\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
