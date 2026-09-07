#!/bin/bash

# Keep CUDA 13/cu130 uv images working when NVIDIA ships the cu13 package.
# This is a no-op on images without nvidia.cu13.
axolotl_prepend_cu13_ld_library_path() {
    local updated_ld_library_path
    if updated_ld_library_path="$(python - <<'PY'
from axolotl.utils.cuda13 import prepend_cu13_ld_library_path
import os

print(prepend_cu13_ld_library_path(os.environ.get("LD_LIBRARY_PATH")))
PY
    )"; then
        export LD_LIBRARY_PATH="$updated_ld_library_path"
    fi
}

axolotl_prepend_cu13_ld_library_path
unset -f axolotl_prepend_cu13_ld_library_path
