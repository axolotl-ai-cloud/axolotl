#!/usr/bin/env bash
# Install the dependencies for Axolotl's native NVFP4 training feature set.
#
# A vanilla Axolotl install does not pull these in. NVFP4 training on Blackwell
# (sm_120) additionally needs:
#
#   1. A CUDA 13.0 PyTorch (>= 2.11) whose bundled Triton exposes `tl.dot_scaled`
#      with native NVFP4 (e2m1 data + e4m3 group-16 scales) — the FP4 tensor-core
#      path the kernels emit. Stable CUDA wheels do not yet ship this; use the
#      cu130 (or nightly cu130) index.
#   2. transformers >= 4.57  (Qwen3.5 / `qwen3_5` model support).
#   3. mslk  —  github.com/meta-pytorch/MSLK, Meta/PyTorch's FP4 quant Triton kernels
#      (`triton_quantize_nvfp4` etc.), used by the NVFP4 base GEMM path.
#      Published on the PyTorch wheel index.
#
# Defaults follow Axolotl's uv-based workflow. Pass `--tool pip` for plain pip,
# or `--create-venv PATH` to provision an isolated env first.
#
# Usage:
#   scripts/install_nvfp4.sh [options]
#     --tool {uv|pip}        installer frontend (default: uv)
#     --create-venv PATH     create + activate a fresh venv at PATH first
#     --with-torch           also (re)install torch/vision/audio from --torch-index
#     --torch-index URL      cu130 torch index (default below)
#     --mslk-index URL       index for mslk (default: stable cu130 — ABI-matches
#                            the stable cu130 torch this script installs)
#     --mslk-stable          force the stable cu130 index for mslk (the default)
#     --mslk-nightly         use the nightly cu130 index for mslk (--pre); pair
#                            this with a nightly cu130 torch or the ABIs mismatch
#     --build-mslk           build mslk from source (github.com/meta-pytorch/MSLK) instead
#     --no-axolotl           skip installing this Axolotl checkout (deps only)
#     -h|--help              show this help
set -euo pipefail

TOOL="uv"
CREATE_VENV=""
WITH_TORCH=0
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
# Default to the STABLE cu130 index: it ABI-matches the stable cu130 torch this
# script installs. The nightly mslk wheel races ahead of stable torch and dlopens
# with an undefined c10:: symbol (ABI break) — only use it with a nightly torch.
MSLK_INDEX="https://download.pytorch.org/whl/cu130"
MSLK_PRE=""
BUILD_MSLK=0
MSLK_REPO="https://github.com/meta-pytorch/MSLK.git"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_AXOLOTL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool)
      case "${2:-}" in
        uv|pip) TOOL="$2";;
        *) echo "ERROR: --tool must be one of: uv, pip" >&2; exit 2;;
      esac
      shift 2;;
    --create-venv) CREATE_VENV="$2"; shift 2;;
    --with-torch) WITH_TORCH=1; shift;;
    --torch-index) TORCH_INDEX="$2"; shift 2;;
    --mslk-index) MSLK_INDEX="$2"; shift 2;;
    --mslk-stable) MSLK_INDEX="https://download.pytorch.org/whl/cu130"; MSLK_PRE=""; shift;;
    --mslk-nightly) MSLK_INDEX="https://download.pytorch.org/whl/nightly/cu130"; MSLK_PRE="--pre"; shift;;
    --build-mslk) BUILD_MSLK=1; shift;;
    --no-axolotl) INSTALL_AXOLOTL=0; shift;;
    -h|--help) sed -n '2,33p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

say(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# --- optional fresh venv (uv preferred, else stdlib venv) ----------------------
if [[ -n "$CREATE_VENV" ]]; then
  say "Creating venv at $CREATE_VENV"
  if [[ "$TOOL" == "uv" ]] && command -v uv >/dev/null 2>&1; then
    uv venv "$CREATE_VENV"
  else
    python3 -m venv "$CREATE_VENV"
  fi
  # shellcheck disable=SC1091
  source "$CREATE_VENV/bin/activate"
fi

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: no active environment. Activate a venv/conda env, or pass --create-venv PATH." >&2
  exit 1
fi

# `uv pip` installs into the active venv; pip uses the active interpreter.
PIP(){ if [[ "$TOOL" == "uv" ]] && command -v uv >/dev/null 2>&1; then uv pip "$@"; else python -m pip "$@"; fi; }

# --- 1. torch (cu130) ----------------------------------------------------------
if [[ "$WITH_TORCH" == "1" ]]; then
  say "Installing PyTorch (cu130) from $TORCH_INDEX"
  PIP install torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi

# --- 2. transformers -----------------------------------------------------------
# >=4.57 brings Qwen3.5 (`qwen3_5`) support. The NVFP4 stack is validated on
# transformers 5.9.0 + liger-kernel 0.8.0 (the versions pinned in pyproject); the
# `axolotl` install below pins `transformers==5.9.0`, so this just enforces the floor.
say "Installing transformers >= 4.57 (Qwen3.5 support; validated on 5.9.0)"
PIP install "transformers>=4.57.0"

# --- 3. mslk (FP4 quant kernels for the base GEMM path) ------------------------
if [[ "$BUILD_MSLK" == "1" ]]; then
  say "Building mslk from source ($MSLK_REPO)"
  MSLK_DIR="$(dirname "$REPO_ROOT")/MSLK"
  [[ -d "$MSLK_DIR/.git" ]] || git clone "$MSLK_REPO" "$MSLK_DIR"
  ( cd "$MSLK_DIR" && PIP install -e . )
else
  say "Installing mslk from the PyTorch index ($MSLK_INDEX)"
  # shellcheck disable=SC2086
  PIP install $MSLK_PRE mslk --index-url "$MSLK_INDEX"
fi

# --- 4. Axolotl + the flash-attn extra -----------------------------------------
if [[ "$INSTALL_AXOLOTL" == "1" ]]; then
  say "Installing Axolotl + nvfp4 + flash-attn extras"
  # nvfp4 extra resolves the declared NVFP4 deps (mslk, triton>=3.7). flash-attn
  # extra: the recommended recipe sets attn_implementation: flash_attention_2
  # (required on Blackwell). flash-attn 2.8.3 has no cu130 wheel — it builds from
  # source, so wheel/setuptools/ninja must be present (no-build-isolation).
  PIP install wheel setuptools ninja packaging psutil
  MAX_JOBS="${MAX_JOBS:-16}" \
    PIP install -e "${REPO_ROOT}[nvfp4,flash-attn]" --no-build-isolation
  # torchao is pinned ==0.17.0; pull the cu130 build (matches the validated env) so the
  # NVFP4Tensor prototype mx_formats path lines up with the cu130 torch.
  if [[ "$TOOL" == "uv" ]] && command -v uv >/dev/null 2>&1; then
    PIP install --reinstall torchao==0.17.0 --index-url "$TORCH_INDEX"
  else
    PIP install --force-reinstall torchao==0.17.0 --index-url "$TORCH_INDEX"
  fi
fi

# --- 5. validate ---------------------------------------------------------------
say "Validating the NVFP4 toolchain"
python - <<'PY'
import importlib, re, sys
def ok(msg): print(f"  [ok] {msg}")
def bad(msg): print(f"  [!!] {msg}"); sys.exit(1)

import torch
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
ok(f"torch {torch.__version__} (CUDA {torch.version.cuda}, device cap {cap[0]}.{cap[1]})")
if cap[0] < 10:
    print("  [warn] no Blackwell device visible — NVFP4 kernels need sm_100/sm_120 at runtime.")

import triton
ok(f"triton {triton.__version__}")
tv = tuple(int(x) for x in re.findall(r"\d+", triton.__version__)[:2])
if tv < (3, 7):
    bad(f"triton {triton.__version__} < 3.7 — the nvfp4 extra requires triton>=3.7.0.")
if not hasattr(triton.language, "dot_scaled"):
    bad("triton.language.dot_scaled missing — this torch/triton lacks native FP4 GEMM support.")

import transformers
ok(f"transformers {transformers.__version__}")

importlib.import_module("mslk")
ok("mslk importable (base GEMM FP4 path)")
print("\nNVFP4 feature set ready.")
PY
