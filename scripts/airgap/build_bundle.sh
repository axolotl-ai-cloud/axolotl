#!/usr/bin/env bash
# Build an air-gap bundle for Axolotl on an ONLINE staging host.
# Companion to docs/airgapped.qmd; every step below maps to a section there.
#
# Usage:
#   scripts/airgap/build_bundle.sh [step ...]
#
# With no arguments every step runs in order. Pass step names to run a subset,
# e.g. `build_bundle.sh hf_cache manifest archive` after fixing a download.
#
# Knobs (environment variables, defaults in parentheses):
#   BUNDLE          output directory              ($HOME/axolotl-airgap)
#   AXOLOTL_REF     git ref to clone              (main)
#   UV_VERSION      uv release to bundle          (version of the uv on PATH)
#   PY_VERSION      CPython version               (3.14.4)
#   PBS_RELEASE     python-build-standalone tag   (20260414)
#   TARGET_ARCH     x86_64 | aarch64              (uname -m of this host)
#   TORCH_BACKEND   uv torch backend              (cu130)
#   CONFIG          training config to bundle     ($BUNDLE/axolotl/examples/llama-3/lora-1b.yml)
#   KERNELS         space separated: fa2 fa3 fa4  (fa2)
#   EXTRAS          comma separated pyproject extras   (fla,ringmaster,ray,deepspeed; aarch64: ringmaster,ray)
#                   deepspeed is sdist-only and is built here with DS_BUILD_OPS=0
#   WITH_CCE        1 builds the cut-cross-entropy plugin wheel from its pinned git tag (1)
#   PREPROCESS      1 tokenizes the dataset on staging (1)
#   SPLIT_SIZE      split(1) chunk size, empty for no split (empty)
#   STAGING_VENV    venv used for hf/kernels/axolotl preprocess ($BUNDLE/../.venv-airgap-staging)
set -euo pipefail

BUNDLE=${BUNDLE:-$HOME/axolotl-airgap}
AXOLOTL_REF=${AXOLOTL_REF:-main}
PY_VERSION=${PY_VERSION:-3.14.4}
PBS_RELEASE=${PBS_RELEASE:-20260414}
TARGET_ARCH=${TARGET_ARCH:-$(uname -m)}
TORCH_BACKEND=${TORCH_BACKEND:-cu130}
KERNELS=${KERNELS:-fa2}
if [ "${TARGET_ARCH:-$(uname -m)}" = aarch64 ]; then EXTRAS=${EXTRAS-ringmaster,ray}; else EXTRAS=${EXTRAS-fla,ringmaster,ray,deepspeed}; fi
WITH_CCE=${WITH_CCE:-1}
CCE_REQ=${CCE_REQ:-cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@v0.1.0-rc0}
PREPROCESS=${PREPROCESS:-1}
SPLIT_SIZE=${SPLIT_SIZE:-}
STAGING_VENV=${STAGING_VENV:-$(dirname "$BUNDLE")/.venv-airgap-staging}
HOST_ARCH=$(uname -m)
PY_MINOR=${PY_VERSION%.*}            # 3.14
PY_TAG=cp${PY_MINOR/./}              # cp314

# Pins the target can not build; see "Split out the packages without wheels".
PUREPY_SDISTS="langdetect rouge-score sqlitedict word2number axolotl-contribs-lgpl axolotl-contribs-mit"
NOWHEEL_RE='^(zstandard|langdetect|rouge-score|sqlitedict|word2number|axolotl-contribs-lgpl|axolotl-contribs-mit|deepspeed)=='

log()  { printf '\n==> %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

unset UV_TORCH_BACKEND HF_HUB_CACHE
export HF_HOME=$BUNDLE/hf
export UV_PYTHON_DOWNLOADS=${UV_PYTHON_DOWNLOADS:-automatic}

step_prepare() {
  need git; need curl; need uv; need tar; need sha256sum
  case $TARGET_ARCH in x86_64|aarch64) ;; *) die "TARGET_ARCH must be x86_64 or aarch64";; esac
  UV_VERSION=${UV_VERSION:-$(uv --version | awk '{print $2}')}
  log "bundle=$BUNDLE target=$TARGET_ARCH uv=$UV_VERSION python=$PY_VERSION+$PBS_RELEASE torch=$TORCH_BACKEND kernels=$KERNELS"
  mkdir -p "$BUNDLE"/{uv,python-mirror,wheelhouse,hf}
  [ "$(uv --version | awk '{print $2}')" = "$UV_VERSION" ] || die "uv on PATH is not $UV_VERSION; install it with: curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | sh"
}

step_clone() {
  log "clone axolotl @ $AXOLOTL_REF"
  if [ ! -d "$BUNDLE/axolotl/.git" ]; then
    git clone https://github.com/axolotl-ai-cloud/axolotl.git "$BUNDLE/axolotl"
  fi
  git -C "$BUNDLE/axolotl" fetch --tags origin
  git -C "$BUNDLE/axolotl" checkout --quiet "$AXOLOTL_REF"
  git -C "$BUNDLE/axolotl" rev-parse HEAD
  CONFIG=${CONFIG:-$BUNDLE/axolotl/examples/llama-3/lora-1b.yml}
  [ -f "$CONFIG" ] || die "CONFIG not found: $CONFIG"
}

step_uv() {
  log "fetch uv $UV_VERSION for $TARGET_ARCH"
  local base="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
  local f="uv-$TARGET_ARCH-unknown-linux-gnu.tar.gz"
  (cd "$BUNDLE/uv" && curl -fLO "$base/$f" && curl -fLO "$base/$f.sha256" && sha256sum -c "$f.sha256")
}

step_python() {
  log "fetch python-build-standalone $PY_VERSION+$PBS_RELEASE for $TARGET_ARCH"
  local f="cpython-$PY_VERSION+$PBS_RELEASE-$TARGET_ARCH-unknown-linux-gnu-install_only_stripped.tar.gz"
  local url="https://releases.astral.sh/github/python-build-standalone/releases/download/$PBS_RELEASE/${f/+/%2B}"
  mkdir -p "$BUNDLE/python-mirror/$PBS_RELEASE"
  curl -fL -o "$BUNDLE/python-mirror/$PBS_RELEASE/$f" "$url"
  # The same interpreter for this host, used to build wheels below.
  uv python install "$PY_VERSION"
}

step_resolve() {
  log "resolve pins for $TARGET_ARCH"
  local extra=() e
  for e in ${EXTRAS//,/ }; do extra+=(--extra "$e"); done
  local inputs=(pyproject.toml)
  if [[ " $KERNELS " == *" fa4 "* ]]; then
    printf '%s\n' 'nvidia-cutlass-dsl[cu13]' apache-tvm-ffi einops > "$BUNDLE/kernel-deps.in"
    inputs+=("$BUNDLE/kernel-deps.in")
  fi
  (cd "$BUNDLE/axolotl" && uv pip compile "${inputs[@]}" "${extra[@]}" \
      --python-version "$PY_MINOR" \
      --python-platform "$TARGET_ARCH-manylinux_2_28" \
      --torch-backend "$TORCH_BACKEND" \
      --no-annotate --no-header \
      -o "$BUNDLE/requirements.target.txt")
  grep -v -E -i "$NOWHEEL_RE" "$BUNDLE/requirements.target.txt" > "$BUNDLE/requirements.binary.txt"
}

step_wheels() {
  log "download binary wheels"
  local plat=() v
  for v in $(seq 28 -1 5); do plat+=(--platform "manylinux_2_${v}_$TARGET_ARCH"); done
  plat+=(--platform "manylinux2014_$TARGET_ARCH" --platform "manylinux2010_$TARGET_ARCH" --platform "manylinux1_$TARGET_ARCH")
  uvx pip download --no-deps --only-binary=:all: "${plat[@]}" \
    --python-version "$PY_MINOR" --implementation cp \
    --abi "$PY_TAG" --abi abi3 --abi none \
    --extra-index-url "https://download.pytorch.org/whl/$TORCH_BACKEND" \
    -r "$BUNDLE/requirements.binary.txt" -d "$BUNDLE/wheelhouse"

  log "build pure-python wheels"
  local pins
  pins=$(grep -E -i "^($(echo "$PUREPY_SDISTS" | tr ' ' '|'))==" "$BUNDLE/requirements.target.txt")
  # shellcheck disable=SC2086
  uvx --python "$PY_MINOR" pip wheel --no-deps -w "$BUNDLE/wheelhouse" $pins

  if [ "$HOST_ARCH" = "$TARGET_ARCH" ]; then
    log "build zstandard (no build isolation)"
    local zpin; zpin=$(grep -i '^zstandard==' "$BUNDLE/requirements.target.txt")
    uv venv --clear --seed --python "$PY_MINOR" "$BUNDLE/../.zstd-build"
    "$BUNDLE/../.zstd-build/bin/python" -m pip install -q setuptools wheel cffi
    "$BUNDLE/../.zstd-build/bin/python" -m pip wheel --no-deps --no-build-isolation -w "$BUNDLE/wheelhouse" "$zpin"
    if grep -qi '^deepspeed==' "$BUNDLE/requirements.target.txt"; then
      log "build deepspeed (DS_BUILD_OPS=0; ops JIT-compile on the target)"
      local dpin; dpin=$(grep -i '^deepspeed==' "$BUNDLE/requirements.target.txt")
      uv venv --clear --seed --python "$PY_MINOR" "$BUNDLE/../.ds-build"
      uv pip install --python "$BUNDLE/../.ds-build/bin/python" --torch-backend "$TORCH_BACKEND" setuptools wheel ninja "$(grep -i '^torch==' "$BUNDLE/requirements.target.txt" | sed 's/+.*//')"
      DS_BUILD_OPS=0 "$BUNDLE/../.ds-build/bin/python" -m pip wheel --no-deps --no-build-isolation -w "$BUNDLE/wheelhouse" "$dpin"
    fi
  else
    log "WARNING: host is $HOST_ARCH, target is $TARGET_ARCH. Build zstandard (and deepspeed, if in EXTRAS) on a $TARGET_ARCH machine and copy the wheels into $BUNDLE/wheelhouse"
  fi

  log "build axolotl wheel"
  (cd "$BUNDLE/axolotl" && uv build --wheel --python "$PY_MINOR" --out-dir "$BUNDLE/wheelhouse")

  if [ "$WITH_CCE" = 1 ]; then
    log "build cut-cross-entropy wheel"
    uvx --python "$PY_MINOR" pip wheel --no-deps -w "$BUNDLE/wheelhouse" "$CCE_REQ"
  fi

  log "check wheelhouse"
  local missing=0 name norm
  while read -r name; do
    norm=$(echo "$name" | tr 'A-Z.-' 'a-z__')
    ls "$BUNDLE/wheelhouse" | tr 'A-Z.-' 'a-z__' | grep -q "^${norm}-" || { echo "MISSING: $name"; missing=1; }
  done < <(grep '==' "$BUNDLE/requirements.target.txt" | cut -d= -f1 | sed 's/\[.*//')
  ls "$BUNDLE"/wheelhouse/axolotl-*.whl >/dev/null
  if [ "$WITH_CCE" = 1 ]; then ls "$BUNDLE"/wheelhouse/cut_cross_entropy-*.whl >/dev/null; fi
  [ $missing = 0 ] || log "WARNING: wheels are missing (see above); the target install will fail until they are added"
}

step_staging_env() {
  log "staging environment at $STAGING_VENV"
  local whl; whl=$(ls "$BUNDLE"/wheelhouse/axolotl-*.whl | head -1)
  local extra_whl=()
  if [ "$WITH_CCE" = 1 ]; then extra_whl=("$BUNDLE"/wheelhouse/cut_cross_entropy-*.whl); fi
  if [ "$HOST_ARCH" = "$TARGET_ARCH" ]; then
    uv venv --clear --python "$PY_MINOR" "$STAGING_VENV"
    uv pip sync --python "$STAGING_VENV/bin/python" --offline --no-index --no-build --find-links "$BUNDLE/wheelhouse" "$BUNDLE/requirements.target.txt"
    uv pip install --python "$STAGING_VENV/bin/python" --offline --no-index --no-build --no-deps "$whl" "${extra_whl[@]}"
  else
    uv venv --clear --python 3.12 "$STAGING_VENV"
    uv pip install --python "$STAGING_VENV/bin/python" --torch-backend "$TORCH_BACKEND" "$whl"
  fi
}

step_hf_cache() {
  log "HF cache at $HF_HOME"
  local hf="$STAGING_VENV/bin/hf"
  [ -x "$hf" ] || die "run the staging_env step first"
  local model datasets
  model=$("$STAGING_VENV/bin/python" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["base_model"])' "$CONFIG")
  datasets=$("$STAGING_VENV/bin/python" -c 'import sys,yaml; print("\n".join(d["path"] for d in yaml.safe_load(open(sys.argv[1]))["datasets"] if "/" in d["path"]))' "$CONFIG")
  "$hf" download "$model" --exclude 'original/*'
  local d; for d in $datasets; do "$hf" download "$d" --repo-type dataset; done

  local k
  for k in $KERNELS; do
    case $k in
      fa2) "$hf" download kernels-community/flash-attn2 --repo-type kernel --revision v3 \
             --include "build/torch-stable-abi210-${TORCH_BACKEND}-${TARGET_ARCH}-linux/*" ;;
      fa3) [ "$TARGET_ARCH" = x86_64 ] || die "flash-attn3 has no $TORCH_BACKEND build for $TARGET_ARCH"
           "$hf" download kernels-community/flash-attn3 --repo-type kernel --revision v1 \
             --include "build/torch-stable-abi29-${TORCH_BACKEND}-x86_64-linux/*" ;;
      fa4) "$hf" download kernels-community/flash-attn4 --repo-type kernel --revision v0 \
             --include 'build/torch-cuda/*' ;;
      *) die "unknown kernel '$k' (fa2 fa3 fa4)";;
    esac
  done
  ls "$HF_HOME/hub"
}

step_preprocess() {
  log "write config.yaml"
  grep -v '^dataset_prepared_path:' "$CONFIG" > "$BUNDLE/config.yaml"
  echo 'dataset_prepared_path: /opt/axolotl/bundle/prepared' >> "$BUNDLE/config.yaml"
  [ "$PREPROCESS" = 1 ] || return 0
  log "preprocess on staging"
  (cd "$BUNDLE" && DO_NOT_TRACK=1 "$STAGING_VENV/bin/axolotl" preprocess config.yaml --dataset-prepared-path "$BUNDLE/prepared")
  ls "$BUNDLE/prepared"
}

step_manifest() {
  log "clean up and write manifest"
  git -C "$BUNDLE/axolotl" clean -fdXq
  rm -rf "$BUNDLE/hf/xet" "$BUNDLE/requirements.binary.txt"
  (cd "$BUNDLE" && {
    echo "created:        $(date -u +%FT%TZ)"
    echo "axolotl_commit: $(git -C axolotl rev-parse HEAD)"
    echo "uv:             uv $UV_VERSION"
    echo "python:         cpython-$PY_VERSION, python-build-standalone $PBS_RELEASE"
    echo "python_file:    $(ls "python-mirror/$PBS_RELEASE/")"
    echo "target_arch:    $TARGET_ARCH"
    echo "torch_backend:  $TORCH_BACKEND"
    echo "extras:         $EXTRAS$([ "$WITH_CCE" = 1 ] && echo ' + cut-cross-entropy')"
    echo "kernels:        $KERNELS"
    echo "wheel_count:    $(find wheelhouse -name '*.whl' | wc -l)"
    echo "hf_refs:"
    find hf/hub -path '*/refs/*' -type f | sort | while read -r r; do
      repo=${r#hf/hub/}; repo=${repo%%/*}
      echo "  $repo ${r#*/refs/} $(cat "$r")"
    done
  } > MANIFEST.txt && cat MANIFEST.txt)

  log "checksums"
  (cd "$BUNDLE" \
    && find . -type l -printf '%p -> %l\n' | sort > SYMLINKS.txt \
    && { [ -z "$(find hf -xtype l)" ] || die "dangling symlinks under hf/"; } \
    && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
}

step_archive() {
  local out; out=$(dirname "$BUNDLE")
  local name; name=$(basename "$BUNDLE")
  log "archive $out/$name.tar"
  (cd "$out" && tar -cf "$name.tar" "$name" && sha256sum "$name.tar" > "$name.tar.sha256")
  if [ -n "$SPLIT_SIZE" ]; then
    (cd "$out" && split -b "$SPLIT_SIZE" -d -a 3 "$name.tar" "$name.tar.part-" && sha256sum "$name.tar.part-"* > PARTS.sha256)
  fi
  du -sh "$out/$name.tar"
}

ALL_STEPS="prepare clone uv python resolve wheels staging_env hf_cache preprocess manifest archive"
if [ $# -gt 0 ]; then
  STEPS="prepare clone $*"
else
  STEPS=$ALL_STEPS
fi
for s in $STEPS; do
  declare -F "step_$s" >/dev/null || die "unknown step '$s'. Steps: $ALL_STEPS"
done
for s in $STEPS; do "step_$s"; done
log "done: $BUNDLE"
