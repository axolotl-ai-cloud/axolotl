#!/bin/bash
set -euo pipefail

mkdir -p "${HF_HOME:?}/hub/"
cache_tmp=$(mktemp -d)
trap 'rm -rf "$cache_tmp"' EXIT
archive="$cache_tmp/hf-cache.tar.zst"
started=$SECONDS

# Resume interrupted transfers before extracting; never erase the shared Hub cache.
for attempt in 1 2 3; do
  remaining=$((900 - SECONDS + started))
  if (( remaining <= 0 )); then
    break
  fi
  if curl --silent --show-error --fail --location \
    --connect-timeout 30 --max-time "$remaining" \
    --speed-limit 1048576 --speed-time 60 \
    --continue-at - --output "$archive" \
    https://axolotl-ci.b-cdn.net/hf-cache.tar.zst; then
    tar -xpf "$archive" -C "${HF_HOME}/hub/" --use-compress-program unzstd --strip-components=1
    echo "HF cache extracted successfully"
    exit 0
  fi
  echo "HF cache transfer attempt $attempt failed; retrying from the partial download" >&2
done

echo "HF cache download failed (3 attempts / 15-minute download budget)" >&2
exit 1
