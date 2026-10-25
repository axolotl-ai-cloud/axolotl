#!/bin/bash
set -euo pipefail

url=https://axolotl-ci.b-cdn.net/hf-cache.tar.zst
mkdir -p "${HF_HOME:?}/hub/"

# The hub cache is a persistent Modal volume, so skip the multi-GB transfer when it already holds
# this archive version. The marker is written only after a full extract.
marker="${HF_HOME}/hub/.hf-cache-archive-version"
version=$(
  curl --silent --show-error --fail --location --head --connect-timeout 30 --max-time 60 "$url" |
    tr -d '\r' |
    awk 'tolower($1) == "etag:" {etag = $2} tolower($1) == "last-modified:" {sub(/^[^:]*: */, ""); lm = $0}
         END {print (etag != "" ? etag : lm)}'
) || version=""
if [[ -n "$version" && -f "$marker" && "$(cat "$marker")" == "$version" ]]; then
  echo "HF cache already at $version, skipping download"
  exit 0
fi

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
    "$url"; then
    tar -xpf "$archive" -C "${HF_HOME}/hub/" --use-compress-program unzstd --strip-components=1
    if [[ -n "$version" ]]; then
      printf '%s' "$version" > "$marker"
    fi
    echo "HF cache extracted successfully"
    exit 0
  fi
  echo "HF cache transfer attempt $attempt failed; retrying from the partial download" >&2
done

echo "HF cache download failed (3 attempts / 15-minute download budget)" >&2
exit 1
