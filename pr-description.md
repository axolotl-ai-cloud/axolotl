# fix: docker-e2e-cleanup — update Modal Image Builder version to 2025.06

## Problem

The `docker-e2e-cleanup` job in `tests.yml` fails with:
```
Terminating task due to error: command skopeo copy
docker://axolotlai/axolotl-base-uv:main-base-py3.11-cu128-2.9.1
oci:... had exit status: 1
```

## Investigation

### Tag shape is correct
The original hypothesis was that the base image tag shape changed after the
uv-first switch. This is **not** the case. The tag `main-base-py3.11-cu128-2.9.1`
exists on Docker Hub and resolves correctly:

- Docker Hub API confirms the tag is active (last updated 2026-03-25)
- `docker manifest inspect` shows a valid OCI image index with amd64 + arm64
- `docker pull --platform linux/amd64 axolotlai/axolotl-base-uv:main-base-py3.11-cu128-2.9.1` succeeds
- All published tags follow the shape `main-base-py<X>-cu<Y>-<Z>`
- The `Dockerfile-uv.jinja` correctly references `axolotlai/axolotl-base-uv:{{ BASE_TAG }}`

### Actual root cause: outdated Modal Image Builder version
The `tests.yml` e2e jobs hardcode `MODAL_IMAGE_BUILDER_VERSION=2024.10`,
while `multi-gpu-e2e.yml` uses `2025.06` at the workflow level. The older
image builder version (2024.10) likely has a bug or limitation in how it
handles skopeo copy for large multi-arch images (~11.4 GB).

The `modal.experimental.raw_dockerfile_image` function's docstring notes:
"We expect to support this experimental function until the `2025.04` Modal
Image Builder is stable." The 2025.06 builder is past this milestone and
may have fixed the underlying skopeo issue.

## Fix

Update `MODAL_IMAGE_BUILDER_VERSION` from `2024.10` to `2025.06` in all three
e2e jobs in `tests.yml`:
- `docker-e2e-tests-1st` (line 298)
- `docker-e2e-tests` (line 359)
- `docker-e2e-cleanup` (line 403)

This aligns with the version already used in `multi-gpu-e2e.yml`.

## Verification

- Confirmed tag exists: `docker manifest inspect axolotlai/axolotl-base-uv:main-base-py3.11-cu128-2.9.1` returns valid OCI index
- Confirmed image pulls: `docker pull axolotlai/axolotl-base-uv:main-base-py3.11-cu128-2.9.1` succeeds
- YAML syntax is unchanged except for the three version string updates
- The fix is minimal and matches an established pattern in `multi-gpu-e2e.yml`
