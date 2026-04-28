# fix: docker-e2e-cleanup — update base image tag for uv switch

## Summary

Investigation of the `docker-e2e-cleanup` CI failure (run #24761481292, Apr 22) revealed that the original hypothesis — a broken tag shape post-uv switch — is **incorrect**. The tag `main-base-py3.11-cu128-2.9.1` exists on Docker Hub and resolves correctly. The actual failure was a **transient IPv6 network issue** on the Modal runner.

## Root Cause

```
Error: initializing source docker://axolotlai/axolotl-base-uv:main-base-py3.11-cu128-2.9.1:
  pinging container registry registry-1.docker.io: Get "https://registry-1.docker.io/v2/":
  dial tcp [2600:1f18:2148:bc00:154:b6de:c219:cb4e]:443: connect: network is unreachable
```

Modal's internal `skopeo copy` (used during image build) attempted to reach Docker Hub via IPv6, which was unreachable from the Modal runner's network. This is a transient infrastructure issue, not a missing tag.

## Evidence

- `docker manifest inspect axolotlai/axolotl-base-uv:main-base-py3.11-cu128-2.9.1` — succeeds, returns valid multi-arch manifest (amd64 + arm64)
- All tag combinations used by the test matrix exist:
  - `main-base-py3.11-cu128-2.9.1` — EXISTS
  - `main-base-py3.12-cu128-2.9.1` — EXISTS
  - `main-base-py3.11-cu130-2.9.1` — EXISTS
  - `main-base-py3.12-cu130-2.9.1` — EXISTS
  - `main-base-py3.11-cu128-2.10.0` — EXISTS
  - `main-base-py3.12-cu128-2.10.0` — EXISTS
  - `main-base-py3.12-cu130-2.10.0` — EXISTS
  - `main-base-py3.11-cu130-2.10.0` — MISSING (but not referenced by any e2e job)
- The tag shape `main-base-py<ver>-cu<ver>-<torch>` is published by `base.yml` line 235 and matches the reference in `tests.yml`
- The same cleanup matrix combination (cuda=128, py=3.11, torch=2.9.1) succeeded in the docker-e2e-tests job of the same run — confirming the tag is accessible

## Fix

Added the missing `E2E_DOCKERFILE` env var to the `docker-e2e-cleanup` job. All other e2e jobs (`docker-e2e-tests`, `docker-e2e-tests-1st`) explicitly set this variable, but the cleanup job relied on the default in `single_gpu.py`. While behavior is unchanged (both resolve to `Dockerfile-uv.jinja`), making it explicit ensures consistency and prevents breakage if the default ever changes.

## Related

- Failing run: https://github.com/axolotl-ai-cloud/axolotl/actions/runs/24761481292
- Base image build workflow: `.github/workflows/base.yml` (tag shape at line 235)
- Test workflow: `.github/workflows/tests.yml` (BASE_TAG at lines 293, 354, 398)
