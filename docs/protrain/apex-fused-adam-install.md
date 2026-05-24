# Apex FusedAdam install guide

This page documents how to install NVIDIA Apex so that the `adamw_apex_fused`
optimizer can be used from an axolotl config. The optimizer name is already
accepted by the axolotl validator (see commit `539eff293`, which whitelists
`adamw_apex_fused` in `_SUPPORTED_OPTIMIZERS`). What's not enforced by the
validator is that the Apex package is importable at runtime, which in turn
requires a CUDA-aligned build environment.

## Why use Apex FusedAdam

Apex's `FusedAdam` fuses the element-wise Adam update kernels into a single
CUDA launch per parameter group. On large full fine-tunes where the optimizer
step is on the critical path, this is typically ~10-20% faster than
`adamw_torch`. For LoRA and QLoRA runs the gain is usually negligible, since
the optimizer touches far fewer parameters and the step is not the bottleneck.

If you are unsure whether your run is optimizer-bound, profile a few steps and
compare the time spent in `optimizer.step()` against the forward/backward
passes. If the step is under 5% of the iteration time, switching optimizers is
not worth the install complexity.

## CUDA alignment requirement

Apex builds CUDA extensions at install time. The system CUDA toolkit (the one
`nvcc` resolves to) must match the CUDA version that the installed PyTorch
wheel was compiled against. The match must be exact at the `major.minor`
level: `12.1` and `12.4` are not interchangeable.

If the versions don't match, one of two failure modes occurs:

- **Compile-time** — the install fails with errors like
  `error: identifier "..." is undefined` or
  `error: too many arguments in function call` against headers from the
  installed toolkit.
- **Runtime** — the install succeeds, but importing or using Apex raises
  `RuntimeError: CUDA error: invalid device function` on the first kernel
  launch.

Neither failure is recoverable without a clean rebuild against an aligned
toolkit.

## Check alignment

Run both commands and confirm the reported versions match at `major.minor`.

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version | grep release
```

Example of an aligned environment:

```text
13.0
Cuda compilation tools, release 13.0, V13.0.48
```

Example of a mismatched environment (will fail Apex install):

```text
13.0
Cuda compilation tools, release 13.2, V13.2.93
```

## Install steps

Once the versions match, build Apex from source. Do not install from PyPI;
the PyPI `apex` package is unrelated to NVIDIA's project.

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-build-isolation \
    --no-cache-dir --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
```

Both `--cpp_ext` and `--cuda_ext` are required for `FusedAdam`. Omitting
`--cuda_ext` will produce a partial install that imports cleanly but raises
`ModuleNotFoundError: No module named 'amp_C'` the first time an Apex
optimizer is constructed.

The build is single-threaded by default and can take 15-30 minutes on a
typical workstation. If `MAX_JOBS` is set in the environment, the C++
compile phase will parallelize, but the CUDA phase will not.

## Fixing a mismatch

You have two options. Pick whichever is less disruptive to the rest of your
environment.

### Option A — install a CUDA toolkit matching the torch wheel

If your torch wheel is `cu130`, install CUDA 13.0:

```bash
conda install cuda -c nvidia/label/cuda-13.0.0
```

Or system-level, by pinning the apt package to the matching version. After
installing, confirm `nvcc --version` reports the expected release before
attempting the Apex build.

### Option B — install a torch wheel matching the system CUDA

If your system CUDA is 13.0, install a `cu130` torch wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Note that not every torch release publishes wheels for every CUDA minor
version. Check the PyTorch download index before committing to this path.

## Validate the install

After the build finishes, confirm the optimizer can be imported:

```python
from apex.optimizers import FusedAdam
```

If this raises `ModuleNotFoundError` or `ImportError`, the install did not
produce the CUDA extension; rebuild with both `--cpp_ext` and `--cuda_ext`
as shown above.

For a slightly stronger smoke test, construct the optimizer against a real
tensor:

```python
import torch
from apex.optimizers import FusedAdam

p = torch.zeros(4, device="cuda", requires_grad=True)
opt = FusedAdam([p], lr=1e-3)
p.grad = torch.ones_like(p)
opt.step()
```

A successful run means Apex is wired up correctly. A `RuntimeError: CUDA
error: invalid device function` here indicates a runtime CUDA mismatch
between Apex's compiled kernels and the active torch runtime.

## Use with axolotl

Once `from apex.optimizers import FusedAdam` succeeds, set the optimizer in
your YAML config:

```yaml
optimizer: adamw_apex_fused
```

The axolotl validator accepts this name (whitelisted in commit `539eff293`).
No other config changes are required; learning rate, betas, weight decay,
and epsilon are passed through from the standard Hugging Face training
arguments.

## Troubleshooting

### `ModuleNotFoundError: No module named 'amp_C'`

Apex was installed but the CUDA C extension was not built. Reinstall with
both build flags:

```bash
pip install -v --disable-pip-version-check --no-build-isolation \
    --no-cache-dir --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
```

### `RuntimeError: CUDA error: invalid device function`

Apex's compiled kernels target a different CUDA version than the one torch
is using at runtime. This happens after upgrading torch or the CUDA
toolkit without rebuilding Apex. Wipe the Apex install and rebuild against
the current torch + CUDA pair:

```bash
pip uninstall apex
cd apex
git clean -xdf
pip install -v --disable-pip-version-check --no-build-isolation \
    --no-cache-dir --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
```

### Apex install fails with header errors

The compiler is finding headers from a CUDA toolkit that doesn't match the
torch wheel. Re-check the alignment commands above and confirm `which nvcc`
points at the expected install. If a conda CUDA toolkit is shadowing a
system one, activate the appropriate environment before rebuilding.

### Native PyTorch alternative

On PyTorch 2.4 and newer, several Apex code paths have been upstreamed. The
built-in fused AdamW is a near-equivalent:

```python
torch.optim.AdamW(params, fused=True)
```

In most workloads it is at most 5% slower than Apex's `FusedAdam` and
avoids the build-time CUDA alignment requirement entirely. In axolotl, this
is exposed as `optimizer: adamw_torch_fused`. If the install complexity
above is not worth the marginal speedup, use the torch fused path instead.

## Notes for the axolotl validation environment

The protrain phase 2 PR was developed against torch 2.9.1 + cu130 on a
system reporting cu13.2, so the Apex install path was not exercised
end-to-end in this environment. The whitelist change in commit `539eff293`
is correctness-verified through unit tests against the validator, not
through a full train-time round-trip with `FusedAdam`.

Users running in a CUDA-aligned environment who hit install or runtime
issues should open a GitHub issue with the output of the two version-check
commands above plus the full pip log, so the validation path can be
exercised against a known-good toolchain.
