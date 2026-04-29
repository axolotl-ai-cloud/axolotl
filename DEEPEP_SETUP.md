# DeepEP Setup — Validated Install Steps

End-to-end install for running DeepEP intranode kernels on Ampere/Hopper + NVLink. Complements [`DEEP_EP.md`](DEEP_EP.md), which describes the design; this doc is the runnable user how-to.

These steps were validated on a 2× A100-SXM4-80GB box (CUDA 13.0, driver 550.127.05, PyTorch 2.10+cu130). Hopper builds drop the Ampere-specific flags noted below.

---

## 1. Hardware support matrix

| Path | GPU | Interconnect | Kernels you get |
|---|---|---|---|
| **Intranode-only (this doc)** | Ampere (sm_80, A100) or Hopper (sm_90, H100/H200) | NVLink, all-pairs | Normal `dispatch` / `combine` |
| Intranode + internode + LL | Hopper only | NVLink + IBGDA-capable IB | Normal + low-latency |

DeepEP rejects PCIe-only A100s in `check_nvlink_connections()`. Verify with `nvidia-smi nvlink --status`: every link line should show `25 GB/s` (A100) or `26.562 GB/s` (H100). If the command returns nothing, you have PCIe-only GPUs and DeepEP won't run.

Low-latency (LL) kernels and any internode (multi-node) path require Hopper + IBGDA-capable InfiniBand. The Ampere path covers training/prefill on a single node; LL decode is out of scope until you have Hopper hardware.

---

## 2. Prereqs

| Component | Verified version |
|---|---|
| OS | Ubuntu 22.04 |
| Driver | 550.127.05 (also see §6 for forward-compat) |
| CUDA toolkit | 13.0 |
| gcc | 11.4 |
| PyTorch | 2.10.0+cu130 |
| Python | 3.12 |

```bash
nvcc --version            # CUDA 13.x for cu130 torch
gcc --version             # 11.x recommended
nvidia-smi nvlink --status   # all-pairs NVLink required
python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_capability(0))"
```

---

## 3. NVSHMEM

**Skip on Ampere intranode** — the intranode kernels use CUDA IPC over NVLink and don't need NVSHMEM. You'll pass `DISABLE_NVSHMEM=1` to the DeepEP build in §4.

For Hopper (or future internode/LL builds): NVIDIA ships NVSHMEM as a pip wheel; no source build needed.

```bash
uv pip install nvidia-nvshmem-cu13   # or -cu12 for CUDA 12.x torch
```

If you have torch 2.10+cu130 in the venv, `nvidia-nvshmem-cu13` 3.4.5 is already installed as a transitive dep — confirm with `uv pip show nvidia-nvshmem-cu13`. The wheel installs to `<venv>/lib/python3.12/site-packages/nvidia/nvshmem/{include,lib}/`. DeepEP's `setup.py` auto-detects it via `importlib.util.find_spec("nvidia.nvshmem")` — no need to set `NVSHMEM_DIR`.

---

## 4. DeepEP

The PyPI `deep-ep==1.0.0` sdist is hardcoded to Hopper (sm_90) and has the wrong NVSHMEM link layout — **don't use it.** Build from GitHub HEAD.

The HEAD setup.py supports Ampere via `DISABLE_SM90_FEATURES=1`, but it asserts `disable_nvshmem=True` in that mode and there's no env var to force it. Two small upstream patches are required for an Ampere intranode-only build (one in `setup.py`, one in `csrc/deep_ep.cpp`).

```bash
cd /workspace
git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
```

### 4.1 Patch `setup.py` — `DISABLE_NVSHMEM=1` env-var bypass

The default flow auto-detects an installed `nvidia.nvshmem` wheel and refuses to build without it. We add a one-line short-circuit so `DISABLE_NVSHMEM=1` in the env forces `disable_nvshmem=True` regardless. Apply this near the top of the `if __name__ == '__main__':` block:

```python
# In setup.py, replace the existing nvshmem auto-detect block with this pattern:
if int(os.getenv('DISABLE_NVSHMEM', '0')):
    disable_nvshmem = True
    nvshmem_dir = None
elif nvshmem_dir is None:
    try:
        nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
        # ... existing auto-detect ...
```

### 4.2 Patch `csrc/deep_ep.cpp` — gate the three `mask_buffer` methods

Upstream's `DISABLE_NVSHMEM` gating misses three methods near line 1825 that unconditionally call into `internode_ll::*`. Without the gate, the link step fails with:

```
ImportError: deep_ep_cpp.cpython-...-x86_64-linux-gnu.so: undefined symbol:
_ZN7deep_ep12internode_ll17query_mask_bufferEPiiS1_P11CUstream_st
```

Wrap the three method bodies (`low_latency_update_mask_buffer`, `low_latency_query_mask_buffer`, `low_latency_clean_mask_buffer`) in `#ifndef DISABLE_NVSHMEM ... #else EP_HOST_ASSERT(false ...) #endif`, matching the pattern already used at `get_next_low_latency_combine_buffer` just above.

### 4.3 Build

```bash
DISABLE_NVSHMEM=1 \
DISABLE_SM90_FEATURES=1 \
TORCH_CUDA_ARCH_LIST=8.0 \
MAX_JOBS=16 \
  uv pip install --no-build-isolation .
```

| Env var | Why |
|---|---|
| `DISABLE_NVSHMEM=1` | Skip NVSHMEM linking; intranode kernels don't need it. |
| `DISABLE_SM90_FEATURES=1` | Disable Hopper-only code (FP8, TMA, async launch). Required for sm_80. |
| `TORCH_CUDA_ARCH_LIST=8.0` | Build sm_80 PTX/SASS only. The default `9.0` would emit Hopper code that won't run on A100. |
| `MAX_JOBS=16` | Parallel compile. Adjust to your core count. |
| `--no-build-isolation` | Reuse the venv's installed `torch` for the build (faster; avoids version drift). |

Build takes ~30 s on a 16-core box. Output: `<venv>/lib/python3.12/site-packages/deep_ep_cpp.cpython-3xx-x86_64-linux-gnu.so` plus the `deep_ep` Python package.

### 4.4 Hopper build (full-fat)

```bash
TORCH_CUDA_ARCH_LIST=9.0 uv pip install --no-build-isolation .
```

Drop both `DISABLE_NVSHMEM=1` and `DISABLE_SM90_FEATURES=1`. The two patches in §4.1 / §4.2 are not needed on Hopper because NVSHMEM is enabled (the gate condition is never hit).

---

## 5. Sanity check

```bash
python -c "import deep_ep; print(deep_ep.Buffer)"
# <class 'deep_ep.buffer.Buffer'>

# 2-GPU intranode forward (run from the repo root):
torchrun --nproc-per-node=2 bench_deep_ep.py --mode deep_ep --seq-len 1024
```

The repro script at `bench_deep_ep.py` is the validation harness (Stage 1 of the integration plan). Modes: `single`, `eager_dist`, `deep_ep`, `deep_ep_scattermoe`. See its module docstring for full usage.

---

## 6. Driver / CUDA forward-compat (driver < 580)

If your driver is older than 580 (CUDA 12.4 era and earlier), CUDA 13 binaries fail at runtime with `cudaErrorNoKernelImageForDevice` or `CUBLAS_STATUS_INVALID_VALUE` on basic GEMMs. NVIDIA ships forward-compat libs at `/usr/local/cuda-13.0/compat/libcuda.so.580.x` plus a CUDA 13 cublas at `<venv>/lib/python3.12/site-packages/nvidia/cu13/lib/`. Both must be on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/workspace/axolotl-venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

Verify the fix with a basic GEMM:

```bash
python -c "
import torch, torch.nn.functional as F
x = torch.randn(64, 32, device='cuda', dtype=torch.bfloat16)
w = torch.randn(16, 32, device='cuda', dtype=torch.bfloat16)
print(F.linear(x, w).shape)
"
```

Driver ≥ 580 boxes don't need this.

---

## 7. Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `undefined symbol: _ZN7deep_ep12internode_ll17query_mask_bufferEPiiS1_...` | §4.2 patch missing on an Ampere build. | Apply the `#ifndef DISABLE_NVSHMEM` gate to the three `low_latency_*_mask_buffer` methods. |
| `ImportError: libnvshmem_host.so.3: cannot open shared object file` | Build linked against NVSHMEM but the `.so` isn't on `LD_LIBRARY_PATH`. | Either rebuild with `DISABLE_NVSHMEM=1` (Ampere intranode) or `export LD_LIBRARY_PATH=$(python -c "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])")/lib:$LD_LIBRARY_PATH` (Hopper). |
| `nvcc fatal: Unsupported gpu architecture 'compute_90'` | `DISABLE_SM90_FEATURES=1` unset on an Ampere build. | Set both `DISABLE_SM90_FEATURES=1` and `TORCH_CUDA_ARCH_LIST=8.0`. |
| `cudaErrorNoKernelImageForDevice` or `CUBLAS_STATUS_INVALID_VALUE` on basic GEMM | Driver < 580 with CUDA 13 toolkit; missing forward-compat lib on `LD_LIBRARY_PATH`. | See §6. |
| `RuntimeError: NVLink connection check failed` | PCIe-only A100, or NVLink disabled. | Use SXM4 hardware. Verify `nvidia-smi nvlink --status` shows all links active. |
| Build hangs at `nvcc … intranode.cu` | Default `MAX_JOBS` on a small box can OOM. | Lower `MAX_JOBS` (try 4) or add swap. |
| `Skipping import of cpp extensions due to incompatible torch version. Please upgrade to torch >= 2.11.0` | Harmless warning from a downstream package that doesn't support torch 2.10. | Ignore — it's not from DeepEP. |

---

## 8. What this does *not* cover

- **Multi-node / InfiniBand setup.** Hopper + IBGDA + NVSHMEM transports. `nvshmem_transport_ibgda.so.3` ships in the wheel; tuning `NVSHMEM_IB_*` env vars is cluster-specific. See DeepEP's main README.
- **UCCL-EP.** Drop-in alternative for AWS EFA / Broadcom / AMD MI300x. Separate install — see [`DEEP_EP.md §6.2`](DEEP_EP.md).
- **Low-latency kernels on Ampere.** Not supported — pure-RDMA Hopper-only.
- **CUDA 12.x.** This doc validates CUDA 13.0 / cu130 torch only. CUDA 12.x should work with `nvidia-nvshmem-cu12` substituted but is untested here.

---

## 9. Upstream PR opportunities

The two patches in §4 are local workarounds. Worth contributing back to deepseek-ai/DeepEP:

1. `setup.py`: honor a `DISABLE_NVSHMEM=1` env var so wheel-install of NVSHMEM doesn't force `disable_nvshmem=False`. Three-line change.
2. `csrc/deep_ep.cpp`: wrap `low_latency_{update,query,clean}_mask_buffer` bodies in `#ifndef DISABLE_NVSHMEM`, matching the pattern at `get_next_low_latency_combine_buffer`. ~12-line change.
