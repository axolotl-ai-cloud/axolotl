"""Topology-aware default for ``protrain_own_lora_grad_sync``.

The detector parses ``nvidia-smi topo -m`` to determine whether all visible
GPU pairs are NVLinked. Header-row handling and CUDA_VISIBLE_DEVICES masking
are the two tricky cases that have to be tested with mocked output (Codex
review found that uncolorized output included the header row and broke
detection).
"""

from __future__ import annotations

import os

import pytest

# Sample outputs captured from real ``nvidia-smi topo -m`` runs.
# `_COLORIZED` is what the pod returns when a TTY is attached (header has
# ANSI underline escape codes that incidentally protect the parser).
# `_PLAIN` is what CI / non-TTY contexts return — no ANSI codes, and the
# header row starts with the same `GPU` prefix as data rows.

_COLORIZED_2GPU_NV12 = """
\t\x1b[4mGPU0\tGPU1\tNIC0\tNIC1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m
GPU0\t X \tNV12\tPXB\tPXB\t0-63\t0\t\tN/A
GPU1\tNV12\t X \tSYS\tSYS\t64-127\t1\t\tN/A
NIC0\tPXB\tSYS\t X \tPXB\t\t\t\t
NIC1\tPXB\tSYS\tPXB\t X \t\t\t\t

Legend:

  X    = Self
"""

_PLAIN_2GPU_NV12 = """
\tGPU0\tGPU1\tNIC0\tNIC1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tNV12\tPXB\tPXB\t0-63\t0\t\tN/A
GPU1\tNV12\t X \tSYS\tSYS\t64-127\t1\t\tN/A
NIC0\tPXB\tSYS\t X \tPXB\t\t\t\t
NIC1\tPXB\tSYS\tPXB\t X \t\t\t\t

Legend:

  X    = Self
"""

_PLAIN_2GPU_PCIE = """
\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tPHB\t0-63\t0\t\tN/A
GPU1\tPHB\t X \t64-127\t1\t\tN/A

Legend:

  X    = Self
"""

_PLAIN_4GPU_MIXED = """
\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tNV12\tPHB\tPHB\t0-63\t0\t\tN/A
GPU1\tNV12\t X \tPHB\tPHB\t0-63\t0\t\tN/A
GPU2\tPHB\tPHB\t X \tNV12\t64-127\t1\t\tN/A
GPU3\tPHB\tPHB\tNV12\t X \t64-127\t1\t\tN/A
"""


def _import_detect_with_fresh_cache():
    """Import the detector and clear its lru_cache so each test sees a fresh probe."""
    from axolotl.integrations.protrain.plugin import _detect_nvlink_topology

    _detect_nvlink_topology.cache_clear()
    return _detect_nvlink_topology


def _patched_subprocess_run(stdout: str, returncode: int = 0):
    """Build a `subprocess.run` mock that returns `stdout`/`returncode`."""

    class _Proc:
        def __init__(self) -> None:
            self.stdout = stdout
            self.returncode = returncode

    def _run(cmd, *args, **kwargs):  # noqa: ARG001
        assert cmd[:3] == ["/usr/bin/nvidia-smi", "topo", "-m"]
        assert os.path.isabs(cmd[0])
        return _Proc()

    return _run


@pytest.fixture(autouse=True)
def _mock_nvidia_smi_binary(monkeypatch):
    """Pretend `nvidia-smi` resolves to an absolute binary path."""
    import shutil

    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    yield


@pytest.fixture
def _mock_2_visible_gpus(monkeypatch):
    """Pretend torch sees 2 visible CUDA devices (identity-mapped to physical 0/1)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    yield


@pytest.fixture
def _mock_4_visible_gpus(monkeypatch):
    """Pretend torch sees 4 visible CUDA devices (identity-mapped to physical 0..3)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    yield


def test_colorized_nv12_classified_as_nvlink(_mock_2_visible_gpus, monkeypatch):
    """The TTY-colorized variant (with ANSI escapes in the header) → True."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", _patched_subprocess_run(_COLORIZED_2GPU_NV12)
    )
    detect = _import_detect_with_fresh_cache()
    assert detect() is True


def test_plain_nv12_classified_as_nvlink(_mock_2_visible_gpus, monkeypatch):
    """The uncolorized variant (no ANSI escapes) → must also return True.

    Codex review #2: the prior parser used `lstrip().startswith("GPU")` to
    detect data rows, which included the plain header line and shifted the
    physical-index parsing, mis-classifying NVLink as PCIe.
    """
    import subprocess

    monkeypatch.setattr(subprocess, "run", _patched_subprocess_run(_PLAIN_2GPU_NV12))
    detect = _import_detect_with_fresh_cache()
    assert detect() is True


def test_plain_pcie_classified_as_non_nvlink(_mock_2_visible_gpus, monkeypatch):
    """PHB topology (PCIe via host bridge) → False."""
    import subprocess

    monkeypatch.setattr(subprocess, "run", _patched_subprocess_run(_PLAIN_2GPU_PCIE))
    detect = _import_detect_with_fresh_cache()
    assert detect() is False


def test_mixed_topology_classified_as_non_nvlink(_mock_4_visible_gpus, monkeypatch):
    """Heterogeneous topology (NVLink within pair, PCIe across pair) → False.

    Conservative: any pair without an active NVLink reduces classification
    to PCIe-class so Path B stays enabled for the slowest interconnect.
    """
    import subprocess

    monkeypatch.setattr(subprocess, "run", _patched_subprocess_run(_PLAIN_4GPU_MIXED))
    detect = _import_detect_with_fresh_cache()
    assert detect() is False


def test_nvidia_smi_failure_returns_false(_mock_2_visible_gpus, monkeypatch):
    """`nvidia-smi topo -m` returning non-zero → safe default False."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", _patched_subprocess_run("error", returncode=1)
    )
    detect = _import_detect_with_fresh_cache()
    assert detect() is False


def test_missing_nvidia_smi_returns_false(_mock_2_visible_gpus, monkeypatch):
    """No resolved `nvidia-smi` binary means safe default False without spawning."""
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess.run should not be called"),
    )

    detect = _import_detect_with_fresh_cache()
    assert detect() is False


def test_nvidia_smi_spawn_uses_absolute_which_path(
    _mock_2_visible_gpus, monkeypatch
):
    """The subprocess argv starts with the absolute path returned by shutil.which."""
    import shutil
    import subprocess

    resolved = "/opt/nvidia/bin/nvidia-smi"
    calls: list[list[str]] = []

    class _Proc:
        stdout = _PLAIN_2GPU_NV12
        returncode = 0

    def _run(cmd, *args, **kwargs):  # noqa: ARG001
        calls.append(cmd)
        return _Proc()

    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: resolved if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr(subprocess, "run", _run)

    detect = _import_detect_with_fresh_cache()
    assert detect() is True
    assert calls == [[resolved, "topo", "-m"]]


def test_single_gpu_returns_false(monkeypatch):
    """1 visible GPU → False (no pairs to check; default-True semantics preserved)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    detect = _import_detect_with_fresh_cache()
    assert detect() is False


def test_cuda_visible_devices_masking_picks_correct_rows(monkeypatch):
    """`CUDA_VISIBLE_DEVICES=2,3` on a 4-GPU host: torch sees 2 devices →
    physical indices 2 and 3 in the topo matrix. The pair-cells inspected
    must be (GPU2,GPU3), not (GPU0,GPU1).
    """
    import subprocess

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)  # masked view
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(subprocess, "run", _patched_subprocess_run(_PLAIN_4GPU_MIXED))

    detect = _import_detect_with_fresh_cache()
    # GPU2↔GPU3 is NV12 in _PLAIN_4GPU_MIXED, so True.
    assert detect() is True


def test_cuda_visible_devices_masks_to_pcie_pair(monkeypatch):
    """`CUDA_VISIBLE_DEVICES=0,2` on a 4-GPU host: pair-cells (GPU0,GPU2) is
    PHB in `_PLAIN_4GPU_MIXED` → False.
    """
    import subprocess

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    monkeypatch.setattr(subprocess, "run", _patched_subprocess_run(_PLAIN_4GPU_MIXED))

    detect = _import_detect_with_fresh_cache()
    assert detect() is False


def test_cuda_visible_devices_uuid_mask_returns_false(monkeypatch):
    """UUID masks cannot be mapped to nvidia-smi rows, so do not assume GPU0/GPU1."""
    import subprocess

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,GPU-def")
    monkeypatch.setattr(subprocess, "run", _patched_subprocess_run(_PLAIN_2GPU_NV12))

    detect = _import_detect_with_fresh_cache()
    assert detect() is False


def test_cuda_visible_devices_mixed_numeric_and_uuid_returns_false(monkeypatch):
    """Partially numeric masks are ambiguous and must not fall back to identity."""
    import subprocess

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,GPU-def")
    monkeypatch.setattr(subprocess, "run", _patched_subprocess_run(_PLAIN_2GPU_NV12))

    detect = _import_detect_with_fresh_cache()
    assert detect() is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
