"""Fail collection when importing a conftest or test module initializes CUDA.

Once a process has a CUDA context it cannot fork safely, so a module-level device query breaks
fork-based tests and DataLoader fork workers for the rest of the session. Move such queries into
the test or a setup-time skip condition (``tests/cuda_gates.py``).

The guard wraps ``torch.cuda._lazy_init`` from the root conftest's import until collection
finishes. It attributes each call to the innermost ``<module>`` frame on the stack and adds a
collection error for callers outside ``ALLOWED_SITES``. Python device queries
(``get_device_capability``, ``current_device``, ...) call ``_lazy_init`` every time, so all of
them are seen. C++ callers such as ``torch.zeros(..., device="cuda")`` call it only on the first
initialization, so they go unseen once anything else has initialized CUDA.

Set ``AXOLOTL_ALLOW_CUDA_INIT_AT_COLLECTION=1`` to turn the guard off.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

OPT_OUT_ENV = "AXOLOTL_ALLOW_CUDA_INIT_AT_COLLECTION"

# Third-party imports that initialize CUDA. A call is allowed when any frame from the caller up to
# its innermost <module> frame is in one of these paths; drop an entry once its fix is pinned.
ALLOWED_SITES = (
    # until torchao runs has_triton() / is_sm_at_least_100() at call time instead of at import
    "torchao/prototype/mx_formats/kernels.py",
    # until bitsandbytes reads device capabilities lazily instead of when picking its library
    "bitsandbytes/cuda_specs.py",
    # until fla computes its device name/capability constants on first use
    "fla/utils/hardware.py",
    # until xformers' triton_splitk checks _is_triton_available() on first dispatch
    "xformers/",
)

_TORCH_DIR = str(Path(torch.__file__).parent) + os.sep


def _short(path: str, rootdir: Path) -> str:
    root = str(rootdir) + os.sep
    if path.startswith(root):
        return path[len(root) :]
    return path.rpartition("site-packages" + os.sep)[2]


class CudaInitGuard:
    """Records ``torch.cuda._lazy_init`` callers while installed; register it as a pytest plugin."""

    def __init__(self, allowed_sites=ALLOWED_SITES):
        self._allowed = tuple("/" + site for site in allowed_sites)
        self._original = None
        self._wrapper = None
        self.initialized_before = False
        # (module path, line, caller path, line, function) -> allowed; the caller is the
        # innermost frame outside torch
        self.calls: dict[tuple, bool] = {}

    @classmethod
    def from_env(cls) -> "CudaInitGuard | None":
        if os.environ.get(OPT_OUT_ENV) == "1":
            return None
        guard = cls()
        guard.install()
        return guard

    def install(self) -> None:
        self.initialized_before = torch.cuda.is_initialized()
        original = self._original = torch.cuda._lazy_init

        def _lazy_init():
            self._record(sys._getframe(1))
            return original()

        self._wrapper = torch.cuda._lazy_init = _lazy_init

    def uninstall(self) -> None:
        if self._wrapper is not None and torch.cuda._lazy_init is self._wrapper:
            torch.cuda._lazy_init = self._original
        self._wrapper = None

    def _record(self, frame) -> None:
        caller = None
        allowed = False
        while True:
            path = frame.f_code.co_filename
            # torch's own init re-enters _lazy_init beneath this wrapper
            if caller is None and not path.startswith(_TORCH_DIR) and path != __file__:
                caller = frame
            normalized = path.replace(os.sep, "/")
            allowed = allowed or any(site in normalized for site in self._allowed)
            if frame.f_code.co_name == "<module>" or frame.f_back is None:
                break
            frame = frame.f_back
        caller = caller or frame
        key = (
            frame.f_code.co_filename,
            frame.f_lineno,
            caller.f_code.co_filename,
            caller.f_lineno,
            caller.f_code.co_name,
        )
        self.calls[key] = self.calls.get(key, True) and allowed

    def violations(self, rootdir: Path) -> list[str]:
        lines = []
        if self.initialized_before:
            lines.append("CUDA was already initialized when the guard was installed")
        for key, allowed in self.calls.items():
            if allowed:
                continue
            module_path, module_line, caller_path, caller_line, function = key
            line = f"{_short(module_path, rootdir)}:{module_line}"
            if (caller_path, caller_line) != (module_path, module_line):
                line += (
                    f" (via {_short(caller_path, rootdir)}:{caller_line} in {function})"
                )
            lines.append(line)
        return lines

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_finish(self, session):
        self.uninstall()
        violations = self.violations(session.config.rootpath)
        if not violations:
            return
        longrepr = "\n".join(
            [
                "CUDA was initialized while importing test modules or conftests. Defer these "
                "device queries to first use, or to test setup for skip conditions "
                "(tests/cuda_gates.py):",
                *(f"  {line}" for line in violations),
                f"Set {OPT_OUT_ENV}=1 to skip this check.",
            ]
        )
        report = pytest.CollectReport(
            nodeid="", outcome="failed", longrepr=longrepr, result=[]
        )
        session.ihook.pytest_collectreport(report=report)
