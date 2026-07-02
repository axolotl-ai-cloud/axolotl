"""Shared bootstrap + reporting for the sonicmoe NVFP4 smoke scripts.

The scripts import the sonicmoe lib modules by path (no axolotl install needed
on the pod) and compare kernel outputs against fp32 dequantized-operand oracles.
"""

import pathlib
import sys

LIB_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/axolotl/integrations/kernels/libs/sonicmoe"
)
sys.path.insert(0, str(LIB_DIR))

_FAILURES = []


def report(name, out, ref, rtol=2e-2, atol=2e-2):
    """Compare a bf16 kernel output against an fp32 oracle and print a verdict."""
    import torch

    out_f = out.float()
    ref_f = ref.float()
    abs_err = (out_f - ref_f).abs()
    denom = ref_f.abs().clamp(min=1e-6)
    max_abs = float(abs_err.max()) if abs_err.numel() else 0.0
    max_rel = float((abs_err / denom).max()) if abs_err.numel() else 0.0
    ok = bool(torch.allclose(out_f, ref_f, rtol=rtol, atol=atol))
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {name}: max_abs={max_abs:.4e} max_rel={max_rel:.4e} "
        f"(rtol={rtol}, atol={atol}, shape={tuple(out.shape)})"
    )
    if not ok:
        _FAILURES.append(name)
    return ok


def check(name, ok):
    """Record a boolean check with the same PASS/FAIL bookkeeping as report()."""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        _FAILURES.append(name)
    return ok


def finish():
    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) FAILED: {_FAILURES}")
        sys.exit(1)
    print("\nall checks passed")


def require_sm100():
    import torch

    assert torch.cuda.is_available(), "CUDA required"
    cap = torch.cuda.get_device_capability()
    assert cap[0] in (10, 11), f"SM100/SM110 required, got sm_{cap[0]}{cap[1]}"
    return cap
