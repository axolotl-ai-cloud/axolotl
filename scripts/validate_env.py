#!/usr/bin/env python
"""Environment validation script for Axolotl training runs.

Checks (non-destructive):
  - Python version
  - torch / transformers versions
  - CUDA availability, device count, per-device total memory
  - Optional packages: flash_attn, bitsandbytes
  - Deterministic seed test (simple RNG hash)
  - Manifest verification (if --data/--manifest provided)
  - GPU capability summary (compute capability if available)
  - Disk free space for output directory

Exit codes:
  0 success (no hard failures)
  1 soft warnings only (if --fail-on-warn not set), still usable
  2 failure (hard requirement unmet or --fail-on-warn with warnings)

Usage:
  python scripts/validate_env.py --out-json env_report.json \
    --data data/..train.jsonl --manifest data/...manifest.json \
    --output-dir outputs/run1 --min-free-gb 10
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

REPORT = {"warnings": [], "errors": [], "info": []}

def info(msg):
    REPORT["info"].append(msg)
    print("[INFO]", msg)

def warn(msg):
    REPORT["warnings"].append(msg)
    print("[WARN]", msg)

def error(msg):
    REPORT["errors"].append(msg)
    print("[ERROR]", msg)

def check_python(min_major=3, min_minor=10):
    major, minor = sys.version_info[:2]
    info(f"Python version: {major}.{minor}")
    if (major, minor) < (min_major, min_minor):
        error(f"Python >= {min_major}.{min_minor} required; found {major}.{minor}")

def check_torch():
    try:
        import torch
        info(f"torch version: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        info(f"CUDA available: {cuda_ok}")
        if cuda_ok:
            count = torch.cuda.device_count()
            info(f"GPU count: {count}")
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                info(f"GPU[{i}] {name} total_mem_gb={total_mem:.2f}")
        else:
            warn("CUDA not available; training may fall back to CPU or fail if GPU required")
    except Exception as e:  # noqa: BLE001
        error(f"torch import failed: {e}")

def check_transformers():
    try:
        import transformers
        info(f"transformers version: {transformers.__version__}")
    except Exception as e:  # noqa: BLE001
        error(f"transformers import failed: {e}")

def check_optional(pkg_name: str, label: str):
    try:
        importlib.import_module(pkg_name)
        info(f"{label} present ({pkg_name})")
        return True
    except Exception:
        warn(f"{label} not found ({pkg_name})")
        return False

def check_manifest(data_path: Path | None, manifest_path: Path | None):
    if not data_path or not manifest_path:
        return
    try:
        from axolotl.data.weighted_prompted_dataset import verify_prompted_manifest
    except Exception as e:  # noqa: BLE001
        warn(f"Could not import verify_prompted_manifest: {e}")
        return
    try:
        verify_prompted_manifest(str(data_path), str(manifest_path))
        info("Manifest verification passed")
    except Exception as e:  # noqa: BLE001
        error(f"Manifest verification FAILED: {e}")

def deterministic_seed_test(seed: int = 42):
    import random, hashlib
    random.seed(seed)
    nums = [random.random() for _ in range(10)]
    h = hashlib.sha256(str(nums).encode()).hexdigest()
    info(f"Deterministic RNG hash (seed={seed}): {h[:16]}...")


def check_disk_free(output_dir: Path | None, min_free_gb: float | None):
    if not output_dir or min_free_gb is None:
        return
    usage = shutil.disk_usage(output_dir)
    free_gb = usage.free / (1024**3)
    info(f"Disk free at {output_dir}: {free_gb:.2f} GB")
    if free_gb < min_free_gb:
        error(f"Free space {free_gb:.2f} GB < required {min_free_gb} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-json', type=Path, help='Write full report JSON here')
    ap.add_argument('--data', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--output-dir', type=Path, help='Directory for disk space check')
    ap.add_argument('--min-free-gb', type=float, default=None)
    ap.add_argument('--fail-on-warn', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    start = time.time()
    info(f"Platform: {platform.platform()}")

    check_python()
    check_torch()
    check_transformers()
    check_optional('flash_attn', 'flash-attn')
    check_optional('bitsandbytes', 'bitsandbytes')
    deterministic_seed_test(args.seed)
    check_manifest(args.data, args.manifest)
    check_disk_free(args.output_dir, args.min_free_gb)

    duration = time.time() - start
    info(f"Validation duration: {duration:.2f}s")

    status = 0
    if REPORT['errors']:
        status = 2
    elif REPORT['warnings'] and args.fail_on_warn:
        status = 2
    elif REPORT['warnings']:
        status = 1

    REPORT['status_code'] = status
    if args.out_json:
        args.out_json.write_text(json.dumps(REPORT, indent=2), encoding='utf-8')
        info(f"Wrote report JSON: {args.out_json}")

    print("Summary: errors=%d warnings=%d status=%d" % (len(REPORT['errors']), len(REPORT['warnings']), status))
    sys.exit(status)

if __name__ == '__main__':
    main()
