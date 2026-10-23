"""Sharded FSDP2 adapter checkpoints over NCCL: shard writes stage through pinned CPU memory
and consolidate into the gathered adapter file's exact bytes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

WORKER = (
    Path(__file__).parents[2] / "monkeypatch" / "_fsdp2_sharded_adapter_worker.py"
).resolve()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_nccl_sharded_adapter_matches_gathered_adapter(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(WORKER),
            str(tmp_path),
        ],
        env=os.environ
        | {"AXOLOTL_SHARDED_ADAPTER_DEVICE": "cuda", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS sharded-") == 13, result.stdout


def adapter_mismatches(left, right) -> list[str]:
    """Differences between two adapter safetensors files: keys, dtypes, shapes, bytes."""
    import filecmp

    from safetensors.torch import load_file

    if filecmp.cmp(left, right, shallow=False):
        return []
    a, b = load_file(left), load_file(right)
    problems = [f"only in {left}: {k}" for k in sorted(a.keys() - b.keys())]
    problems += [f"only in {right}: {k}" for k in sorted(b.keys() - a.keys())]
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key], b[key]
        if (x.dtype, x.shape) != (y.dtype, y.shape):
            problems.append(
                f"{key}: {x.dtype} {list(x.shape)} vs {y.dtype} {list(y.shape)}"
            )
        elif not torch.equal(
            x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8)
        ):
            problems.append(f"{key}: bytes differ")
    return problems or ["same tensors, different file bytes (header or key order)"]


if __name__ == "__main__":
    # python tests/e2e/multigpu/test_fsdp2_sharded_adapter_save.py checkpoint-4/adapter_model.safetensors adapter_model.safetensors
    found = adapter_mismatches(*sys.argv[1:3])
    print("\n".join(found) or "identical")
    sys.exit(1 if found else 0)
