"""Worker for test_checkpoint_mirror_dist: 4 gloo ranks posing as 2 nodes of 2 ranks."""

import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import torch.distributed as dist
from transformers.trainer_utils import rotate_checkpoints


def main(root: str, mode: str) -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    host = "node-a" if rank < 2 else "node-b"
    socket.gethostname = lambda: host

    from axolotl.core import checkpoint_mirror

    root_path = Path(root)
    output_dir = root_path / ("shared" if mode == "shared" else host)
    mirror_dir = root_path / "nfs"
    callback = checkpoint_mirror.CheckpointMirrorCallback(str(mirror_dir))
    args = SimpleNamespace(output_dir=str(output_dir), save_total_limit=2)
    callback.on_train_begin(args, SimpleNamespace(), SimpleNamespace())

    for step in (1, 2, 3):
        checkpoint = output_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / f"rng_state_{rank}.pth").write_bytes(bytes([rank, step]))
        dist.barrier()
        if rank == 0:
            (checkpoint / "trainer_state.json").write_text(json.dumps({"step": step}))
            # HF rotates on the global main process only, before on_save
            rotate_checkpoints(str(output_dir), save_total_limit=2)
        state = SimpleNamespace(global_step=step, best_model_checkpoint=None)
        callback.on_save(args, state, SimpleNamespace())

    if rank == 0:
        (output_dir / "adapter_model.safetensors").write_bytes(b"final")
    callback.finish()
    dist.barrier()

    local = checkpoint_mirror.checkpoint_steps(output_dir)
    local_last = str(local[max(local)]) if local else None
    resume = checkpoint_mirror.resume_checkpoint(output_dir, mirror_dir, local_last)
    layout = callback.layout
    assert layout is not None
    report = {
        "rank": rank,
        "layout": [layout.uploader, layout.node_index, layout.num_nodes],
        "local": sorted(local),
        "resume": str(Path(resume).relative_to(root_path)) if resume else None,
    }
    if rank == 0:
        report["mirror"] = {
            name: sorted(p.name for p in path.iterdir() if not p.name.startswith("."))
            for name, path in (
                (f"checkpoint-{s}", p)
                for s, p in sorted(
                    checkpoint_mirror.checkpoint_steps(mirror_dir).items()
                )
            )
        }
        report["final"] = (
            (mirror_dir / "adapter_model.safetensors").read_bytes().decode()
        )
        report["staging"] = sorted(p.name for p in mirror_dir.glob(".staging-*"))
    (root_path / f"report-{rank}.json").write_text(json.dumps(report))
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
