"""Shared utility helpers for MCP servers."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


def health() -> dict[str, str]:
    """Return a lightweight health payload for basic checks."""
    return {"status": "ok"}


class TrainingJobManager:
    """Manage long-running training subprocesses for MCP tools."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._repo_root = Path(__file__).resolve().parents[2]
        self._default_data_file = "data/bethpage_black/train_multitask_case_fixed.jsonl"

    def start_training(
        self,
        mode: str,
        data_file: str | None = None,
        max_steps: int | None = None,
        log_path: str | None = None,
    ) -> dict[str, object]:
        """Start training subprocess and register a job id."""
        if mode not in {"debug", "debug_training", "1_hour", "8_hour"}:
            raise ValueError(f"Unsupported mode: {mode}")

        python_exe = os.getenv("AXOLOTL_MCP_PYTHON", sys.executable)
        data = data_file or self._default_data_file

        out_log = log_path or f"outputs/bethpage-lora/mcp_training_{mode}_{int(time.time())}.log"
        out_path = self._repo_root / out_log
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_exe,
            "train_lora_bethpage_strat.py",
            "--mode",
            mode,
            "--data-file",
            data,
        ]
        if isinstance(max_steps, int) and max_steps > 0:
            cmd.extend(["--max-steps", str(max_steps)])

        logfile = open(out_path, "a", encoding="utf-8")
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=str(self._repo_root),
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )

        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "id": job_id,
            "mode": mode,
            "data_file": data,
            "max_steps": max_steps,
            "log_path": str(out_path),
            "cmd": cmd,
            "pid": proc.pid,
            "process": proc,
            "log_handle": logfile,
            "started_at": time.time(),
        }

        return {
            "ok": True,
            "job_id": job_id,
            "pid": proc.pid,
            "mode": mode,
            "log_path": str(out_path),
        }

    def get_status(self, job_id: str, tail_lines: int = 30) -> dict[str, object]:
        """Return status and recent logs for a training job."""
        if job_id not in self._jobs:
            return {"ok": False, "error": f"Unknown job_id: {job_id}"}

        job = self._jobs[job_id]
        proc = job["process"]
        return_code = proc.poll()
        running = return_code is None
        elapsed_sec = int(time.time() - job["started_at"])

        logs = ""
        log_path = Path(job["log_path"])
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
                logs = "".join(lines[-max(1, tail_lines) :]).rstrip()

        return {
            "ok": True,
            "job_id": job_id,
            "mode": job["mode"],
            "pid": job["pid"],
            "running": running,
            "return_code": return_code,
            "elapsed_sec": elapsed_sec,
            "log_path": str(log_path),
            "logs_tail": logs,
        }

    def stop(self, job_id: str) -> dict[str, object]:
        """Stop a running training job by id."""
        if job_id not in self._jobs:
            return {"ok": False, "error": f"Unknown job_id: {job_id}"}

        job = self._jobs[job_id]
        proc = job["process"]
        if proc.poll() is not None:
            return {"ok": True, "job_id": job_id, "stopped": False, "reason": "already_finished"}

        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

        return {
            "ok": True,
            "job_id": job_id,
            "stopped": True,
            "return_code": proc.returncode,
        }


_TRAINING_MANAGER: TrainingJobManager | None = None


def get_training_job_manager() -> TrainingJobManager:
    """Return process-wide singleton training manager."""
    global _TRAINING_MANAGER
    if _TRAINING_MANAGER is None:
        _TRAINING_MANAGER = TrainingJobManager()
    return _TRAINING_MANAGER
