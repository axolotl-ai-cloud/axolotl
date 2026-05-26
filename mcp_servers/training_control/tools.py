"""Training control MCP tools."""

from __future__ import annotations

from mcp_servers.shared.utils import get_training_job_manager


def start_training(
	mode: str,
	data_file: str = "data/bethpage_black/train_multitask_case_fixed.jsonl",
	max_steps: int | None = None,
) -> dict[str, object]:
	"""Start a new training job and return job metadata."""
	manager = get_training_job_manager()
	return manager.start_training(mode=mode, data_file=data_file, max_steps=max_steps)


def get_training_status(job_id: str, tail_lines: int = 30) -> dict[str, object]:
	"""Fetch running/completed status for one training job."""
	manager = get_training_job_manager()
	return manager.get_status(job_id=job_id, tail_lines=tail_lines)


def stop_training(job_id: str) -> dict[str, object]:
	"""Stop a training job by id."""
	manager = get_training_job_manager()
	return manager.stop(job_id=job_id)


def resume_from_checkpoint(
	mode: str,
	data_file: str = "data/bethpage_black/train_multitask_case_fixed.jsonl",
	max_steps: int | None = None,
) -> dict[str, object]:
	"""Resume behaves like start because training script auto-resumes from checkpoint state."""
	manager = get_training_job_manager()
	return manager.start_training(mode=mode, data_file=data_file, max_steps=max_steps)
