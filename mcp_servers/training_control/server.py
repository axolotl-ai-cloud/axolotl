"""Training control MCP server entrypoint."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_servers.training_control.tools import (
    get_training_status,
    resume_from_checkpoint,
    start_training,
    stop_training,
)

mcp = FastMCP("axolotl-golf-training-control")


@mcp.tool()
def start_training_job(
    mode: str,
    data_file: str = "data/bethpage_black/train_multitask_case_fixed.jsonl",
    max_steps: int | None = None,
) -> dict[str, object]:
    """Start a training subprocess and return a job id."""
    return start_training(mode=mode, data_file=data_file, max_steps=max_steps)


@mcp.tool()
def training_status(job_id: str, tail_lines: int = 30) -> dict[str, object]:
    """Get current status and recent logs for a job id."""
    return get_training_status(job_id=job_id, tail_lines=tail_lines)


@mcp.tool()
def stop_training_job(job_id: str) -> dict[str, object]:
    """Stop a training subprocess by job id."""
    return stop_training(job_id=job_id)


@mcp.tool()
def resume_training_job(
    mode: str,
    data_file: str = "data/bethpage_black/train_multitask_case_fixed.jsonl",
    max_steps: int | None = None,
) -> dict[str, object]:
    """Resume training by restarting mode; script auto-detects checkpoint state."""
    return resume_from_checkpoint(mode=mode, data_file=data_file, max_steps=max_steps)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
