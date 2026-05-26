"""Dataset validator MCP server entrypoint."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_servers.dataset_validator.tools import (
    filter_by_task,
    fix_dataset_issues,
    get_dataset_stats,
    validate_dataset,
)

mcp = FastMCP("axolotl-golf-dataset-validator")


@mcp.tool()
def validate_dataset_file(jsonl_path: str) -> dict[str, object]:
    """Validate JSONL dataset structure and basic content constraints."""
    return validate_dataset(jsonl_path=jsonl_path)


@mcp.tool()
def dataset_stats(jsonl_path: str) -> dict[str, object]:
    """Return task distribution and quality metrics."""
    return get_dataset_stats(jsonl_path=jsonl_path)


@mcp.tool()
def filter_dataset_by_task(input_jsonl: str, task_type: str, output_jsonl: str) -> dict[str, object]:
    """Filter a JSONL file to a specific task_type."""
    return filter_by_task(input_jsonl=input_jsonl, task_type=task_type, output_jsonl=output_jsonl)


@mcp.tool()
def repair_dataset(jsonl_path: str, issue_type: str = "all") -> dict[str, object]:
    """Apply dataset repairs for case labels and/or strategy format."""
    return fix_dataset_issues(jsonl_path=jsonl_path, issue_type=issue_type)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
