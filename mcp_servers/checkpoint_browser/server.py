"""Checkpoint browser MCP server entrypoint."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_servers.checkpoint_browser.tools import (
    compare_checkpoints,
    export_model,
    get_checkpoint_metadata,
    list_checkpoints,
)

mcp = FastMCP("axolotl-golf-checkpoint-browser")


@mcp.tool()
def list_model_checkpoints(base_dir: str = "outputs/bethpage-lora") -> dict[str, object]:
    """List model checkpoints with lightweight metadata."""
    return list_checkpoints(base_dir=base_dir)


@mcp.tool()
def checkpoint_metadata(checkpoint_dir: str) -> dict[str, object]:
    """Return detailed metadata for one checkpoint."""
    return get_checkpoint_metadata(checkpoint_dir=checkpoint_dir)


@mcp.tool()
def compare_model_checkpoints(checkpoint1: str, checkpoint2: str) -> dict[str, object]:
    """Compare two checkpoints by size and available metrics."""
    return compare_checkpoints(checkpoint1=checkpoint1, checkpoint2=checkpoint2)


@mcp.tool()
def export_checkpoint_artifacts(checkpoint_dir: str, format: str = "adapters") -> dict[str, object]:
    """Export checkpoint artifacts to a reusable location."""
    return export_model(checkpoint_dir=checkpoint_dir, format=format)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
