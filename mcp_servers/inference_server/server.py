"""Inference MCP server entrypoint."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_servers.inference_server.tools import batch_analyze, get_description, get_strategy

mcp = FastMCP("axolotl-golf-inference")


@mcp.tool()
def strategy_for_hole(hole: int, course_conditions: str, handicap: int) -> dict[str, object]:
    """Get tee-shot strategy recommendation for a single hole."""
    return get_strategy(hole=hole, course_conditions=course_conditions, handicap=handicap)


@mcp.tool()
def description_for_strategy(hole: int, strategy: str) -> dict[str, object]:
    """Generate a short narrative for a chosen strategy."""
    return get_description(hole=hole, strategy=strategy)


@mcp.tool()
def analyze_scenarios(scenarios: list[dict]) -> dict[str, object]:
    """Analyze multiple scenarios in one call."""
    return batch_analyze(scenarios=scenarios)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
