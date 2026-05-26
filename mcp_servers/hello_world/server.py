"""Minimal MCP server used for Phase 0 bootstrap learning."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("axolotl-mcp-hello")


@mcp.tool()
def ping(message: str = "golf") -> dict[str, str]:
    """Return a predictable payload to validate tool wiring."""
    return {
        "status": "ok",
        "echo": message,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
