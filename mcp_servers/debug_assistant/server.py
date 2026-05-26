"""Debug assistant MCP server entrypoint."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_servers.debug_assistant.tools import (
    compare_single_vs_multitask,
    sample_failures,
    test_task_embedding,
)

mcp = FastMCP("axolotl-golf-debug-assistant")


@mcp.tool()
def sample_strategy_failures(
    task: str = "strategy_selection",
    count: int = 10,
    inference_file: str = "outputs/bethpage-lora/inference_multitask_strategy.jsonl",
    data_file: str = "outputs/bethpage-lora/strategy_eval.jsonl",
) -> dict[str, object]:
    """Sample strategy mismatches from inference outputs."""
    return sample_failures(task=task, count=count, inference_file=inference_file, data_file=data_file)


@mcp.tool()
def diagnose_task_prefix(
    scenario_text: str,
    model_name: str = "gpt2",
    adapter_dir: str = "",
) -> dict[str, object]:
    """Diagnose whether prompt task prefix appears recognized by the model."""
    return test_task_embedding(scenario_text=scenario_text, model_name=model_name, adapter_dir=adapter_dir)


@mcp.tool()
def compare_adapters(
    scenario: dict,
    model_name: str = "gpt2",
    single_adapter_dir: str = "",
    multitask_adapter_dir: str = "",
) -> dict[str, object]:
    """Compare single-task and multitask adapter outputs for same scenario."""
    return compare_single_vs_multitask(
        scenario=scenario,
        model_name=model_name,
        single_adapter_dir=single_adapter_dir,
        multitask_adapter_dir=multitask_adapter_dir,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
