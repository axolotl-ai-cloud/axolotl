"""End-to-end MCP learning demo using direct tool calls.

This script exercises the same logic exposed by the MCP servers so the workflow
can be tested quickly in a local Python session.
"""

from __future__ import annotations

import json

from mcp_servers.checkpoint_browser.tools import list_checkpoints
from mcp_servers.dataset_validator.tools import get_dataset_stats, validate_dataset
from mcp_servers.debug_assistant.tools import compare_single_vs_multitask, test_task_embedding
from mcp_servers.inference_server.tools import get_description, get_strategy


def run_demo() -> dict[str, object]:
    dataset_path = "data/bethpage_black/train_multitask_case_fixed.jsonl"

    results: dict[str, object] = {}
    results["dataset_validation"] = validate_dataset(dataset_path)
    results["dataset_stats"] = get_dataset_stats(dataset_path)

    strategy = get_strategy(
        hole=11,
        course_conditions="crosswind with firm fairway and right-side bunker pressure",
        handicap=14,
    )
    results["single_inference"] = strategy

    results["description"] = get_description(
        hole=11,
        strategy=str(strategy.get("strategy", "Strategy: 260 yards")),
    )

    scenario_text = (
        "TASK: strategy_selection\\n"
        "Hole 11, Bethpage Black: Par 4, 435 yards. Golfer drives 290 yards. "
        "Best strategy? Respond with: Strategy: <number> yards"
    )
    results["task_prefix_diagnostic"] = test_task_embedding(scenario_text=scenario_text)

    results["adapter_compare"] = compare_single_vs_multitask(
        scenario={"prompt": scenario_text},
    )

    results["checkpoints"] = list_checkpoints("outputs/bethpage-lora")

    return results


def main() -> None:
    payload = run_demo()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
