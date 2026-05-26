"""Debug assistant MCP tools."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from mcp_servers.inference_server.tools import _coerce_strategy_line
from mcp_servers.shared.model_loader import get_model_loader

STD_RE = re.compile(r"^\s*Strategy:\s*(\d{2,4})\s*(?:yards?)?\s*$", re.IGNORECASE)
NUM_RE = re.compile(r"(\d{2,4})")


def _load_jsonl(path: str) -> list[dict]:
	with open(path, "r", encoding="utf-8-sig") as fh:
		return [json.loads(line) for line in fh if line.strip()]


def _extract_strategy(text: str) -> int | None:
	m = STD_RE.search(text)
	if m:
		return int(m.group(1))
	n = NUM_RE.search(text)
	if n:
		return int(n.group(1))
	return None


def _expected_cutoff(example: dict) -> int | None:
	exp = example.get("expected_cutoff_yards")
	if isinstance(exp, int):
		return exp
	available = example.get("available_cutoffs")
	if isinstance(available, list):
		vals = [int(v) for v in available if isinstance(v, int)]
		if vals:
			return max(vals)
	strategies = example.get("tee_shot_strategies")
	if isinstance(strategies, list):
		vals = []
		for item in strategies:
			if isinstance(item, dict) and isinstance(item.get("cutoff_distance"), int):
				vals.append(int(item["cutoff_distance"]))
		if vals:
			return max(vals)
	return None


def _run_generation(prompt: str, model_name: str, adapter_dir: str | None, max_new_tokens: int = 36) -> str:
	loader = get_model_loader()
	loader.load(model_name=model_name, adapter_dir=adapter_dir)
	return loader.generate(prompt, max_new_tokens=max_new_tokens)


def sample_failures(
	task: str = "strategy_selection",
	count: int = 10,
	inference_file: str = "outputs/bethpage-lora/inference_multitask_strategy.jsonl",
	data_file: str = "outputs/bethpage-lora/strategy_eval.jsonl",
) -> dict[str, object]:
	"""Sample mismatch examples for quick failure inspection."""
	inf_path = Path(inference_file)
	data_path = Path(data_file)
	if not inf_path.exists() or not data_path.exists():
		return {
			"ok": False,
			"error": "inference or dataset file not found",
			"inference_file": inference_file,
			"data_file": data_file,
			"hint": "Run strategy evaluation flow first, then retry sample_failures.",
		}

	inference_rows = _load_jsonl(str(inf_path))
	data_rows = _load_jsonl(str(data_path))

	failures: list[dict] = []
	for idx, (inf, dat) in enumerate(zip(inference_rows, data_rows)):
		if task and dat.get("task_type") and dat.get("task_type") != task:
			continue
		expected = _expected_cutoff(dat)
		chosen = _extract_strategy(str(inf.get("completion", "")))
		if expected is None:
			continue
		if chosen != expected:
			failures.append(
				{
					"index": idx,
					"hole": dat.get("hole"),
					"expected": expected,
					"chosen": chosen,
					"prompt": dat.get("prompt", ""),
					"completion": inf.get("completion", ""),
				}
			)

	sampled = failures if len(failures) <= count else random.sample(failures, count)
	return {
		"ok": True,
		"task": task,
		"total_checked": min(len(inference_rows), len(data_rows)),
		"failure_count": len(failures),
		"sampled_count": len(sampled),
		"samples": sampled,
	}


def test_task_embedding(
	scenario_text: str,
	model_name: str = "gpt2",
	adapter_dir: str = "",
) -> dict[str, object]:
	"""Check whether prompt task prefix appears consistent and optionally probe the model."""
	prefix_match = re.search(r"TASK:\s*(strategy_selection|description_synthesis)", scenario_text, re.IGNORECASE)
	detected_task = prefix_match.group(1).lower() if prefix_match else "unknown"

	expected_style = "strategy_line" if detected_task == "strategy_selection" else "free_text"
	result: dict[str, object] = {
		"ok": True,
		"detected_task": detected_task,
		"expected_style": expected_style,
		"prefix_present": bool(prefix_match),
	}

	if not prefix_match:
		result["task_recognized"] = False
		result["confidence"] = 0.0
		result["details"] = "No TASK prefix found in scenario text"
		return result

	if not adapter_dir:
		adapter_dir = os.getenv("AXOLOTL_MCP_ADAPTER_DIR", "")

	try:
		generated = _run_generation(
			prompt=scenario_text,
			model_name=model_name,
			adapter_dir=adapter_dir or None,
			max_new_tokens=40,
		)
		looks_like_strategy = bool(STD_RE.search(generated) or NUM_RE.search(generated))
		if detected_task == "strategy_selection":
			recognized = looks_like_strategy
			confidence = 0.9 if recognized else 0.3
		else:
			recognized = not looks_like_strategy or len(generated.split()) >= 10
			confidence = 0.85 if recognized else 0.35

		result.update(
			{
				"task_recognized": recognized,
				"confidence": confidence,
				"generated_preview": generated[:280],
			}
		)
		return result
	except Exception as exc:
		result.update(
			{
				"task_recognized": bool(prefix_match),
				"confidence": 0.5,
				"details": f"Model probe unavailable ({exc}); returning prefix-based assessment",
			}
		)
		return result


def compare_single_vs_multitask(
	scenario: dict,
	model_name: str = "gpt2",
	single_adapter_dir: str = "",
	multitask_adapter_dir: str = "",
) -> dict[str, object]:
	"""Generate on same scenario with two adapters and return structured diff."""
	if not single_adapter_dir:
		single_adapter_dir = os.getenv("AXOLOTL_MCP_SINGLE_ADAPTER_DIR", "outputs/bethpage-lora/checkpoint-quick")
	if not multitask_adapter_dir:
		multitask_adapter_dir = os.getenv("AXOLOTL_MCP_MULTITASK_ADAPTER_DIR", "outputs/bethpage-lora/checkpoint-multitask")

	prompt = str(scenario.get("prompt", ""))
	if not prompt:
		hole = int(scenario.get("hole", 1))
		handicap = int(scenario.get("handicap", 15))
		conditions = str(scenario.get("course_conditions", "standard fairway conditions"))
		prompt = (
			f"TASK: strategy_selection\n[Hole {hole}] Golfer handicap: {handicap}. "
			f"Conditions: {conditions}. Respond with: Strategy: <number> yards"
		)

	errors: list[str] = []

	try:
		single_raw = _run_generation(prompt=prompt, model_name=model_name, adapter_dir=single_adapter_dir or None)
	except Exception as exc:
		single_raw = ""
		errors.append(f"single adapter unavailable: {exc}")

	try:
		multi_raw = _run_generation(prompt=prompt, model_name=model_name, adapter_dir=multitask_adapter_dir or None)
	except Exception as exc:
		multi_raw = ""
		errors.append(f"multitask adapter unavailable: {exc}")

	single_strategy = _coerce_strategy_line(single_raw) if single_raw else ""
	multi_strategy = _coerce_strategy_line(multi_raw) if multi_raw else ""

	return {
		"ok": True,
		"prompt": prompt,
		"single": {
			"adapter": single_adapter_dir,
			"raw": single_raw,
			"strategy": single_strategy,
		},
		"multitask": {
			"adapter": multitask_adapter_dir,
			"raw": multi_raw,
			"strategy": multi_strategy,
		},
		"diff": {
			"different_raw": single_raw != multi_raw,
			"different_strategy": single_strategy != multi_strategy,
		},
		"errors": errors,
	}
