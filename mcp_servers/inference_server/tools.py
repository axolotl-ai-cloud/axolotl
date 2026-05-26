"""Inference MCP tools for golf strategy workflows."""

from __future__ import annotations

import os
import re

from mcp_servers.shared.model_loader import get_model_loader

STRATEGY_LINE_RE = re.compile(r"^\s*Strategy:\s*(\d{2,4})\s*(?:yards?)?\s*$", re.IGNORECASE)
NUMBER_RE = re.compile(r"(\d{2,4})")


def _build_strategy_prompt(hole: int, course_conditions: str, handicap: int) -> str:
	return (
		f"[Hole {hole}] Golfer handicap: {handicap}. "
		f"Conditions: {course_conditions}. "
		"Recommend tee shot strategy. "
		"Answer only in this format: Strategy: <N> yards"
	)


def _coerce_strategy_line(text: str) -> str:
	m = STRATEGY_LINE_RE.search(text)
	if m:
		return f"Strategy: {m.group(1)} yards"

	n = NUMBER_RE.search(text)
	if n:
		return f"Strategy: {int(n.group(1))} yards"

	return "Strategy: 260 yards"


def _fallback_strategy(handicap: int) -> str:
	if handicap <= 10:
		return "Strategy: 290 yards"
	if handicap <= 20:
		return "Strategy: 260 yards"
	return "Strategy: 230 yards"


def get_strategy(hole: int, course_conditions: str, handicap: int) -> dict[str, str | int | bool]:
	"""Return a strategy selection for one hole."""
	model_name = os.getenv("AXOLOTL_MCP_MODEL_NAME", "gpt2")
	adapter_dir = os.getenv("AXOLOTL_MCP_ADAPTER_DIR", "") or None
	loader = get_model_loader()

	prompt = _build_strategy_prompt(hole=hole, course_conditions=course_conditions, handicap=handicap)
	fallback = _fallback_strategy(handicap)

	try:
		load_info = loader.load(model_name=model_name, adapter_dir=adapter_dir)
		raw = loader.generate(prompt, max_new_tokens=24)
		strategy = _coerce_strategy_line(raw)
		return {
			"ok": True,
			"hole": hole,
			"strategy": strategy,
			"raw": raw,
			"mode": "model",
			"cached": bool(load_info.get("cached", False)),
		}
	except Exception as exc:
		return {
			"ok": True,
			"hole": hole,
			"strategy": fallback,
			"raw": "",
			"mode": "fallback",
			"cached": False,
			"warning": f"Model inference unavailable: {exc}",
		}


def get_description(hole: int, strategy: str) -> dict[str, str | int | bool]:
	"""Return a short description synthesis using chosen strategy."""
	model_name = os.getenv("AXOLOTL_MCP_MODEL_NAME", "gpt2")
	adapter_dir = os.getenv("AXOLOTL_MCP_ADAPTER_DIR", "") or None
	loader = get_model_loader()

	prompt = (
		f"[Hole {hole}] Chosen tee-shot plan: {strategy}. "
		"Provide a concise one-paragraph explanation for this strategy."
	)

	try:
		loader.load(model_name=model_name, adapter_dir=adapter_dir)
		text = loader.generate(prompt, max_new_tokens=60)
		return {
			"ok": True,
			"hole": hole,
			"strategy": strategy,
			"description": text,
			"mode": "model",
		}
	except Exception:
		# Keep fallback deterministic so downstream tools can rely on stable output shape.
		text = (
			f"Hole {hole}: Use {strategy} as a controlled tee-shot target, then adjust the approach "
			"based on lie quality and remaining distance."
		)
		return {
			"ok": True,
			"hole": hole,
			"strategy": strategy,
			"description": text,
			"mode": "fallback",
		}


def batch_analyze(scenarios: list[dict]) -> dict[str, object]:
	"""Run strategy + description generation for multiple scenarios."""
	results = []
	for idx, scenario in enumerate(scenarios):
		hole = int(scenario.get("hole", idx + 1))
		course_conditions = str(scenario.get("course_conditions", "standard fairway conditions"))
		handicap = int(scenario.get("handicap", 15))

		strategy_rec = get_strategy(hole=hole, course_conditions=course_conditions, handicap=handicap)
		strategy_text = str(strategy_rec.get("strategy", "Strategy: 260 yards"))
		description_rec = get_description(hole=hole, strategy=strategy_text)
		results.append(
			{
				"hole": hole,
				"strategy": strategy_text,
				"strategy_mode": strategy_rec.get("mode", "fallback"),
				"description": description_rec.get("description", ""),
				"description_mode": description_rec.get("mode", "fallback"),
			}
		)

	return {
		"ok": True,
		"count": len(results),
		"results": results,
	}
