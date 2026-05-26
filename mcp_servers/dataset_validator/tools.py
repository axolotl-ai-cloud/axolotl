"""Dataset validator MCP tools."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _load_jsonl(path: str) -> list[dict]:
	with open(path, "r", encoding="utf-8-sig") as fh:
		return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, records: list[dict]) -> None:
	with open(path, "w", encoding="utf-8") as fh:
		for row in records:
			fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_strategy_number(text: str) -> int | None:
	patterns = [
		r"Strategy:\s*(\d+)\s*yards",
		r"(\d+)-yard\s+(?:strategy|cutoff)",
		r"use\s+the\s+(\d+)-yard",
		r"select\s+the\s+(\d+)-yard",
		r"(\d+)\s+yards",
	]
	for pattern in patterns:
		m = re.search(pattern, text, re.IGNORECASE)
		if m:
			return int(m.group(1))
	return None


def validate_dataset(jsonl_path: str) -> dict[str, object]:
	"""Validate JSONL structure, required fields, and strategy formatting constraints."""
	path = Path(jsonl_path)
	if not path.exists():
		return {"ok": False, "error": f"File not found: {jsonl_path}"}

	issues = []
	total = 0
	task_counts: dict[str, int] = {}

	with open(path, "r", encoding="utf-8-sig") as fh:
		for line_no, line in enumerate(fh, start=1):
			raw = line.strip()
			if not raw:
				continue
			total += 1
			try:
				row = json.loads(raw)
			except json.JSONDecodeError as exc:
				issues.append({"line": line_no, "issue": "invalid_json", "detail": str(exc)})
				continue

			task = row.get("task_type")
			if isinstance(task, str):
				task_counts[task] = task_counts.get(task, 0) + 1
			else:
				issues.append({"line": line_no, "issue": "missing_task_type"})

			if "prompt" not in row or not isinstance(row.get("prompt"), str):
				issues.append({"line": line_no, "issue": "missing_or_invalid_prompt"})

			if "completion" not in row or not isinstance(row.get("completion"), str):
				issues.append({"line": line_no, "issue": "missing_or_invalid_completion"})

			if task == "strategy_selection":
				completion = str(row.get("completion", ""))
				if _extract_strategy_number(completion) is None:
					issues.append({"line": line_no, "issue": "unparseable_strategy_completion"})

	return {
		"ok": True,
		"valid": len(issues) == 0,
		"total_records": total,
		"issue_count": len(issues),
		"task_distribution": task_counts,
		"issues": issues,
	}


def get_dataset_stats(jsonl_path: str) -> dict[str, object]:
	"""Compute high-level dataset and task quality metrics."""
	path = Path(jsonl_path)
	if not path.exists():
		return {"ok": False, "error": f"File not found: {jsonl_path}"}

	rows = _load_jsonl(jsonl_path)
	if not rows:
		return {
			"ok": True,
			"total_examples": 0,
			"task_distribution": {},
			"quality_metrics": {},
		}

	task_distribution: dict[str, int] = {}
	strategy_standard = 0
	strategy_total = 0
	use_for_training_true = 0

	for row in rows:
		task = str(row.get("task_type", "unknown"))
		task_distribution[task] = task_distribution.get(task, 0) + 1
		if row.get("use_for_training") is True:
			use_for_training_true += 1

		if task == "strategy_selection":
			strategy_total += 1
			completion = str(row.get("completion", ""))
			if completion.lower().startswith("strategy:") and "yard" in completion.lower():
				strategy_standard += 1

	total_examples = len(rows)
	quality_metrics = {
		"strategy_standard_format_rate": (strategy_standard / strategy_total) if strategy_total else 1.0,
		"use_for_training_rate": use_for_training_true / total_examples,
	}

	return {
		"ok": True,
		"total_examples": total_examples,
		"task_distribution": task_distribution,
		"quality_metrics": quality_metrics,
	}


def filter_by_task(input_jsonl: str, task_type: str, output_jsonl: str) -> dict[str, object]:
	"""Filter input dataset by task_type and write result to output path."""
	in_path = Path(input_jsonl)
	if not in_path.exists():
		return {"ok": False, "error": f"File not found: {input_jsonl}"}

	rows = _load_jsonl(input_jsonl)
	kept = [row for row in rows if row.get("task_type") == task_type]
	out_path = Path(output_jsonl)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	_write_jsonl(str(out_path), kept)

	return {
		"ok": True,
		"input_count": len(rows),
		"filtered_count": len(kept),
		"task_type": task_type,
		"output_jsonl": str(out_path),
	}


def _fix_case(record: dict) -> tuple[dict, int]:
	changes = 0
	prompt = str(record.get("prompt", ""))

	if "TASK: STRATEGY_SELECTION" in prompt:
		prompt = prompt.replace("TASK: STRATEGY_SELECTION", "TASK: strategy_selection")
		changes += 1
	if "TASK: DESCRIPTION_SYNTHESIS" in prompt:
		prompt = prompt.replace("TASK: DESCRIPTION_SYNTHESIS", "TASK: description_synthesis")
		changes += 1

	record["prompt"] = prompt
	return record, changes


def _fix_strategy_format(record: dict) -> tuple[dict, int]:
	if record.get("task_type") != "strategy_selection":
		return record, 0

	completion = str(record.get("completion", ""))
	if completion.startswith("Strategy:") and "yards" in completion:
		return record, 0

	expected = record.get("expected_cutoff_yards")
	yards = int(expected) if isinstance(expected, int) else _extract_strategy_number(completion)
	if yards is None:
		return record, 0

	record["completion"] = f"Strategy: {yards} yards"
	return record, 1


def fix_dataset_issues(jsonl_path: str, issue_type: str = "all") -> dict[str, object]:
	"""Apply case and/or strategy format fixes and write a new JSONL output."""
	in_path = Path(jsonl_path)
	if not in_path.exists():
		return {"ok": False, "error": f"File not found: {jsonl_path}"}
	if issue_type not in {"all", "case", "format"}:
		return {"ok": False, "error": "issue_type must be one of: all, case, format"}

	rows = _load_jsonl(jsonl_path)
	fixed_rows = []
	case_changes = 0
	format_changes = 0

	for row in rows:
		if issue_type in {"all", "case"}:
			row, changes = _fix_case(row)
			case_changes += changes
		if issue_type in {"all", "format"}:
			row, changes = _fix_strategy_format(row)
			format_changes += changes
		fixed_rows.append(row)

	suffix = f".{issue_type}.fixed"
	out_path = in_path.with_name(in_path.stem + suffix + in_path.suffix)
	_write_jsonl(str(out_path), fixed_rows)

	return {
		"ok": True,
		"input_jsonl": jsonl_path,
		"output_jsonl": str(out_path),
		"records": len(fixed_rows),
		"case_changes": case_changes,
		"format_changes": format_changes,
	}
