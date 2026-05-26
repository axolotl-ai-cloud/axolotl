"""Checkpoint browser MCP tools."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _iso(ts: float) -> str:
	return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _maybe_read_json(path: Path) -> dict:
	if not path.exists():
		return {}
	try:
		with open(path, "r", encoding="utf-8") as fh:
			return json.load(fh)
	except Exception:
		return {}


def _find_best_log_for_checkpoint(base_dir: Path, checkpoint_name: str) -> Path | None:
	mode_hint = checkpoint_name.replace("checkpoint-", "")
	log_candidates = sorted(base_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
	for log_path in log_candidates:
		if mode_hint in log_path.name:
			return log_path
	return log_candidates[0] if log_candidates else None


def _parse_train_loss(log_path: Path | None) -> float | None:
	if log_path is None or not log_path.exists():
		return None
	pattern = re.compile(r"train_loss[:'=\s]+([0-9]+(?:\.[0-9]+)?)")
	last_val = None
	try:
		with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
			for line in fh:
				m = pattern.search(line)
				if m:
					last_val = float(m.group(1))
	except Exception:
		return None
	return last_val


def list_checkpoints(base_dir: str = "outputs/bethpage-lora") -> dict[str, object]:
	"""List available checkpoint directories with lightweight metadata."""
	root = Path(base_dir)
	if not root.exists():
		return {"ok": False, "error": f"Directory not found: {base_dir}"}

	checkpoints = []
	for cp in sorted(root.glob("checkpoint-*")):
		if not cp.is_dir():
			continue
		log_path = _find_best_log_for_checkpoint(root, cp.name)
		checkpoints.append(
			{
				"name": cp.name,
				"path": str(cp),
				"created_at": _iso(cp.stat().st_ctime),
				"updated_at": _iso(cp.stat().st_mtime),
				"has_adapter": (cp / "adapter_model.safetensors").exists(),
				"has_config": (cp / "adapter_config.json").exists(),
				"train_loss": _parse_train_loss(log_path),
				"log_path": str(log_path) if log_path else "",
			}
		)

	return {
		"ok": True,
		"base_dir": str(root),
		"count": len(checkpoints),
		"checkpoints": checkpoints,
	}


def get_checkpoint_metadata(checkpoint_dir: str) -> dict[str, object]:
	"""Return detailed metadata for one checkpoint directory."""
	cp = Path(checkpoint_dir)
	if not cp.exists() or not cp.is_dir():
		return {"ok": False, "error": f"Checkpoint directory not found: {checkpoint_dir}"}

	adapter_cfg = _maybe_read_json(cp / "adapter_config.json")
	trainer_state = _maybe_read_json(cp / "trainer_state.json")

	file_sizes = {}
	total_size = 0
	for path in cp.rglob("*"):
		if path.is_file():
			size = path.stat().st_size
			total_size += size
			file_sizes[path.name] = size

	metrics = {}
	if isinstance(trainer_state.get("log_history"), list):
		for entry in reversed(trainer_state["log_history"]):
			if isinstance(entry, dict) and "loss" in entry:
				metrics["last_step_loss"] = entry.get("loss")
				metrics["step"] = entry.get("step")
				break

	return {
		"ok": True,
		"checkpoint_dir": str(cp),
		"created_at": _iso(cp.stat().st_ctime),
		"updated_at": _iso(cp.stat().st_mtime),
		"total_size_bytes": total_size,
		"file_sizes": file_sizes,
		"lora_config": {
			"r": adapter_cfg.get("r"),
			"lora_alpha": adapter_cfg.get("lora_alpha"),
			"lora_dropout": adapter_cfg.get("lora_dropout"),
			"target_modules": adapter_cfg.get("target_modules"),
		},
		"metrics": metrics,
	}


def compare_checkpoints(checkpoint1: str, checkpoint2: str) -> dict[str, object]:
	"""Compare two checkpoints by metadata and extracted metrics."""
	left = get_checkpoint_metadata(checkpoint1)
	right = get_checkpoint_metadata(checkpoint2)

	if not left.get("ok"):
		return left
	if not right.get("ok"):
		return right

	left_size = int(left.get("total_size_bytes", 0))
	right_size = int(right.get("total_size_bytes", 0))
	left_loss = left.get("metrics", {}).get("last_step_loss")
	right_loss = right.get("metrics", {}).get("last_step_loss")

	loss_delta = None
	if isinstance(left_loss, (int, float)) and isinstance(right_loss, (int, float)):
		loss_delta = right_loss - left_loss

	return {
		"ok": True,
		"checkpoint1": left,
		"checkpoint2": right,
		"diff": {
			"size_delta_bytes": right_size - left_size,
			"loss_delta": loss_delta,
			"same_lora_rank": left.get("lora_config", {}).get("r") == right.get("lora_config", {}).get("r"),
		},
	}


def export_model(checkpoint_dir: str, format: str = "adapters") -> dict[str, object]:
	"""Export checkpoint artifacts to a standard export location."""
	cp = Path(checkpoint_dir)
	if not cp.exists() or not cp.is_dir():
		return {"ok": False, "error": f"Checkpoint directory not found: {checkpoint_dir}"}
	if format not in {"adapters", "safetensors", "onnx"}:
		return {"ok": False, "error": "format must be one of: adapters, safetensors, onnx"}

	export_root = cp.parent / "exports"
	export_root.mkdir(parents=True, exist_ok=True)

	if format == "adapters":
		target = export_root / f"{cp.name}-adapters"
		if target.exists():
			shutil.rmtree(target)
		shutil.copytree(cp, target)
		return {
			"ok": True,
			"format": format,
			"export_path": str(target),
			"file_size": sum(p.stat().st_size for p in target.rglob("*") if p.is_file()),
		}

	if format == "safetensors":
		source = cp / "adapter_model.safetensors"
		if not source.exists():
			return {"ok": False, "error": "adapter_model.safetensors not found in checkpoint"}
		target = export_root / f"{cp.name}.safetensors"
		shutil.copy2(source, target)
		return {
			"ok": True,
			"format": format,
			"export_path": str(target),
			"file_size": target.stat().st_size,
		}

	existing_onnx = list(cp.glob("*.onnx"))
	if existing_onnx:
		source = existing_onnx[0]
		target = export_root / source.name
		shutil.copy2(source, target)
		return {
			"ok": True,
			"format": format,
			"export_path": str(target),
			"file_size": target.stat().st_size,
		}

	return {
		"ok": False,
		"error": "No ONNX artifact found. Generate ONNX externally, then rerun export.",
		"checkpoint_dir": str(cp),
	}
