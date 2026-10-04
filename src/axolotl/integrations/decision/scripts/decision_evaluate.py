"""Evaluate prepared typed decision canvases through Axolotl's generic HF loader."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import torch

from axolotl.cli.config import load_cfg
from axolotl.cli.utils import load_model_and_tokenizer
from axolotl.integrations.decision.data_audit import preparation_audit_path
from axolotl.integrations.decision.datasets import (
    load_decision_datasets,
    load_decision_evaluation_dataset,
)
from axolotl.integrations.decision.evaluation import (
    evaluate_artifacts,
    evaluate_prepared_rows,
    file_sha256,
    prepared_canvas_sha256,
    read_prediction_jsonl,
    write_metrics_json,
    write_prediction_jsonl,
)
from axolotl.integrations.decision.metrics import paired_bootstrap
from axolotl.integrations.decision.readers.hf import HFReader
from axolotl.model_support import get_model_support_for_cfg, resolve_model_support


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="validated Axolotl decision config")
    parser.add_argument("--output-dir", type=Path, required=True)
    arm = parser.add_mutually_exclusive_group(required=True)
    arm.add_argument("--base", action="store_true")
    arm.add_argument("--adapter", type=Path)
    parser.add_argument("--before-predictions", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--split",
        choices=("dev", "validation", "eval", "test", "calibration", "ood"),
        help="evaluate only sources explicitly declared for this original split",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="independent prepared canvases per packed reader call",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=16384,
        help="maximum concatenated prompt and canvas tokens per packed reader call",
    )
    parser.add_argument(
        "--hold-label-noise",
        action="store_true",
        help="hold initially noised labels through every K-step read",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="number of reads, bounded by diffusion.unroll.k_max",
    )
    parser.add_argument("--backend", choices=("dense", "flex_attention", "varlen"))
    parser.add_argument("--device")
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument(
        "--ordinal-metadata",
        action="store_true",
        help="emit source-ordered score metadata and report normalized RPS",
    )
    return parser


def _validate_scope(cfg: Any, *, hold_label_noise: bool = False) -> None:
    decision = _value(cfg, "decision")
    if decision is None:
        raise ValueError("decision evaluation requires decision settings")
    latent = _value(decision, "latent")
    mode = _value(latent, "mode", "none")
    if mode != "none":
        raise ValueError("latent slots are not supported")
    diffusion = _value(cfg, "diffusion")
    unroll = _value(diffusion, "unroll")
    k_max = _value(unroll, "k_max", 1)
    if k_max > 1 and not hold_label_noise:
        raise NotImplementedError(
            "decision K-step evaluation requires --hold-label-noise"
        )
    carry = _value(decision, "carry")
    if _value(carry, "enabled", False):
        raise NotImplementedError("decision evaluation does not yet support carry")


def _spec(cfg: Any):
    profile = resolve_model_support(get_model_support_for_cfg(cfg))
    spec = None if profile is None else profile.diffusion
    if spec is None:
        raise ValueError("decision evaluation requires a resolved native DiffusionSpec")
    return spec


def _reader(model: Any, cfg: Any, backend: str | None) -> HFReader:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    vocab_size = _value(text_config, "vocab_size")
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError("loaded native model must expose a positive vocabulary size")
    configured_backend = _value(cfg, "attn_implementation")
    if backend is not None:
        selected = backend
    elif configured_backend in {"dense", "flex_attention", "varlen"}:
        selected = configured_backend
    elif configured_backend in {None, "eager", "sdpa"}:
        selected = "dense"
    else:
        raise ValueError(
            "decision evaluation requires dense, flex_attention, or varlen backend"
        )
    return HFReader(
        vocab_size=vocab_size,
        mask_token_id=_value(config, "mask_token_id"),
        sliding_window=_value(text_config, "sliding_window"),
        attention_backend=selected,
        kernel_options=_value(cfg, "flex_attn_compile_kwargs"),
    )


def _provenance(
    cfg: Any,
    model: Any,
    *,
    adapter: Path | None,
    backend: str,
    seed: int,
    warmup: int,
    dataset_manifest: Mapping[str, Any],
    tokenizer: Any,
    preparation_audit: Path,
    prepared_canvas_digest: str,
    evaluation_selection: Mapping[str, Any],
    effective_overrides: Mapping[str, Any],
    read_precision: Mapping[str, Any],
    read_stats: Mapping[str, Any] | None = None,
    steps: int = 1,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    config_path = Path(str(_value(cfg, "axolotl_config_path")))
    return {
        "base_model": _value(cfg, "base_model"),
        "requested_model_revision": _value(cfg, "revision_of_model"),
        "resolved_model_revision": _value(config, "_commit_hash"),
        "adapter": None if adapter is None else str(adapter),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_manifest_sha256": _mapping_sha256(dataset_manifest),
        "preparation_audit": str(preparation_audit),
        "prepared_canvas_sha256": prepared_canvas_digest,
        "evaluation_selection": dict(evaluation_selection),
        "tokenizer": {
            "name_or_path": _value(tokenizer, "name_or_path"),
            "revision": _value(_value(tokenizer, "init_kwargs", {}), "revision"),
            "vocab_size": len(tokenizer),
        },
        "seed": seed,
        "steps": steps,
        "warmup": warmup,
        "backend": backend,
        "torch": torch.__version__,
        "read_precision": dict(read_precision),
        "reader_read_stats": None if read_stats is None else dict(read_stats),
        "effective_overrides": dict(effective_overrides),
    }


def _read_precision(model: Any) -> dict[str, Any]:
    parameter = next(model.parameters(), None)
    device = torch.device("cpu") if parameter is None else parameter.device
    autocast_enabled = torch.is_autocast_enabled(device.type)
    return {
        "inference_mode": torch.is_inference_mode_enabled(),
        "autocast_enabled": autocast_enabled,
        "autocast_dtype": (
            str(torch.get_autocast_dtype(device.type)) if autocast_enabled else None
        ),
        "device": str(device),
        "weight_dtype": None if parameter is None else str(parameter.dtype),
    }


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, default=str, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _evaluation_selection(
    manifest: Mapping[str, Any], requested_split: str | None
) -> dict[str, Any]:
    selected_split = manifest.get("selected_split", requested_split or "dev")
    return {
        "selected_split": selected_split,
        "source_inputs": manifest.get("selected_source_inputs", ()),
        "source_counts": {
            key: manifest[key]
            for key in ("input_rows", "grouped_rows", "eval_rows")
            if key in manifest
        },
        "source_names": manifest.get("eval_sources", ()),
        "drops": {
            "canvas_too_long": manifest.get("canvas_too_long", {}),
            "budget_drops": manifest.get("budget_drops", {}),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_cfg(str(args.config))
    _validate_scope(cfg, hold_label_noise=args.hold_label_noise)
    original_prepared_path = _value(cfg, "dataset_prepared_path")
    if args.base:
        cfg.lora_model_dir = None
        cfg.adapter = None
    else:
        assert args.adapter is not None
        if not args.adapter.is_dir():
            raise FileNotFoundError(f"adapter path is not a directory: {args.adapter}")
        if not _value(cfg, "adapter"):
            raise ValueError("--adapter requires an adapter-enabled Axolotl config")
        cfg.lora_model_dir = str(args.adapter)
    if not original_prepared_path:
        cfg.dataset_prepared_path = str(args.output_dir / "prepared")
    audit_dir = str(args.output_dir / "prepared")
    if isinstance(cfg, MutableMapping):
        cfg["_decision_preparation_audit_dir"] = audit_dir
    else:
        cfg._decision_preparation_audit_dir = audit_dir
    model, tokenizer, _processor = load_model_and_tokenizer(cfg=cfg, inference=True)
    if args.device is not None:
        model = model.to(torch.device(args.device))
    model.eval()
    if args.split is None:
        metadata = load_decision_datasets(cfg, tokenizer=tokenizer)
        dataset = metadata.eval_dataset
    else:
        dataset = load_decision_evaluation_dataset(cfg, tokenizer, args.split)
    if dataset is None:
        raise ValueError(
            "decision evaluation requires a nonempty prepared eval dataset"
        )
    spec = _spec(cfg)
    unroll = _value(_value(cfg, "diffusion"), "unroll")
    configured_steps = int(_value(unroll, "k_max", 1))
    steps = configured_steps if args.steps is None else args.steps
    if steps < 1 or steps > configured_steps:
        raise ValueError("--steps must be between 1 and diffusion.unroll.k_max")
    seed = int(_value(cfg, "seed", 0) if args.seed is None else args.seed)
    reader = _reader(model, cfg, args.backend)
    device = next(model.parameters()).device
    synchronize = (
        (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    )
    rows = tuple(dataset[index] for index in range(len(dataset)))
    with torch.inference_mode(), reader.autocast_context(device):
        read_precision = _read_precision(model)
        decision = _value(cfg, "decision")
        labels = _value(decision, "labels")
        run = evaluate_prepared_rows(
            reader,
            model,
            spec,
            rows,
            steps=steps,
            seed=seed,
            warmup=args.warmup,
            synchronize=synchronize,
            ordinal_metadata=args.ordinal_metadata,
            codebook=_value(labels, "codebook", "vendored26"),
            hold_label_noise=args.hold_label_noise,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
        )
    metrics = evaluate_artifacts(run)
    read_stats = dict(run.read_stats or {})
    if device.type == "cuda":
        read_stats["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    if args.before_predictions is not None:
        metrics["paired_bootstrap"] = paired_bootstrap(
            read_prediction_jsonl(args.before_predictions),
            run.rows,
            seed=seed,
            draws=args.bootstrap_draws,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = write_prediction_jsonl(args.output_dir / "predictions.jsonl", run)
    audit_path = (
        Path(str(dataset.manifest["preparation_audit"]))
        if args.split is not None and "preparation_audit" in dataset.manifest
        else preparation_audit_path(cfg)
    )
    if not audit_path.is_file():
        raise FileNotFoundError(f"prepared decision audit is missing: {audit_path}")
    provenance = _provenance(
        cfg,
        model,
        adapter=args.adapter,
        backend=reader.attention_backend,
        seed=seed,
        warmup=args.warmup,
        dataset_manifest=dataset.manifest,
        tokenizer=tokenizer,
        preparation_audit=audit_path,
        prepared_canvas_digest=prepared_canvas_sha256(run),
        evaluation_selection=_evaluation_selection(dataset.manifest, args.split),
        read_precision=read_precision,
        read_stats=read_stats,
        steps=steps,
        effective_overrides={
            "arm": "base" if args.base else "adapter",
            "adapter": None if args.adapter is None else str(args.adapter),
            "dataset_prepared_path": cfg.dataset_prepared_path,
            "configured_dataset_prepared_path": original_prepared_path,
            "ordinal_metadata": args.ordinal_metadata,
            "hold_label_noise": args.hold_label_noise,
            "batch_size": args.batch_size,
            "max_batch_tokens": args.max_batch_tokens,
            "latency_measurement": (
                "per-record read latency"
                if args.batch_size == 1
                else "packed-batch latency replicated for each record; not per-request latency"
            ),
            "label_codebook": _value(labels, "codebook", "vendored26"),
        },
    )
    provenance["predictions_sha256"] = file_sha256(prediction_path)
    metric_path = write_metrics_json(
        args.output_dir / "metrics.json", metrics, provenance
    )
    print(
        json.dumps({"predictions": str(prediction_path), "metrics": str(metric_path)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
