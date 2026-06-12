# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Translate an axolotl config into a TorchSpec flat-args ``Namespace``.

TorchSpec owns the whole run (Ray + Mooncake + inference engines + FSDP training
actors); axolotl's role is to validate the YAML and produce the exact
``argparse.Namespace`` that ``torchspec.train_entry.train_async_no_generation``
expects. We reuse TorchSpec's own ``load_config`` / ``config_to_flat_args`` and
replicate the post-processing that ``parse_config`` applies, so a translated run
is byte-for-byte equivalent to ``python -m torchspec.train_entry``.
"""

from __future__ import annotations

import argparse
from typing import Any

from axolotl.integrations.torchspec.args import TorchSpecArgs
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# axolotl chat_template name -> TorchSpec dataset.chat_template name
_CHAT_TEMPLATE_MAP = {
    "llama3": "llama3",
    "llama3_2": "llama3",
    "qwen3": "qwen",
    "qwen2": "qwen",
    "qwen_25": "qwen",
    "chatml": "qwen",
}

_BACKEND_MAP = {"sgl": "sglang", "vllm": "vllm", "hf": "hf"}

_TORCHSPEC_IMPORT_ERROR = (
    "TorchSpec is not installed. Install the optional dependency with "
    "`pip install -e '.[torchspec]'` (also requires Ray, Mooncake, and an "
    "SGLang/vLLM backend). See src/axolotl/integrations/torchspec/README.md."
)


def _require_torchspec():
    try:
        from torchspec.config.train_config import (  # noqa: F401
            config_to_flat_args,
            load_config,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without torchspec
        raise ImportError(_TORCHSPEC_IMPORT_ERROR) from exc


def _resolve_batch_size(args: argparse.Namespace) -> None:
    """Derive dp_size / per_dp_rank_batch_size / global_batch_size.

    Inlined from ``torchspec.train_entry._resolve_batch_size`` so we don't import
    the heavy Ray-laden entry module just to flatten a config. Sequence-parallel
    (USP) is delegated to TorchSpec at launch; here we only support the default
    (non-USP) path, which is all the translator needs.
    """
    world_size = args.training_num_nodes * args.training_num_gpus_per_node
    if getattr(args, "attention_backend", None) == "usp":
        sp_size = getattr(args, "sp_ulysses_size", 1) * getattr(args, "sp_ring_size", 1)
        if sp_size <= 0:
            raise ValueError(f"USP requires positive sp_size, got {sp_size}")
        if world_size % sp_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by USP sp_size ({sp_size})"
            )
        dp_size = getattr(args, "dp_size", None) or (world_size // sp_size)
        args.dp_size = dp_size
        args.sp_size = sp_size
        args.per_dp_rank_batch_size = 1
    else:
        dp_size = getattr(args, "dp_size", None) or world_size
        args.dp_size = dp_size
        sp_size = getattr(args, "sp_size", None) or 1
        args.per_dp_rank_batch_size = args.micro_batch_size * sp_size

    accumulation_steps = getattr(args, "draft_accumulation_steps", 1)
    args.global_batch_size = args.per_dp_rank_batch_size * dp_size * accumulation_steps


def _get_spec_args(cfg: DictDefault) -> TorchSpecArgs:
    """Return the validated ``speculator`` block from an axolotl config."""
    spec = cfg.get("speculator")
    if spec is None:
        raise ValueError(
            "TorchSpec training requires a `speculator:` block in the axolotl config."
        )
    if isinstance(spec, TorchSpecArgs):
        return spec
    if isinstance(spec, dict):
        return TorchSpecArgs(**spec)
    return TorchSpecArgs(**dict(spec))


def _resolve_chat_template(cfg: DictDefault, spec: TorchSpecArgs) -> str:
    if spec.chat_template:
        return spec.chat_template
    axo_template = cfg.get("chat_template")
    if axo_template and axo_template in _CHAT_TEMPLATE_MAP:
        return _CHAT_TEMPLATE_MAP[axo_template]
    if axo_template:
        raise ValueError(
            f"chat_template '{axo_template}' has no TorchSpec mapping. Set "
            f"`speculator.chat_template` explicitly (one of: "
            f"{sorted(set(_CHAT_TEMPLATE_MAP.values()))})."
        )
    raise ValueError(
        "No chat_template found. Set top-level `chat_template` or "
        "`speculator.chat_template`."
    )


def _first_dataset_path(datasets: Any) -> str:
    if not datasets:
        raise ValueError(
            "TorchSpec training requires at least one entry in `datasets`."
        )
    first = datasets[0]
    path = (
        first.get("path") if isinstance(first, dict) else getattr(first, "path", None)
    )
    if not path:
        raise ValueError("First dataset entry is missing a `path`.")
    return path


def _prune_none(obj: Any) -> Any:
    """Recursively drop keys whose value is None so they don't clobber schema defaults."""
    if isinstance(obj, dict):
        return {k: _prune_none(v) for k, v in obj.items() if v is not None}
    return obj


def build_overrides(
    cfg: DictDefault,
    train_path: str | None = None,
    eval_path: str | None = None,
    draft_config_path: str | None = None,
) -> dict[str, Any]:
    """Build the nested override dict matching TorchSpec's ``Config`` sections.

    ``train_path``/``eval_path`` override the dataset paths (set when the
    standardization bridge has produced normalized JSONL); ``draft_config_path``
    overrides the draft-config path (set when generated from ``draft_*`` knobs).
    When omitted, the first axolotl dataset path and ``speculator.draft_model_config``
    are used as-is. Kept pure (no I/O) so it is safe for ``--dry-run`` and tests.
    """
    spec = _get_spec_args(cfg)

    if spec.inference_num_gpus is None or spec.training_num_gpus is None:
        raise ValueError(
            "speculator.inference_num_gpus and speculator.training_num_gpus must "
            "both be set (the inference/training GPU split)."
        )

    backend = _BACKEND_MAP[spec.inference_engine]
    output_dir = spec.output_dir or cfg.get("output_dir") or "./outputs/speculator"
    cache_dir = spec.cache_dir or f"{output_dir.rstrip('/')}/cache"

    train_data_path = train_path or _first_dataset_path(cfg.get("datasets"))
    if eval_path is None and cfg.get("test_datasets"):
        eval_path = _first_dataset_path(cfg.get("test_datasets"))

    # axolotl stores these as float; TorchSpec's structured schema types them as
    # int, and OmegaConf rejects a float on merge.
    def _as_int(key: str) -> int | None:
        val = cfg.get(key)
        return int(val) if val is not None else None

    engine_cfg = {
        "tp_size": spec.inference_tp_size,
        "mem_fraction_static": spec.mem_fraction_static,
    }

    overrides: dict[str, Any] = {
        "model": {
            "target_model_path": cfg.get("base_model"),
            "target_model_backend": backend,
            "trust_remote_code": bool(cfg.get("trust_remote_code")),
            "draft_model_config": draft_config_path or spec.draft_model_config,
        },
        "dataset": {
            "train_data_path": train_data_path,
            "eval_data_path": eval_path,
            "chat_template": _resolve_chat_template(cfg, spec),
            "prompt_key": spec.prompt_key,
        },
        "training": {
            "max_seq_length": _as_int("sequence_len"),
            "learning_rate": cfg.get("learning_rate"),
            "num_epochs": _as_int("num_epochs"),
            "num_train_steps": _as_int("max_steps"),
            "micro_batch_size": _as_int("micro_batch_size"),
            "warmup_ratio": cfg.get("warmup_ratio"),
            "max_grad_norm": cfg.get("max_grad_norm"),
            "weight_decay": cfg.get("weight_decay"),
            "seed": cfg.get("seed"),
            "gradient_checkpointing": cfg.get("gradient_checkpointing"),
            "ttt_length": spec.ttt_length,
            "ploss_weights": spec.ploss_weights,
            "draft_accumulation_steps": spec.draft_accumulation_steps,
            "max_concurrent_batches": spec.max_concurrent_batches,
            "attention_backend": spec.attention_backend,
            "fsdp_strategy": spec.fsdp_strategy,
            "train_with_decode": spec.train_with_decode,
            "training_num_gpus_per_node": spec.training_num_gpus,
            "training_num_nodes": spec.training_num_nodes,
        },
        "inference": {
            "inference_engine_type": spec.inference_engine,
            "inference_num_gpus": spec.inference_num_gpus,
            "inference_num_gpus_per_engine": spec.inference_num_gpus_per_engine,
            "inference_num_gpus_per_node": spec.inference_num_gpus_per_node,
            "inference_batch_size": spec.inference_batch_size,
            "aux_hidden_states_layers": spec.aux_hidden_states_layers,
            "last_hidden_states_prenorm": spec.last_hidden_states_prenorm,
            "sglang": engine_cfg if spec.inference_engine == "sgl" else None,
            "vllm": engine_cfg if spec.inference_engine == "vllm" else None,
        },
        "mooncake": {
            "protocol": spec.mooncake_protocol,
            "global_segment_size": spec.mooncake_global_segment_size,
            "local_buffer_size": spec.mooncake_local_buffer_size,
            "device_name": spec.mooncake_device_name,
            "master_server_address": spec.mooncake_master_server_address,
            "metadata_server": spec.mooncake_metadata_server,
        },
        "output_dir": output_dir,
        "cache_dir": cache_dir,
    }
    return _prune_none(overrides)


def build_torchspec_args(
    cfg: DictDefault, extra_overrides: list[str] | None = None
) -> argparse.Namespace:
    """Translate an axolotl config into a TorchSpec flat-args ``Namespace``.

    Args:
        cfg: validated axolotl config (must contain a ``speculator`` block).
        extra_overrides: optional OmegaConf dotlist overrides (e.g.
            ``["training.num_train_steps=10"]``) applied last, matching the
            unknown-args passthrough of ``torchspec.train_entry``.
    """
    _require_torchspec()
    from omegaconf import OmegaConf
    from torchspec.config.train_config import config_to_flat_args, load_config

    spec = _get_spec_args(cfg)

    train_path = eval_path = None
    if spec.prepare_dataset:
        from axolotl.integrations.torchspec.dataset_bridge import prepare_datasets

        train_path, eval_path = prepare_datasets(cfg)

    from axolotl.integrations.torchspec.draft_config import build_draft_model_config

    output_dir = spec.output_dir or cfg.get("output_dir") or "./outputs/speculator"
    draft_config_path = build_draft_model_config(cfg, spec, output_dir)

    base = OmegaConf.create(
        build_overrides(
            cfg,
            train_path=train_path,
            eval_path=eval_path,
            draft_config_path=draft_config_path,
        )
    )
    config = load_config(config_path=None, base_config=base, cli_args=extra_overrides)

    flat_args = config_to_flat_args(config)
    flat_args.rank = 0
    flat_args.world_size = (
        flat_args.training_num_nodes * flat_args.training_num_gpus_per_node
    )
    defaults = {
        "colocate": False,
        "debug_train_only": False,
        "debug_inference_only": False,
        "dp_size": None,
        "save_debug_train_data": None,
    }
    for key, value in defaults.items():
        if not hasattr(flat_args, key) or getattr(flat_args, key) is None:
            setattr(flat_args, key, value)

    _resolve_batch_size(flat_args)
    return flat_args
