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
Standardize axolotl ``datasets:`` into a conversations dataset for TorchSpec.

TorchSpec does its own (EAGLE-3-correct) tokenization and assistant masking, so
we don't reuse axolotl's *tokenization* — only its dataset *loading* (the broad
local/HF-Hub/cloud/data_files/split support and multi-dataset merging). Each row
is normalized to ``{"conversations": [{"role", "content"}, ...]}`` using the same
``message_property_mappings`` / ``roles`` / ``field_system`` knobs as axolotl's
chat_template strategy, then written as JSONL that TorchSpec ingests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# axolotl's default source-role -> canonical-role aliases (see ChatTemplatePrompter)
_DEFAULT_ROLES = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
}
_DEFAULT_PROPERTY_MAPPINGS = {"role": "role", "content": "content"}


def _invert_roles(roles: dict[str, list[str]] | None) -> dict[str, str]:
    """axolotl config stores ``{canonical: [sources]}``; invert to ``{source: canonical}``."""
    if not roles:
        return dict(_DEFAULT_ROLES)
    return {src: canonical for canonical, sources in roles.items() for src in sources}


def _content_to_str(content: Any) -> str:
    """Coerce message content to a plain string (speculator training is text-only)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI-style content parts: keep text segments
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type", "text") == "text"
        ]
        return "".join(parts) if parts else json.dumps(content)
    return str(content)


def _normalize_row(
    row: dict,
    *,
    field_messages: str,
    field_system: str,
    property_mappings: dict[str, str],
    role_map: dict[str, str],
    drop_system_message: bool,
) -> dict | None:
    messages = row.get(field_messages)
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return None
    if not isinstance(messages, list) or not messages:
        return None

    role_key = property_mappings.get("role", "role")
    content_key = property_mappings.get("content", "content")

    turns: list[dict[str, str]] = []

    # inject a top-level system field if the first message isn't a system turn
    first_role = role_map.get(messages[0].get(role_key), messages[0].get(role_key))
    if first_role != "system" and row.get(field_system):
        turns.append({"role": "system", "content": str(row[field_system])})

    for message in messages:
        if not isinstance(message, dict):
            return None
        raw_role = message.get(role_key)
        if raw_role is None:
            return None
        role = role_map.get(raw_role, raw_role)
        turns.append(
            {"role": role, "content": _content_to_str(message.get(content_key))}
        )

    if drop_system_message and turns and turns[0]["role"] == "system":
        turns = turns[1:]

    return {"conversations": turns}


def _standardize_one(ds_cfg: DictDefault, use_auth_token: bool):
    """Load + normalize a single axolotl dataset entry to a conversations dataset."""
    from axolotl.utils.data.shared import load_dataset_with_config

    dataset = load_dataset_with_config(ds_cfg, use_auth_token=use_auth_token)

    field_messages = ds_cfg.get("field_messages") or "messages"
    field_system = ds_cfg.get("field_system") or "system"
    property_mappings = (
        ds_cfg.get("message_property_mappings") or _DEFAULT_PROPERTY_MAPPINGS
    )
    role_map = _invert_roles(ds_cfg.get("roles"))
    drop_system = bool(ds_cfg.get("drop_system_message"))

    def _map(row):
        normalized = _normalize_row(
            row,
            field_messages=field_messages,
            field_system=field_system,
            property_mappings=property_mappings,
            role_map=role_map,
            drop_system_message=drop_system,
        )
        # datasets.map can't drop rows; mark invalid ones for a follow-up filter
        return normalized or {"conversations": []}

    return dataset.map(_map, remove_columns=dataset.column_names, desc="standardize")


def _write_jsonl(dataset, path: Path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            convs = row.get("conversations")
            if not convs:
                continue
            f.write(json.dumps({"conversations": convs}, ensure_ascii=False) + "\n")
            n += 1
    return n


def standardize_datasets(
    cfg: DictDefault, datasets_key: str, out_path: Path
) -> str | None:
    """Standardize ``cfg[datasets_key]`` into a conversations JSONL at ``out_path``.

    Returns the path string, or None if there are no datasets under the key.
    """
    from datasets import concatenate_datasets

    entries = cfg.get(datasets_key)
    if not entries:
        return None

    use_auth_token = bool(cfg.get("hf_use_auth_token"))
    standardized = [
        _standardize_one(DictDefault(entry), use_auth_token) for entry in entries
    ]
    merged = (
        standardized[0]
        if len(standardized) == 1
        else concatenate_datasets(standardized)
    )
    if cfg.get("shuffle_merged_datasets", True) and not cfg.get("curriculum_sampling"):
        merged = merged.shuffle(seed=cfg.get("seed") or 42)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = _write_jsonl(merged, out_path)
    LOG.info("Standardized %d conversations -> %s", written, out_path)
    if written == 0:
        raise ValueError(
            f"No valid conversations produced from `{datasets_key}`. Check "
            f"field_messages / message_property_mappings / roles."
        )
    return str(out_path)


def prepare_datasets(cfg: DictDefault) -> tuple[str, str | None]:
    """Run the standardization bridge; return (train_path, eval_path|None)."""
    output_dir = cfg.get("output_dir") or "./outputs/speculator"
    data_dir = Path(output_dir) / "torchspec_data"

    train_path = standardize_datasets(cfg, "datasets", data_dir / "train.jsonl")
    if train_path is None:
        raise ValueError("TorchSpec training requires at least one `datasets` entry.")
    eval_path = standardize_datasets(cfg, "test_datasets", data_dir / "eval.jsonl")
    return train_path, eval_path
