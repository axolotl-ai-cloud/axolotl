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

"""Generate a TorchSpec EAGLE-3 draft-model config from `speculator.draft_*` knobs.

TorchSpec only auto-generates a draft config (a 1-layer Llama EAGLE-3 head) when
no `draft_model_config` path is given. To let users tune the draft architecture
from the axolotl config without hand-writing JSON, we take TorchSpec's
target-derived base config and overlay the requested overrides, then write it to
a JSON file that is passed through as `model.draft_model_config`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axolotl.integrations.torchspec.args import TorchSpecArgs
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_DRAFT_OVERRIDE_FIELDS = (
    "draft_num_hidden_layers",
    "draft_hidden_size",
    "draft_intermediate_size",
    "draft_vocab_size",
    "draft_architecture",
    "draft_config_overrides",
)


def has_draft_overrides(spec: TorchSpecArgs) -> bool:
    return any(getattr(spec, f) is not None for f in _DRAFT_OVERRIDE_FIELDS)


def apply_draft_overrides(base: dict[str, Any], spec: TorchSpecArgs) -> dict[str, Any]:
    config = dict(base)
    if spec.draft_architecture:
        config["architectures"] = [spec.draft_architecture]
    if spec.draft_num_hidden_layers is not None:
        config["num_hidden_layers"] = spec.draft_num_hidden_layers
    if spec.draft_hidden_size is not None:
        config["hidden_size"] = spec.draft_hidden_size
    if spec.draft_intermediate_size is not None:
        config["intermediate_size"] = spec.draft_intermediate_size
    if spec.draft_vocab_size is not None:
        config["draft_vocab_size"] = spec.draft_vocab_size
    if spec.draft_config_overrides:
        config.update(spec.draft_config_overrides)
    return config


def build_draft_model_config(
    cfg: DictDefault, spec: TorchSpecArgs, output_dir: str
) -> str | None:
    """Write a generated draft config and return its path, or None if not needed.

    Returns None when the user gave an explicit ``draft_model_config`` path
    (theirs wins) or set no ``draft_*`` knobs (TorchSpec auto-generates).
    """
    if spec.draft_model_config or not has_draft_overrides(spec):
        return None

    from torchspec.config.utils import generate_draft_model_config

    base = generate_draft_model_config(
        target_model_path=cfg.get("base_model"),
        cache_dir=cfg.get("model_download_dir"),
    )
    draft_config = apply_draft_overrides(base, spec)

    out_path = Path(output_dir) / "draft_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(draft_config, f, indent=2)
    LOG.info(
        "Generated draft model config (%s layers, arch=%s) -> %s",
        draft_config.get("num_hidden_layers"),
        draft_config.get("architectures"),
        out_path,
    )
    return str(out_path)
