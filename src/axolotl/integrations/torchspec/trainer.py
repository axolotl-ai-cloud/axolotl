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

"""Launcher trainer that hands an `axolotl train` run off to TorchSpec."""

from __future__ import annotations

import os
from typing import Any, Optional

from transformers.trainer_utils import TrainOutput

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _detected_world_size() -> int:
    """Number of distributed processes, if launched under accelerate/torchrun."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:  # pragma: no cover - defensive
        pass
    for var in ("WORLD_SIZE", "ACCELERATE_NUM_PROCESSES"):
        val = os.environ.get(var)
        if val and val.isdigit():
            return int(val)
    return 1


class TorchSpecLauncherTrainer(AxolotlTrainer):
    """Replaces the HF training loop with a TorchSpec launch.

    TorchSpec is the sole Ray driver for the run, so this trainer must execute as
    a single process — i.e. ``axolotl train --launcher python config.yaml``, not
    under multi-process ``accelerate``/``torchrun``. The dataset/model that
    axolotl built are ignored; TorchSpec loads + tokenizes the data itself.
    """

    _axolotl_cfg: Optional[DictDefault] = None

    def train(self, *args: Any, **kwargs: Any) -> TrainOutput:  # noqa: D102
        world_size = _detected_world_size()
        if world_size > 1:
            raise RuntimeError(
                "TorchSpec must run as the sole Ray driver process, but this run "
                f"was launched with WORLD_SIZE={world_size}. Re-run with "
                "`axolotl train --launcher python <config>` (single process), or "
                "use `axolotl train-speculator <config>`."
            )

        cfg = self._axolotl_cfg
        if cfg is None:
            raise RuntimeError(
                "TorchSpecLauncherTrainer._axolotl_cfg not set; the "
                "TorchSpecPlugin must be registered."
            )

        from axolotl.integrations.torchspec.translate import build_torchspec_args

        ts_args = build_torchspec_args(cfg)

        from torchspec.train_entry import train_async_no_generation

        LOG.info(
            "Launching TorchSpec EAGLE-3 training for target=%s "
            "(%s inference + %s training GPUs).",
            ts_args.target_model_path,
            ts_args.inference_num_gpus,
            ts_args.training_num_gpus_per_node * ts_args.training_num_nodes,
        )
        train_async_no_generation(ts_args)

        return TrainOutput(global_step=0, training_loss=0.0, metrics={})
