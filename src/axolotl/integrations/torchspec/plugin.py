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

"""Axolotl plugin for training EAGLE-3 speculators via TorchSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, Trainer

    from axolotl.common.datasets import TrainDatasetMeta

LOG = get_logger(__name__)


class TorchSpecPlugin(BasePlugin):
    """Drive TorchSpec EAGLE-3 speculator training from an axolotl config.

    TorchSpec owns the entire run (Ray-orchestrated inference engines, Mooncake
    hidden-state transfer, and FSDP training workers) and must be the sole Ray
    driver. The recommended entry point is the dedicated CLI command::

        axolotl train-speculator config.yaml

    This plugin additionally lets the standard ``axolotl train`` verb dispatch to
    TorchSpec when run single-process::

        plugins:
          - axolotl.integrations.torchspec.TorchSpecPlugin

        axolotl train --launcher python config.yaml

    It hands the whole run off (Hatchery-style): the local model and dataset are
    stubbed out — TorchSpec loads + tokenizes the data and runs training itself.
    Multi-process ``accelerate``/``torchrun`` launches are rejected at train time.
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.torchspec.args.TorchSpecArgsMixin"

    def register(self, cfg: dict):
        # TorchSpec ignores axolotl's tokenized columns; keep the stub dataset intact.
        if cfg.get("remove_unused_columns") is None:
            cfg["remove_unused_columns"] = False

    def load_datasets(
        self, cfg: DictDefault, preprocess: bool = False
    ) -> "TrainDatasetMeta":
        """Skip axolotl tokenization; TorchSpec consumes raw conversations.

        Returns a 1-row stub so the trainer can be constructed. Under
        ``axolotl preprocess`` we materialize the standardized conversations
        JSONL (the part axolotl is genuinely reused for).
        """
        from datasets import Dataset

        from axolotl.common.datasets import TrainDatasetMeta

        if preprocess:
            from axolotl.integrations.torchspec.dataset_bridge import prepare_datasets

            train_path, _ = prepare_datasets(cfg)
            LOG.info("TorchSpec dataset prepared at %s", train_path)

        stub = Dataset.from_dict(
            {"input_ids": [[0, 0]], "attention_mask": [[1, 1]], "labels": [[0, 0]]}
        )
        return TrainDatasetMeta(
            train_dataset=stub, eval_dataset=None, total_num_steps=1
        )

    def pre_model_load(self, cfg: DictDefault):
        """Avoid loading the (large) target model in the driver process."""
        import torch
        from transformers import AutoConfig, PreTrainedModel

        from axolotl.loaders import ModelLoader

        def _stub_build_model(loader_self) -> bool:
            config = AutoConfig.from_pretrained(
                loader_self.cfg.base_model,
                trust_remote_code=loader_self.cfg.get("trust_remote_code", False),
            )

            class _Stub(PreTrainedModel):
                config_class = type(config)
                _no_split_modules: list[str] = []
                supports_gradient_checkpointing = False

                def __init__(self, cfg):
                    super().__init__(cfg)
                    self.embed_tokens = torch.nn.Embedding(
                        getattr(cfg, "vocab_size", 32000), 1
                    )

                def get_input_embeddings(self):
                    return self.embed_tokens

                def set_input_embeddings(self, value):
                    pass

                def get_output_embeddings(self):
                    return None

            loader_self.model = _Stub(config)
            return True

        ModelLoader._build_model = _stub_build_model  # type: ignore[method-assign,assignment]

    def get_trainer_cls(self, cfg: DictDefault) -> "type[Trainer] | None":
        from .trainer import TorchSpecLauncherTrainer

        return TorchSpecLauncherTrainer

    def post_trainer_create(self, cfg: DictDefault, trainer: "Trainer"):
        from .trainer import TorchSpecLauncherTrainer

        if isinstance(trainer, TorchSpecLauncherTrainer):
            trainer._axolotl_cfg = cfg

    def post_train(self, cfg: DictDefault, model: "PreTrainedModel"):
        LOG.info(
            "TorchSpec: draft checkpoints are written under output_dir by the "
            "training workers; convert with TorchSpec's tools/convert_to_hf.py."
        )
