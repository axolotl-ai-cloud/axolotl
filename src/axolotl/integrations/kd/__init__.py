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
Plugin init to add KD support to Axolotl.
"""

from typing import Any

from transformers import Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.kd.callbacks import KDTemperatureSchedulerCallback

from .args import KDArgs as KDArgs


class KDPlugin(BasePlugin):
    """
    Plugin for KD support in Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.kd.KDArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.kd.args.KDTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.kd_trainer:
            from .trainer import AxolotlKDTrainer

            return AxolotlKDTrainer
        return None

    def get_training_args(self, cfg):
        return {
            "kd_ce_alpha": cfg.kd_ce_alpha,
            "kd_alpha": cfg.kd_alpha,
            "kd_temperature": cfg.kd_temperature,
            "kd_beta": cfg.kd_beta,
            "kd_normalize_topk": cfg.kd_normalize_topk,
        }

    def get_collator_cls_and_kwargs(self, cfg, is_eval=False):
        if not cfg.kd_trainer:
            return None, None

        from .collator import DataCollatorForKD, KDBatchSamplerDataCollatorForSeq2Seq

        use_batch_sampler_collator = False
        if is_eval is False and cfg.sample_packing:
            use_batch_sampler_collator = True
        if cfg.eval_sample_packing and is_eval:
            use_batch_sampler_collator = True

        if cfg.kd_online_server_base_url:
            from .collator_online_teacher import OnlineTeacherCollator

            return OnlineTeacherCollator, {
                "kd_online_server_base_url": cfg.kd_online_server_base_url,
                "kd_online_topk": cfg.kd_online_topk,
                "kd_temperature": cfg.kd_temperature,
                "kd_online_server": cfg.kd_online_server,
                "kd_online_timeout": cfg.kd_online_timeout,
                "kd_normalize_topk": cfg.kd_normalize_topk,
            }

        if use_batch_sampler_collator:
            return KDBatchSamplerDataCollatorForSeq2Seq, {}
        return DataCollatorForKD, {}

    def pre_model_load(self, cfg):
        from .kernels.models import apply_kernel

        apply_kernel(cfg.model_config_type)

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Trainer) -> list:
        """
        Adds temp scheduler callback to the Trainer instance.

        Args:
            cfg (Any): Configuration object containing the sparse recipe.
            trainer (Trainer): Huggingface Trainer instance.

        Returns:
            list: List containing the configured callback instances.
        """
        if cfg.kd_temperature_min is not None and cfg.kd_online_server_base_url:
            callback = KDTemperatureSchedulerCallback(
                cfg.kd_temperature,
                cfg.kd_temperature_min,
                trainer,
            )
            return [callback]

        return []
