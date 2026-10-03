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
Plugin for on-policy distillation (GKD / OPD) in Axolotl.

The student generates its own rollouts and a teacher provides dense, token-level
supervision on those student-visited states — the "missing middle" between GRPO
(on-policy, sparse reward) and off-policy KD (dense KL, fixed prefixes).
"""

from axolotl.integrations.base import BasePlugin

from .args import GKDArgs as GKDArgs


class GKDPlugin(BasePlugin):
    """Plugin for GKD / on-policy distillation support in Axolotl."""

    def get_input_args(self):
        return "axolotl.integrations.gkd.GKDArgs"

    def get_training_args_mixin(self):
        return "axolotl.integrations.gkd.args.GKDTrainingArgsMixin"

    def get_trainer_cls(self, cfg):
        if cfg.gkd_trainer:
            from .trainer import AxolotlGKDTrainer

            return AxolotlGKDTrainer
        return None

    def get_training_args(self, cfg):
        if not cfg.gkd_trainer:
            return None
        return {
            "gkd_teacher": cfg.gkd_teacher,
            "gkd_teacher_init_kwargs": cfg.gkd_teacher_init_kwargs,
            "gkd_lmbda": cfg.gkd_lmbda,
            "gkd_seq_kd": cfg.gkd_seq_kd,
            "gkd_max_new_tokens": cfg.gkd_max_new_tokens,
            "gkd_top_k": cfg.gkd_top_k,
            "gkd_beta": cfg.gkd_beta,
            "gkd_divergence": cfg.gkd_divergence,
            "gkd_temperature": cfg.gkd_temperature,
        }
