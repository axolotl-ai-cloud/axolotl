# Copyright 2025 Axolotl AI. All rights reserved.
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
Plugin for MixLoRA.
"""

import os

from axolotl.integrations.base import AdapterCapabilities, BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MixLoraPlugin(BasePlugin):
    """
    Plugin for MixLoRA support in Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.mixlora.args.MixLoraArgs"

    def get_adapter_capabilities(self):
        return [
            AdapterCapabilities(name="mixlora", lora_like=True, supports_merge=False)
        ]

    def load_adapter(self, model, cfg, inference=False, config_only=False):
        if cfg.adapter != "mixlora":
            return None

        import safetensors.torch

        from axolotl.loaders.adapter import load_lora

        from .constants import MIXLORA_WEIGHTS_NAME
        from .model import load_mixlora_state_dict
        from .patching import patch_model_with_mixlora

        # First, load standard LoRA for attention layers (q, k, v, o projections)
        peft_model, lora_config = load_lora(
            model, cfg, inference=inference, config_only=config_only
        )
        if config_only:
            return peft_model, lora_config

        # Then, apply MixLoRA patching to FFN layers (router + LoRA experts)
        patch_model_with_mixlora(peft_model, cfg)

        if cfg.lora_model_dir:
            weights_path = os.path.join(cfg.lora_model_dir, MIXLORA_WEIGHTS_NAME)
            if os.path.exists(weights_path):
                load_mixlora_state_dict(
                    peft_model,
                    safetensors.torch.load_file(weights_path),
                    strict=True,
                )
                LOG.info("Loaded MixLoRA router/expert weights from checkpoint")
            else:
                LOG.warning(
                    f"MixLoRA checkpoint is missing {MIXLORA_WEIGHTS_NAME}; "
                    "router/expert weights were initialized from scratch."
                )

        return peft_model, lora_config

    def pre_train(self, cfg, trainer, resume_from_checkpoint=None):
        # HF Trainer's own checkpoint restore doesn't know about MixLoRA's
        # router/expert sidecar file, so it never gets loaded on resume.
        # Load it explicitly before training resumes.
        if cfg.adapter != "mixlora" or not resume_from_checkpoint:
            return

        import safetensors.torch

        from .constants import MIXLORA_WEIGHTS_NAME
        from .model import load_mixlora_state_dict

        weights_path = os.path.join(resume_from_checkpoint, MIXLORA_WEIGHTS_NAME)
        if os.path.exists(weights_path):
            load_mixlora_state_dict(
                trainer.model,
                safetensors.torch.load_file(weights_path),
                strict=True,
            )
            LOG.info("Loaded MixLoRA router/expert weights from checkpoint for resume")
        else:
            LOG.warning(
                "Resuming MixLoRA training but checkpoint is missing "
                f"{MIXLORA_WEIGHTS_NAME}; router/expert weights were not restored."
            )

    def post_model_save(self, cfg, model, output_dir):
        # PEFT's save_pretrained doesn't know about MixLoRA's router/expert
        # weights, so MixLoraTrainer._save is the only place that normally
        # writes the sidecar file. That method only runs for trainer-driven
        # checkpoint saves, not the final save, so write the sidecar here too.
        #
        # This hook only fires on the plain (non-FSDP, non-deepspeed-zero3)
        # save path, where `model` is a fully materialized single-process copy
        # and mixlora_state_dict(model) is safe to call directly. Under FSDP,
        # each rank only holds its local shard, so calling this there would
        # silently write a corrupted, partial sidecar instead of correctly
        # doing nothing — final MixLoRA save isn't supported for
        # FSDP/deepspeed-zero3 yet.
        if cfg.adapter != "mixlora":
            return

        import safetensors.torch

        from .constants import MIXLORA_WEIGHTS_NAME
        from .model import mixlora_state_dict

        state = mixlora_state_dict(model)
        if not state:
            return

        cpu_state = {key: value.detach().cpu() for key, value in state.items()}
        safetensors.torch.save_file(
            cpu_state,
            os.path.join(output_dir, MIXLORA_WEIGHTS_NAME),
            metadata={"format": "pt"},
        )

    def get_trainer_cls(self, cfg):
        if cfg.adapter == "mixlora" and cfg.rl is None:
            from .trainer import MixLoraTrainer

            return MixLoraTrainer
        return None
