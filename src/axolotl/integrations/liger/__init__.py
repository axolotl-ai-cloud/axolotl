# Copyright 2024 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""
Module for the Plugin for LIGER integraton with Axolotl.

Liger Kernel is the collection of Triton-native kernels for LLM Training.
It is designed to be performant, correct, and light-weight.
"""
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.model.llama import lce_forward
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from axolotl.integrations.base import BasePlugin


class LigerPlugin(BasePlugin):
    """
    Plugin for LIGER integraton with Axolotl.
    """

    def pre_model_load(self, cfg):
        if cfg.model_config_type == "llama":
            from transformers.models.llama import modeling_llama

            if cfg.liger_rope:
                modeling_llama.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_rms_norm:
                modeling_llama.LlamaRMSNorm = LigerRMSNorm
            if cfg.liger_swiglu:
                modeling_llama.LlamaMLP = LigerSwiGLUMLP
            if cfg.liger_cross_entropy:
                modeling_llama.CrossEntropyLoss = LigerCrossEntropyLoss
            elif cfg.liger_fused_linear_cross_entropy:
                modeling_llama.LlamaForCausalLM.forward = lce_forward
