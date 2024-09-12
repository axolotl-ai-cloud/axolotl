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
Module for the Plugin for LIGER integraton with Axolotl.

Liger Kernel is the collection of Triton-native kernels for LLM Training.
It is designed to be performant, correct, and light-weight.
"""
import logging
import sys
from functools import partial

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from axolotl.integrations.base import BasePlugin

from .args import LigerArgs  # pylint: disable=unused-import. # noqa: F401


class LigerPlugin(BasePlugin):
    """
    Plugin for LIGER integraton with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.liger.LigerArgs"

    def pre_model_load(self, cfg):
        if cfg.model_config_type == "llama":
            from liger_kernel.transformers.model.llama import (
                lce_forward as llama_lce_forward,
            )
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
                modeling_llama.LlamaForCausalLM.forward = llama_lce_forward

        elif cfg.model_config_type == "mistral":
            from liger_kernel.transformers.model.mistral import (
                lce_forward as mistral_lce_forward,
            )
            from transformers.models.mistral import modeling_mistral

            if cfg.liger_rope:
                modeling_mistral.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_rms_norm:
                modeling_mistral.MistralRMSNorm = LigerRMSNorm
            if cfg.liger_swiglu:
                modeling_mistral.MistralMLP = LigerSwiGLUMLP
            if cfg.liger_cross_entropy:
                modeling_mistral.CrossEntropyLoss = LigerCrossEntropyLoss
            if cfg.liger_fused_linear_cross_entropy:
                modeling_mistral.MistralForCausalLM.forward = mistral_lce_forward

        elif cfg.model_config_type == "gemma":
            from liger_kernel.transformers.model.gemma import (
                lce_forward as gemma_lce_forward,
            )
            from transformers.models.gemma import modeling_gemma

            if cfg.liger_rope:
                modeling_gemma.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_rms_norm:
                modeling_gemma.GemmaRMSNorm = partial(
                    LigerRMSNorm, offset=1.0, init_fn="zeros", casting_mode="gemma"
                )
            if cfg.liger_swiglu:
                modeling_gemma.GemmaMLP = LigerGEGLUMLP
            if cfg.liger_cross_entropy:
                modeling_gemma.CrossEntropyLoss = LigerCrossEntropyLoss
            if cfg.liger_fused_linear_cross_entropy:
                modeling_gemma.GemmaForCausalLM.forward = gemma_lce_forward

        elif cfg.model_config_type == "jamba":
            from transformers.models.jamba import modeling_jamba

            from .models.jamba import lce_forward as jamba_lce_forward

            if cfg.liger_rope:
                modeling_jamba.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_rms_norm:
                modeling_jamba.JambaRMSNorm = LigerRMSNorm
            if cfg.liger_swiglu:
                modeling_jamba.JambaMLP = LigerSwiGLUMLP
            if cfg.liger_cross_entropy:
                modeling_jamba.CrossEntropyLoss = LigerCrossEntropyLoss
            if cfg.liger_fused_linear_cross_entropy:
                modeling_jamba.JambaForCausalLM.forward = jamba_lce_forward

        elif cfg.model_config_type == "qwen2":
            from liger_kernel.transformers.model.qwen2 import (
                lce_forward as qwen2_lce_forward,
            )
            from transformers.models.qwen2 import modeling_qwen2

            if cfg.liger_rope:
                modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_rms_norm:
                modeling_qwen2.Qwen2RMSNorm = LigerRMSNorm
            if cfg.liger_swiglu:
                modeling_qwen2.Qwen2MLP = LigerSwiGLUMLP
            if cfg.liger_cross_entropy:
                modeling_qwen2.CrossEntropyLoss = LigerCrossEntropyLoss
            if cfg.liger_fused_linear_cross_entropy:
                modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward

        elif cfg.model_config_type == "deepseek_v2":
            from accelerate import init_empty_weights
            from transformers import AutoModelForCausalLM

            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.base_model, trust_remote_code=cfg.trust_remote_code or False
                )
                modeling_mod = sys.modules[model.__class__.__module__]

            from .models.deepseekv2 import lce_forward as deepseekv2_lce_forward

            if cfg.liger_rope:
                # The DeepseekV2 version of RoPE is different than upstream LLaMA.
                # See https://github.com/linkedin/Liger-Kernel/issues/129#issuecomment-2313763528
                logging.warning("Fused liger_rope is not supported for DeepseekV2.")
            if cfg.liger_rms_norm:
                modeling_mod.DeepseekV2RMSNorm = LigerRMSNorm
            if cfg.liger_swiglu:
                modeling_mod.DeepseekV2MLP.forward = LigerSwiGLUMLP.forward
            if cfg.liger_cross_entropy:
                modeling_mod.CrossEntropyLoss = LigerCrossEntropyLoss
            if cfg.liger_fused_linear_cross_entropy:
                modeling_mod.DeepseekV2ForCausalLM.forward = deepseekv2_lce_forward

        elif cfg.model_config_type == "gemma2":
            from transformers.models.gemma2 import modeling_gemma2

            if cfg.liger_rope:
                modeling_gemma2.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_rms_norm:
                modeling_gemma2.Gemma2RMSNorm = partial(
                    LigerRMSNorm, offset=1.0, init_fn="zeros", casting_mode="gemma"
                )
            if cfg.liger_swiglu:
                modeling_gemma2.Gemma2MLP = LigerGEGLUMLP
            if cfg.liger_cross_entropy:
                modeling_gemma2.CrossEntropyLoss = LigerCrossEntropyLoss
            if cfg.liger_fused_linear_cross_entropy:
                logging.warning(
                    "Fused linear cross entropy is not supported for Gemma 2."
                )

        elif cfg.model_config_type == "phi3":
            from liger_kernel.transformers.model.phi3 import (
                lce_forward as phi3_lce_forward,
            )
            from transformers.models.phi3 import modeling_phi3

            if cfg.liger_rope:
                modeling_phi3.apply_rotary_pos_emb = liger_rotary_pos_emb
            if cfg.liger_rms_norm:
                modeling_phi3.Phi3RMSNorm = LigerRMSNorm
            if cfg.liger_swiglu:
                modeling_phi3.Phi3MLP = LigerSwiGLUMLP
            if cfg.liger_cross_entropy:
                modeling_phi3.CrossEntropyLoss = LigerCrossEntropyLoss
            if cfg.liger_fused_linear_cross_entropy:
                modeling_phi3.Phi3ForCausalLM.forward = phi3_lce_forward
