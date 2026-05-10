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

"""Shared constants for MixLoRA integration."""

MIXLORA_WEIGHTS_NAME = "mixlora_model.safetensors"
MIXLORA_FFN_MODULE_NAMES = ("gate_proj", "up_proj", "down_proj")

MIXLORA_DEFAULTS = {
    "mixlora_num_experts": 8,
    "mixlora_top_k": 2,
    "mixlora_router_aux_loss_coef": 0.01,
    "mixlora_router_init_range": 0.02,
    "mixlora_jitter_noise": 0.0,
}
