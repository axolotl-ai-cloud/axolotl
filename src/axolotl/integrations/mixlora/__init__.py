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
MixLoRA integration: MoE-style LoRA finetuning of dense models.

Inserts multiple LoRA-based experts into FFN layers with a trainable router,
while keeping the base model frozen. Independent LoRA adapters are also added
to the attention layers via standard PEFT.

Reference: https://arxiv.org/abs/2404.15159
"""

from .patching import patch_model_with_mixlora
from .plugin import MixLoraPlugin

__all__ = ["patch_model_with_mixlora", "MixLoraPlugin"]
