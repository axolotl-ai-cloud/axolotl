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

"""TorchSpec integration: train EAGLE-3 speculative-decoding draft models."""

from .args import TorchSpecArgs, TorchSpecArgsMixin
from .dataset_bridge import prepare_datasets, standardize_datasets
from .draft_config import apply_draft_overrides, build_draft_model_config
from .plugin import TorchSpecPlugin
from .translate import build_torchspec_args

__all__ = [
    "TorchSpecArgs",
    "TorchSpecArgsMixin",
    "TorchSpecPlugin",
    "apply_draft_overrides",
    "build_draft_model_config",
    "build_torchspec_args",
    "prepare_datasets",
    "standardize_datasets",
]
