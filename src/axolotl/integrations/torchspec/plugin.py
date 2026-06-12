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

from axolotl.integrations.base import BasePlugin


class TorchSpecPlugin(BasePlugin):
    """Registers the ``speculator:`` config schema for TorchSpec training.

    TorchSpec owns the entire run (Ray-orchestrated inference engines, Mooncake
    hidden-state transfer, and FSDP training workers), which is incompatible with
    axolotl's in-process HF-``Trainer`` lifecycle. Training is therefore launched
    via the dedicated CLI command rather than ``axolotl train``::

        axolotl train-speculator config.yaml

    This plugin exists so the ``speculator:`` block validates against the axolotl
    schema; it intentionally does not hook the normal model-load/trainer path.
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.torchspec.args.TorchSpecArgsMixin"
