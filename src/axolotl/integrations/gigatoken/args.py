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

"""Module for handling gigatoken input arguments."""

from pydantic import BaseModel, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class GigatokenArgs(BaseModel):
    """Input args for gigatoken."""

    gigatoken: bool = True

    @model_validator(mode="after")
    def check_gigatoken_installed(self):
        if self.gigatoken:
            try:
                import gigatoken  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "gigatoken is not installed. Please install it with "
                    "`pip install gigatoken`."
                ) from exc
        return self
